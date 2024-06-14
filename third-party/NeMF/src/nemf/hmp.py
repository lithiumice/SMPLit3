import os

import holden.BVH as BVH
import numpy as np
import torch
import torch.nn as nn
from nemf_arguments import Arguments
from holden.Animation import Animation
from holden.Quaternions import Quaternions
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from nemf_rotations import matrix_to_quaternion, matrix_to_rotation_6d, quaternion_to_axis_angle, rotation_6d_to_matrix
from nemf_utils import align_joints, compute_trajectory, estimate_angular_velocity, estimate_linear_velocity, normalize

from .base_model import BaseModel
from .fk import ForwardKinematicsLayer
from .global_motion import GlobalMotionPredictor
from .losses import GeodesicLoss
from .neural_motion import NeuralMotionField
from .prior import GlobalEncoder, LocalEncoder
from .skeleton import build_edge_topology


class HMP(BaseModel):
    def __init__(self, args, ngpu, batch_num=0):
        super(HMP, self).__init__(args)

        self.batch_num = batch_num
        args.smpl.joint_num = 16
        args.local_prior.use_group_norm=False
        # import ipdb;ipdb.set_trace()
        
        self.args = args

        self.fk = ForwardKinematicsLayer(args)

        smpl_data = self.fk.smpl_data
        parents = smpl_data['kintree_table'][0].astype(np.int32)
        edges = build_edge_topology(parents)

        self.local_encoder = LocalEncoder(args.local_prior, edges).to(self.device)
        self.field = NeuralMotionField(args.nemf).to(self.device)
        if args.multi_gpu is True:
            self.local_encoder = nn.DataParallel(self.local_encoder, device_ids=range(ngpu))
            self.field = nn.DataParallel(self.field, device_ids=range(ngpu))

        self.models = [self.local_encoder, self.field]

        self.input_data = dict()
        self.recon_data = dict()

        self.std = torch.load(os.path.join(args.dataset_dir, f'train/std-{args.hand_type}-{args.data.clip_length}-{args.data.fps}fps.pt'), map_location=self.device)
        self.mean = torch.load(os.path.join(args.dataset_dir, f'train/mean-{args.hand_type}-{args.data.clip_length}-{args.data.fps}fps.pt'), map_location=self.device)
        print(f'mean: {list(self.mean.keys())}')
        print(f'std: {list(self.std.keys())}')

        self.criterion_geo = GeodesicLoss().to(self.device)
        if self.is_train:
            parameters = list(self.local_encoder.parameters()) + list(self.field.parameters())

            if args.adam_optimizer:
                self.optimizer = torch.optim.Adam(parameters, args.learning_rate, weight_decay=args.weight_decay)
            else:
                self.optimizer = torch.optim.Adamax(parameters, args.learning_rate, weight_decay=args.weight_decay)
            self.optimizers = [self.optimizer]

            self.criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()

            self.iteration = 0
        else:
            self.fps = args.data.fps
            self.test_index = 0
            if args.bvh_viz:
                self.viz_dir = os.path.join(args.save_dir, 'results', 'bvh')
            else:
                self.viz_dir = os.path.join(args.save_dir, 'results', 'smpl')

    def set_input(self, input):
        self.input_data = {k: v.float().to(self.device) for k, v in input.items() if k in ['pos', 'velocity', 'global_xform', 'angular',]}
        self.input_data['rotmat'] = rotation_6d_to_matrix(self.input_data['global_xform'])

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def encode_local(self):
        b_size, t_length = self.input_data['pos'].shape[:2]

        if 'pos' in self.args.data.normalize:
            pos = normalize(self.input_data['pos'], mean=self.mean['pos'], std=self.std['pos'])
        else:
            pos = self.input_data['pos']
        if 'velocity' in self.args.data.normalize:
            velocity = normalize(self.input_data['velocity'], mean=self.mean['velocity'], std=self.std['velocity'])
        else:
            velocity = self.input_data['velocity']
        if 'global_xform' in self.args.data.normalize:
            global_xform = normalize(self.input_data['global_xform'], mean=self.mean['global_xform'], std=self.std['global_xform'])
        else:
            global_xform = self.input_data['global_xform']
        if 'angular' in self.args.data.normalize:
            angular = normalize(self.input_data['angular'], mean=self.mean['angular'], std=self.std['angular'])
        else:
            angular = self.input_data['angular']
            
        # import ipdb;ipdb.set_trace()

        x = torch.cat((pos, velocity, global_xform, angular), dim=-1)
        x = x.view(b_size, t_length, -1)  # (B, T, J x D)
        x = x.permute(0, 2, 1)  # (B, J x D, T)

        mu, logvar = self.local_encoder(x)  # (B, D)
        if self.args.lambda_kl != 0:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        return z, mu, logvar

    def forward(self, step=1):
        z_l, mu_l, logvar_l = self.encode_local()  # (B, D)

        self.recon_data.clear()
        self.recon_data = self.decode(z_l, length=self.args.data.clip_length, step=step)

        self.recon_data['mu_l'] = mu_l
        self.recon_data['logvar_l'] = logvar_l


    def decode(self, z_l, length, step=1):
        b_size = z_l.shape[0]
        n_joints = self.args.smpl.joint_num

        # generate f(t; z)
        t = torch.arange(start=0, end=length, step=step).unsqueeze(0)  # (1, T)
        t = t / self.args.data.clip_length * 2 - 1  # map t to [-1, 1]
        t = t.expand(b_size, -1).unsqueeze(-1).to(self.device)  # (B, T, 1)
        local_motion, _ = self.field(t, z_l, None)  # (B, T, N)

        rot6d_recon = local_motion[:, :, :n_joints * 6].contiguous()  # (B, T, J x 6)
        rot6d_recon = rot6d_recon.view(b_size, -1, n_joints, 6)  # (B, T, J, 6)
        rotmat_recon = rotation_6d_to_matrix(rot6d_recon)  # (B, T, J, 3, 3)
        
    
        identity = torch.zeros(b_size, t.shape[1], 3, 3).to(self.device)
        identity[:, :, 0, 0] = 1
        identity[:, :, 1, 1] = 1
        identity[:, :, 2, 2] = 1
        rotmat_recon[:, :, 0] = identity
        
            
        rotmat_recon[:, :, -2] = rotmat_recon[:, :, -4]
        rotmat_recon[:, :, -1] = rotmat_recon[:, :, -3]

        local_rotmat = self.fk.global_to_local(rotmat_recon.view(-1, n_joints, 3, 3))  # (B x T, J, 3. 3)
        pos_recon, _ = self.fk(local_rotmat)  # (B x T, J, 3)
        pos_recon = pos_recon.contiguous().view(b_size, -1, n_joints, 3)  # (B, T, J, 3)

        output = dict()
        output['rotmat'] = rotmat_recon
        output['pos'] = pos_recon

        return output

    def kl_scheduler(self):
        """
        Cyclical Annealing Schedule
        """
        if (self.epoch_cnt % self.args.annealing_cycles == 0) and (self.iteration % self.batch_num == 0):
            print('KL annealing restart')
            return 0.01

        return min(1, 0.01 + self.iteration / (self.args.annealing_warmup * self.batch_num))

    def backward(self, validation=False):
        loss = 0

        if self.args.geodesic_loss:
            rotmat_recon_loss = self.criterion_geo(self.recon_data['rotmat'].view(-1, 3, 3), self.input_data['rotmat'].view(-1, 3, 3))
        else:
            rotmat_recon_loss = self.criterion_rec(self.recon_data['rotmat'], self.input_data['rotmat'])
        self.loss_recorder.add_scalar('rotmat_recon_loss', rotmat_recon_loss, validation=validation)
        loss += self.args.lambda_rotmat * rotmat_recon_loss

        pos_recon_loss = self.criterion_rec(self.recon_data['pos'], self.input_data['pos'])
        self.loss_recorder.add_scalar('pos_recon_loss', pos_recon_loss, validation=validation)
        loss += self.args.lambda_pos * pos_recon_loss

        if self.args.lambda_kl != 0:
            mu_l = self.recon_data['mu_l']
            logvar_l = self.recon_data['logvar_l']
            local_kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar_l - mu_l.pow(2) - logvar_l.exp(), dim=-1))
            self.loss_recorder.add_scalar('local_kl_loss', local_kl_loss, validation=validation)
            global_kl_loss = 0
            if not validation:
                loss += self.args.lambda_kl * self.kl_scheduler() * (local_kl_loss + global_kl_loss)

        self.loss_recorder.add_scalar('total_loss', loss, validation=validation)

        if not validation:
            loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward(validation=False)
        self.optimizer.step()

        self.iteration += 1

    def report_errors(self):
        local_rotmat = self.fk.global_to_local(self.recon_data['rotmat'].view(-1, self.args.smpl.joint_num, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat_gt = self.fk.global_to_local(self.input_data['rotmat'].view(-1, self.args.smpl.joint_num, 3, 3))  # (B x T, J, 3, 3)
        rotation_error = self.criterion_geo(local_rotmat[:, 1:].reshape(-1, 3, 3), local_rotmat_gt[:, 1:].reshape(-1, 3, 3))

        pos = self.recon_data['pos']  # (B, T, J, 3)
        pos_gt = self.input_data['pos']  # (B, T, J, 3)
        position_error = torch.linalg.norm((pos - pos_gt), dim=-1).mean()

        return {
            'rotation': c2c(rotation_error),
            'position': c2c(position_error),
        }

    def verbose(self):
        res = {}
        for loss in self.loss_recorder.losses.values():
            res[loss.name] = {'train': loss.current()[0], 'val': loss.current()[1]}

        return res

    def validate(self):
        with torch.no_grad():
            self.forward()
            self.backward(validation=True)

    def save(self, optimal=False):
        if optimal:
            path = os.path.join(self.args.save_dir, 'results', 'model')
        else:
            path = os.path.join(self.model_save_dir, f'{self.epoch_cnt:04d}')

        os.makedirs(path, exist_ok=True)
        local_encoder = self.local_encoder.module if isinstance(self.local_encoder, nn.DataParallel) else self.local_encoder
        nemf = self.field.module if isinstance(self.field, nn.DataParallel) else self.field
        torch.save(local_encoder.state_dict(), os.path.join(path, 'local_encoder.pth'))
        torch.save(nemf.state_dict(), os.path.join(path, 'nemf.pth'))

        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pth'))
        if self.args.scheduler.name:
            torch.save(self.schedulers[0].state_dict(), os.path.join(path, 'scheduler.pth'))
        self.loss_recorder.save(path)

        print(f'Save at {path} succeeded')

    def load(self, epoch=None, optimal=False):
        if optimal:
            path = os.path.join(self.args.save_dir, 'results', 'model')
        else:
            if epoch is None:
                all = [int(q) for q in os.listdir(self.model_save_dir)]
                if len(all) == 0:
                    raise RuntimeError(f'Empty loading path {self.model_save_dir}')
                epoch = sorted(all)[-1]
            path = os.path.join(self.model_save_dir, f'{epoch:04d}')

        # print(f'Loading from {path}')
        local_encoder = self.local_encoder.module if isinstance(self.local_encoder, nn.DataParallel) else self.local_encoder
        nemf = self.field.module if isinstance(self.field, nn.DataParallel) else self.field
        local_encoder.load_state_dict(torch.load(os.path.join(path, 'local_encoder.pth'), map_location=self.device))
        nemf.load_state_dict(torch.load(os.path.join(path, 'nemf.pth'), map_location=self.device))

        if self.is_train:
            self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pth')))
            self.loss_recorder.load(path)
            self.iteration = len(self.loss_recorder.losses['total_loss'].loss_step)
            print(f'trained iterations: {self.iteration}')
            if self.args.scheduler.name:
                self.schedulers[0].load_state_dict(torch.load(os.path.join(path, 'scheduler.pth')))
        self.epoch_cnt = epoch if not optimal else 0

        # print('Load succeeded')

    def super_sampling(self, step=1):
        pass

    def compute_test_result(self):
        pass