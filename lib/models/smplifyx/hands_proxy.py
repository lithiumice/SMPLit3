
import os;
import sys;

"""
NeMF dependence
"""
NeMF_path = 'third-party/NeMF'
sys.path.insert(0,f'{NeMF_path}/src')
from NeMF.src.nemf_arguments import Arguments
from NeMF.src.nemf.fk import ForwardKinematicsLayer
from NeMF.src.nemf.generative import Architecture
from NeMF.src.nemf.hmp import HMP
from NeMF.src.nemf_utils import estimate_angular_velocity, estimate_linear_velocity

import torch

torch.set_default_dtype(torch.float32)

from lib.loco.trajdiff import *
  


class HandsProxy():
    def __init__(self):
        with path_enter(NeMF_path):
            # import ipdb;ipdb.set_trace()
            # <========= Left Hand
            L_nemf_args = Arguments(f'configs', filename='hmp_L.yaml')
            L_nemf_args.data.root_transform = False
            L_nemf_args.data.overlap_len = 16
            
            L_nemf_fk = ForwardKinematicsLayer(L_nemf_args)
            
            L_nemf_model = HMP(L_nemf_args, 1)
            L_nemf_model.load(optimal=True)
            L_nemf_model.eval()    
            
            self.L_nemf_fk = L_nemf_fk
            self.L_nemf_args = L_nemf_args
            self.L_nemf_model = L_nemf_model
            
            self.clip_length = self.L_nemf_args.data.clip_length
            self.overlap_len = self.L_nemf_args.data.overlap_len
            self.dt=1.0 / self.L_nemf_args.data.fps
            
            
            # <========= Right Hand
            self.R_nemf_fk = self.L_nemf_fk
            self.R_nemf_args = self.L_nemf_args
            self.R_nemf_model = self.L_nemf_model
                        
        
        self.L_hands_mean = toth(self.L_nemf_fk.smpl_data['hands_mean'].reshape(15,3)).cuda().float()
        self.R_hands_mean = toth(self.L_nemf_fk.smpl_data['hands_mean'].reshape(15,3)).cuda().float()
                    

    def prepare_nemf_inp_data(self, hand_pose, fk):
        """
        hand_pose: T,15,3
        """        
        # import ipdb;ipdb.set_trace()
        poses = torch.cat([torch.zeros(hand_pose.shape[0],1,3).type_as(I), hand_pose],dim=1).reshape(-1,16*3)
        pose = split_tensor(poses, self.clip_length, self.overlap_len)
        pose = pose.view(-1, self.clip_length, 16, 3)  # axis-angle (N, T, J, 3)
        rotmat = a2m(pose)  # rotation matrix (N, T, J, 3, 3)
        
        identity = torch.zeros_like(rotmat[:, :, 0])  # (N, T, 3, 3)
        identity[:, :, 0, 0] = 1
        identity[:, :, 1, 1] = 1
        identity[:, :, 2, 2] = 1
        
        rotmat[:, :, 0] = identity
        rot6d = m2s(rotmat)  # 6D rotation representation (N, T, J, 6)
        pos, global_xform = fk(rot6d.view(-1, 16, 6))
        pos = pos.contiguous().view(-1, self.clip_length, 16, 3)  # local joint positions (N, T, J, 3)
        
        velocity = estimate_linear_velocity(pos, dt=self.dt)  # linear velocity of all the joints (N, T, J, 3)
        angular = estimate_angular_velocity(rotmat.clone(), dt=self.dt)  # angular velocity of all the joints (N, T, J, 3)
        
        global_xform = global_xform.view(-1, self.clip_length, 16, 4, 4)  # global transformation matrix for each joint (N, T, J, 4, 4)
        global_xform = global_xform[:, :, :, :3, :3]  # (N, T, J, 3, 3)
        global_xform = m2s(global_xform)  # (N, T, J, 6)
        
        nemf_input_data = dict(
            pos=pos,
            velocity=velocity,
            global_xform=global_xform,
            angular=angular,
        )
        """
        pos, velocity, global_xform, angular
        """
        return nemf_input_data
        
    def forward(self):
        pass
    
    def enc(self, pose):
        """
        pose: [1, T, 2, 15, 6])
        """
        # import ipdb;ipdb.set_trace()

        pose = s2a(pose.squeeze().reshape(-1,2,15,6)) #[T, 2, 15, 3]
        org_T = pose.shape[0]
        
        Lh_pose  = pose[:,0,:,:]#T,15,3
        Rh_pose  = pose[:,1,:,:]
        
        self.Lh_pose_org = Lh_pose.clone()
        
        def enc_var(org_pose, model, fk, mean):
            true_pose = org_pose + mean
            model.set_input(self.prepare_nemf_inp_data(true_pose, fk))
            z_var, _, _ = model.encode_local()
            return z_var
        
        z_hL = enc_var(Lh_pose, self.L_nemf_model, self.L_nemf_fk, self.L_hands_mean)
        z_hR = enc_var(Rh_pose, self.R_nemf_model, self.R_nemf_fk, self.R_hands_mean)
            
        # Lh_pose = Lh_pose + self.L_hands_mean
        # Rh_pose = Rh_pose + self.R_hands_mean
        
        # self.L_nemf_model.set_input(self.prepare_nemf_inp_data(Lh_pose, self.L_nemf_fk))
        # z_hL, _, _ = self.L_nemf_model.encode_local()
        
        # self.R_nemf_model.set_input(self.prepare_nemf_inp_data(Rh_pose, self.R_nemf_fk))
        # z_hR, _, _ = self.R_nemf_model.encode_local()
        
        # import ipdb;ipdb.set_trace()
        
        proxy_var = self._proxy_var = {
            'org_T': org_T,
            'cvt_vars': {
                'z_hL': z_hL,
                'z_hR': z_hR,
            }
        }
        return self._proxy_var
    
    def dec(self, proxy_var):
        """
        B, 512
        """
        org_T = self._proxy_var['org_T']
        overlap_len = self.overlap_len

        def dec_var(z_var, model, fk, mean):
            nemf_output = model.decode(z_var, length=self.clip_length, step=1.0)
            rotmat = nemf_output['rotmat']  # (B, T, J, 3, 3)
            b_size, _, n_joints = rotmat.shape[:3]
            local_rotmat = fk.global_to_local(rotmat.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
            local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)
            local_rotmat = merge_tensor_w_avg((m2s(local_rotmat)), overlap_len, org_T)# (T, J, 3)   
            rec_pose = local_rotmat[...,1:,:]#T,16,6
            rec_pose = a2s(s2a(rec_pose)- mean)            
            return rec_pose
        
        # import ipdb;ipdb.set_trace()
        
        Lh_rec_pose = dec_var(proxy_var['cvt_vars']['z_hL'], self.L_nemf_model, self.L_nemf_fk, self.L_hands_mean)
        Rh_rec_pose = dec_var(proxy_var['cvt_vars']['z_hR'], self.R_nemf_model, self.R_nemf_fk, self.R_hands_mean)
        
        # (a2s(self.Lh_pose_org)-Lh_rec_pose)
        
        rec_pose = torch.stack([Lh_rec_pose, Rh_rec_pose],dim=1).unsqueeze(0)
        return rec_pose
    
    def get_proxy(self):
        return self._proxy_var
    