import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainDecompOptions
from utils.plot_script import *

import networks.modules as net_modules
from networks.modules import *
from networks.trainers import DecompTrainerDisc
from data.dataset import StyleMotionDataset
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator

from tqdm import tqdm
import pandas
import joblib

import sys;sys.path.insert(0,'/apdcephfs/private_wallyliang/PLANT/difftraj')
from eval.smpl_utils import AnyRep2SMPLjoints
traj2joints = AnyRep2SMPLjoints()

def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        if i>10: break
        # import ipdb;ipdb.set_trace()
        denormed_traj_motion = torch.from_numpy(data[i])
        global_pose = traj2joints.traj_to_joints(denormed_traj_motion, zup_to_yup = True)    
        global_pose = global_pose.cpu().numpy() 
        
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        plot_3d_motion(save_path, kinematic_chain, global_pose, title="None", fps=fps, radius=4)


if __name__ == '__main__':
    parser = TrainDecompOptions()
    opt = parser.parse()
    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1: torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    opt.data_root = ''
    opt.motion_dir = ''
    opt.text_dir = ''
    opt.joints_num = 22
    dim_pose = opt.dim_pose
    fps = opt.ex_fps
    kinematic_chain = paramUtil.t2m_kinematic_chain
    # import ipdb;ipdb.set_trace()
    if opt.ae_type == 'ae3':
        movement_enc = MovementConvEncoder3(opt, dim_pose, 256, 256)
        movement_dec = MovementConvDecoder3(opt, 256, 256, dim_pose)
    if opt.ae_type == 'ae2':
        movement_enc = MovementConvEncoder2(dim_pose, 256, 256)
        movement_dec = MovementConvDecoder2(256, 256, dim_pose)
    if opt.ae_type == 'aeLarge':
        movement_enc = MovementConvEncoderLarge(opt, dim_pose, 1024, 1024, 768)
        movement_dec = MovementConvDecoderLarge(opt, 1024, 1024, dim_pose, 768)
    if opt.ae_type == 'aeZeroEGGs':
        from networks.zeroeggs_modules import *
        movement_enc = StyleEncoder(203, 256, 512, use_vae = True)
        inp = torch.randn(3,60,203)
        emb, mu, logvar = movement_enc(inp)
        import ipdb;ipdb.set_trace()
        movement_dec = StyleDecoder(203, 256, 512)
        rec = movement_dec(emb)
    if opt.ae_type == 'aeTF' or opt.ae_type == 'aeTFUniRep':
        from networks.transformer import *
        from my_constant import *
        parameters = motionclip_ae_param
        if opt.ae_type == 'aeTFUniRep':
            parameters['njoints'] = 1
            parameters['nfeats'] = 203
            parameters['latent_dim'] = 512
        movement_enc = Encoder_TRANSFORMER(**parameters)
        movement_dec = Decoder_TRANSFORMER(**parameters)
    if opt.ae_type == 'actorVAE':
        from networks.actor_vae import *
        movement_enc = ActorAgnosticEncoder(nfeats = 203, latent_dim = 512, ff_size = 1024, num_layers = 6, num_heads = 4, dropout = 0.1, activation = 'gelu', vae = True)
        movement_dec = ActorAgnosticDecoder(nfeats = 203, latent_dim = 512, ff_size = 1024, num_layers = 6, num_heads = 4, dropout = 0.1, activation = 'gelu',)
        
    all_params = 0
    pc_mov_enc = sum(param.numel() for param in movement_enc.parameters())
    print(movement_enc)
    print("Total parameters of prior net: {}".format(pc_mov_enc))
    all_params += pc_mov_enc

    pc_mov_dec = sum(param.numel() for param in movement_dec.parameters())
    print(movement_dec)
    print("Total parameters of posterior net: {}".format(pc_mov_dec))
    all_params += pc_mov_dec

    train_dataset = StyleMotionDataset(opt, 'train')
    val_dataset = StyleMotionDataset(opt, 'test')    
    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=num_workers,shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=num_workers,shuffle=True, pin_memory=True)
    # import ipdb;ipdb.set_trace() 
    trainer = DecompTrainerDisc(opt, movement_enc, movement_dec)
    trainer.train(train_loader, val_loader, plot_t2m)
