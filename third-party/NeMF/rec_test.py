
from argparse import ArgumentParser
import glob
import itertools
import os
import pickle
import random
import subprocess
import glob
import time
from datetime import datetime
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

# import os;os.chdir('/apdcephfs/private_wallyliang/PLANT/NeMF')
import sys;sys.path.insert(0,'/apdcephfs/private_wallyliang/PLANT/Thirdparty')
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import log2file, makepath

# from datasets.amass import AMASS
# from torch.utils.data import Dataset
import sys;sys.path.insert(0,'/apdcephfs/private_wallyliang/PLANT/NeMF/src')
from nemf.fk import ForwardKinematicsLayer
from arguments import Arguments
from nemf.generative import Architecture
from nemf.losses import GeodesicLoss
from nemf_rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, matrix_to_rotation_6d, quaternion_to_matrix, rotation_6d_to_matrix
from soft_dtw_cuda import SoftDTW
import holden.BVH as BVH
from holden.Animation import Animation
from holden.Quaternions import Quaternions
# from nemf_utils import CONTACTS_IDX, align_joints, build_canonical_frame, estimate_angular_velocity, estimate_linear_velocity, normalize
# from nemf_utils import build_canonical_frame, compute_orient_angle, estimate_angular_velocity, estimate_linear_velocity, export_ply_trajectory, slerp

"""
pip list|grep arguments
python /apdcephfs/private_wallyliang/PLANT/NeMF/rec_test.py
"""

args = Arguments('/apdcephfs/private_wallyliang/PLANT/NeMF/configs', filename='application.yaml')
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
ngpu = 1
# if args.multi_gpu is True:
#     ngpu = torch.cuda.device_count()
#     if ngpu == 1:
#         args.multi_gpu = False
# print(f'Number of GPUs: {ngpu}')
# dataset = AMASS(dataset_dir=os.path.join(args.dataset_dir, 'test'))
# data_loader = DataLoader(dataset, batch_size=ngpu * args.batch_size, num_workers=args.num_workers, shuffle=False,pin_memory=True, drop_last=False)
model = Architecture(args, ngpu)
model.load(optimal=True)
model.eval()
fk = ForwardKinematicsLayer(args)
# run_motion_reconstruction()
offset = 0

npz_fname='/root/apdcephfs/private_wallyliang/PLANT_tests/WuDangQuan/WuDangQuan_test_True_adam_st0-et1068_res_0_1068_ID1.npz'
cdata = np.load(npz_fname)
poses = cdata['poses']
N = len(poses)
root_orient = poses[:,0,:]
pose_body = poses[:,1:22,:].reshape(-1,63)
# contacts = cdata['contacts']
trans = cdata['trans']

device = 'cuda'
poses = np.concatenate((root_orient, pose_body, np.zeros((N, 6))), axis=1)
poses = torch.from_numpy(poses).to(device).to(torch.float)
trans = torch.from_numpy(trans).to(device).to(torch.float)



def split_array(array, clip_length, overlap_length):
    assert array.ndim >= 1, "Array should have at least one dimension"
    assert clip_length > overlap_length, "Clip length should be greater than overlap length"
    step = clip_length - overlap_length
    splits = []
    for i in range(0, array.shape[0] - clip_length + 1, step):
        # splits.append(tensor.narrow(0, i, clip_length))
        splits.append(array[i:i+clip_length])
    if array.shape[0] % step != 0:
        last_split = array[array.shape[0] - clip_length:]
        # last_split = tensor.narrow(0, tensor.size(0) - clip_length, clip_length)
        splits.append(last_split)
    return torch.stack(splits)

def merge_arrays(splits, overlap_length, N):
    step = splits[0].shape[0] - overlap_length
    merged_array = torch.zeros((N, *splits.shape[2:])).type_as(splits)
    # merged_array = torch.zeros_like(splits)[:N]
    for i, split in enumerate(splits):
        start_idx = i * step
        end_idx = start_idx + split.shape[0]
        if end_idx <= N:
            merged_array[start_idx:end_idx] = split
        else:
            merged_array[-(N-start_idx):] = split[-(N-start_idx):]
    return merged_array


clip_frames = args.data.clip_length
pose = split_array(poses, clip_frames, 16)
trans = split_array(trans, clip_frames, 16)# global translation (N, T, 3)
pose = pose.view(-1, clip_frames, 24, 3)  # axis-angle (N, T, J, 3)
# Compute necessary data for model training.
rotmat = axis_angle_to_matrix(pose)  # rotation matrix (N, T, J, 3, 3)
root_orient = rotmat[:, :, 0].clone()
root_orient = matrix_to_rotation_6d(root_orient)  # root orientation (N, T, 6)
# if args.unified_orientation:
identity = torch.zeros_like(rotmat[:, :, 0])  # (N, T, 3, 3)
identity[:, :, 0, 0] = 1
identity[:, :, 1, 1] = 1
identity[:, :, 2, 2] = 1
rotmat[:, :, 0] = identity
rot6d = matrix_to_rotation_6d(rotmat)  # 6D rotation representation (N, T, J, 6)
rot_seq = rotmat.clone()
angular = estimate_angular_velocity(rot_seq, dt=1.0 / args.data.fps)  # angular velocity of all the joints (N, T, J, 3)
fk = ForwardKinematicsLayer(args, device=device)
pos, global_xform = fk(rot6d.view(-1, 24, 6))
pos = pos.contiguous().view(-1, clip_frames, 24, 3)  # local joint positions (N, T, J, 3)
global_xform = global_xform.view(-1, clip_frames, 24, 4, 4)  # global transformation matrix for each joint (N, T, J, 4, 4)
velocity = estimate_linear_velocity(pos, dt=1.0 / args.data.fps)  # linear velocity of all the joints (N, T, J, 3)
# contacts = torch.from_numpy(np.asarray(data_contacts, np.float32)).to(device)
# contacts = contacts[:, :, CONTACTS_IDX]  # contacts information (N, T, 8)
root_rotation = rotation_6d_to_matrix(root_orient)  # (N, T, 3, 3)
root_rotation = root_rotation.unsqueeze(2).repeat(1, 1, args.smpl.joint_num, 1, 1)  # (N, T, J, 3, 3)
global_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
height = global_pos + trans.unsqueeze(2)
height = height[:, :, :, 'xyz'.index(args.data.up)]  # (N, T, J)
root_vel = estimate_linear_velocity(trans, dt=1.0 / args.data.fps)  # linear velocity of the root joint (N, T, 3)
global_xform = global_xform[:, :, :, :3, :3]  # (N, T, J, 3, 3)
global_xform = matrix_to_rotation_6d(global_xform)  # (N, T, J, 6)

data = dict(
    rotmat=rotmat,
    pos=pos,
    trans=trans,
    root_vel=root_vel,
    height=height,
    rot6d=rot6d,
    angular=angular,
    global_xform=global_xform,
    velocity=velocity,
    root_orient=root_orient,
)

model.set_input(data)
target = dict()
target['pos'] = data['pos'].to(model.device)
target['rotmat'] = rotation_6d_to_matrix(data['global_xform'].to(model.device))
target['trans'] = data['trans'].to(model.device)
target['root_orient'] = data['root_orient'].to(model.device)

z_l, _, _ = model.encode_local()
if args.data.root_transform:
    z_g, _, _ = model.encode_global()
    
# def L_rot(source, target, T):
#     """
#     Args:
#         source, target: rotation matrices in the shape B x T x J x 3 x 3.
#         T: temporal masks.

#     Returns:
#         reconstruction loss evaluated on the rotation matrices.
#     """
#     criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
#     criterion_geo = GeodesicLoss()

#     if args.geodesic_loss:
#         loss = criterion_geo(source[:, T].view(-1, 3, 3), target[:, T].view(-1, 3, 3))
#     else:
#         loss = criterion_rec(source[:, T], target[:, T])

#     return loss


# def L_pos(source, target, T):
#     """
#     Args:
#         source, target: joint local positions in the shape B x T x J x 3.
#         T: temporal masks.

#     Returns:
#         reconstruction loss evaluated on the joint local positions.
#     """
#     criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
#     loss = criterion_rec(source[:, T], target[:, T])

#     return loss


# def L_orient(source, target, T):
#     """
#     Args:
#         source: predicted root orientation in the shape B x T x 6.
#         target: root orientation in the shape of B x T x 6.
#         T: temporal masks.

#     Returns:
#         reconstruction loss evaluated on the root orientation.
#     """
#     criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
#     criterion_geo = GeodesicLoss()

#     source = rotation_6d_to_matrix(source)  # (B, T, 3, 3)
#     target = rotation_6d_to_matrix(target)  # (B, T, 3, 3)

#     if args.geodesic_loss:
#         loss = criterion_geo(source[:, T].view(-1, 3, 3), target[:, T].view(-1, 3, 3))
#     else:
#         loss = criterion_rec(source[:, T], target[:, T])

#     return loss


# def L_trans(source, target, T, up=True):
#     """
#     Args:
#         source: predict global translation in the shape B x T x 3 (the origin is (0, 0, height)).
#         target: global translation of the root joint in the shape B x T x 3.
#         T: temporal masks.

#     Returns:
#         reconstruction loss evaluated on the global translation.
#     """
#     criterion_pred = nn.L1Loss() if args.l1_loss else nn.MSELoss()

#     origin = target[:, 0]
#     trans = source
#     trans[..., model.v_axis] = trans[..., model.v_axis] + origin[..., model.v_axis].unsqueeze(1)
#     trans_gt = target

#     if not up:
#         trans = trans[..., model.v_axis]
#         trans_gt = trans_gt[..., model.v_axis]

#     loss = criterion_pred(trans[:, T], trans_gt[:, T])

#     return loss


def latent_optimization(args, target, T=None, z_l=None, z_g=None):
# def latent_optimization(target, T=None, z_l=None, z_g=None):
    """
        Slove the latent optimization problem to minimize the reconstruction loss.
    """
    if T is None:
        T = torch.arange(args.data.clip_length)
    if z_l is None:
        z_l = Variable(torch.randn(target['rotmat'].shape[0], args.function.local_z).to(model.device), requires_grad=True)
    else:
        z_l = Variable(z_l, requires_grad=True)
    if z_g is None:
        if args.data.root_transform:  # optimize both z_l and z_g
            z_g = Variable(torch.randn(target['rotmat'].shape[0], args.function.global_z).to(model.device), requires_grad=True)
            optimizer = torch.optim.Adam([z_l, z_g], lr=args.learning_rate)
        else:  # optimize z_l ONLY
            optimizer = torch.optim.Adam([z_l], lr=args.learning_rate)
    else:
        z_g = Variable(z_g, requires_grad=True)
        optimizer = torch.optim.Adam([z_l, z_g], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler.step_size, args.scheduler.gamma, verbose=False)

    start_time = time.time()
    for i in range(args.iterations):
        optimizer.zero_grad()
        output = model.decode(z_l, z_g, length=args.data.clip_length, step=1)

        # evaluate the objective function
        rot_loss = L_rot(output['rotmat'], target['rotmat'], T)
        pos_loss = L_pos(output['pos'], target['pos'], T)
        orient_loss = L_orient(output['root_orient'], target['root_orient'], T)
        trans_loss = L_trans(output['trans'], target['trans'], T, up=True)

        loss = args.lambda_rot * rot_loss + args.lambda_pos * pos_loss + args.lambda_orient * orient_loss + args.lambda_trans * trans_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        print('[{:03d}/{}] rot_loss: {:.4f}\t pos_loss: {:.4f}\t orient_loss: {:.4f}\t trans_loss: {:.4f}\t loss: {:.4f}'.format(
            i, args.iterations, rot_loss.item(), pos_loss.item(), orient_loss.item(), trans_loss.item(), loss.item()))

    end_time = time.time()
    print(f'Optimization finished in {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')

    return z_l, z_g
# from application import latent_optimization
# import importlib
# import application.latent_optimization
# importlib.reload(latent_optimization)
z_l, z_g = latent_optimization(args, target, T=None, z_l=z_l, z_g=z_g)

step=1.0
with torch.no_grad():
    output = model.decode(z_l, z_g, length=args.data.clip_length, step=step)

# fps = int(args.data.fps / step)
# criterion_geo = GeodesicLoss()

rotmat = output['rotmat']  # (B, T, J, 3, 3)
b_size, _, n_joints = rotmat.shape[:3]
local_rotmat = fk.global_to_local(rotmat.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)
root_orient = rotation_6d_to_matrix(output['root_orient'])  # (B, T, 3, 3)
# root_orient_gt = rotation_6d_to_matrix(target['root_orient'])  # (B, T, 3, 3)
if args.data.root_transform:
    local_rotmat[:, :, 0] = root_orient

# pos = output['pos']  # (B, T, J, 3)
# origin = target['trans'][:, 0]
trans = output['trans']
# trans[..., model.v_axis] = trans[..., model.v_axis] + origin[..., model.v_axis].unsqueeze(1)
# trans_gt = target['trans']  # (B, T, 3)

for i in range(1,len(trans)):
    trans[i,:,model.v_axis]+=trans[i-1,-1,model.v_axis]
trans = merge_arrays((trans), 16, N)# (T, J, 3)
trans = c2c(trans)
    
    
poses = merge_arrays((matrix_to_axis_angle(local_rotmat)), 16, N)# (T, J, 3)
poses = c2c(poses)
poses = poses.reshape((poses.shape[0], -1))  # (T, 66)
poses = np.pad(poses, [(0, 0), (0, 93)], mode='constant')
np.savez(f'test2.npz',poses=poses, trans=trans, betas=np.zeros(10), gender=args.data.gender, mocap_framerate=30)
    