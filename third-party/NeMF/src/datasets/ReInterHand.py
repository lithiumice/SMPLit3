# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# AMASS: Archive of Motion Capture as Surface Shapes <https://arxiv.org/abs/1904.03278>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2019.08.09
import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '..'))

import glob
from datetime import datetime

import holden.BVH as BVH
import numpy as np
import torch
from nemf_arguments import Arguments
from holden.Animation import Animation
from holden.Quaternions import Quaternions
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import log2file, makepath
from nemf.fk import ForwardKinematicsLayer
from nemf_rotations import axis_angle_to_matrix, axis_angle_to_quaternion, matrix_to_rotation_6d, rotation_6d_to_matrix
from torch.utils.data import Dataset
from tqdm import tqdm
from nemf_utils import CONTACTS_IDX, align_joints, build_canonical_frame, estimate_angular_velocity, estimate_linear_velocity, normalize
from pathlib import Path
import json

def dump_amass2pytroch(out_posepath, logger=None):
    makepath(out_posepath, isfile=True)

    if logger is None:
        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        logger = log2file(out_posepath.replace(f'pose-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt', '%s.log' % (log_name)))
        logger('Creating pytorch dataset at %s' % out_posepath)



    # <=================== LOAD data
    data_poses = []
    clip_frames = args.data.clip_length
    
    def fill_data(pose):
        N = pose.shape[0]
        nclips = [np.arange(i, i + clip_frames) for i in range(0, N, clip_frames)]
        if N % clip_frames != 0:
            nclips.pop()
        for clip in nclips:
            data_poses.append(pose[clip])     
    
    
    # <=== INTERHAND 1
    # interhand_mano_anno_data = json.load(open('/apdcephfs/share_1330077/wallyliang//InterHand2.6M_train_MANO_NeuralAnnot.json','r'))
    # for seq_id in interhand_mano_anno_data.keys():
    #     seq_data = interhand_mano_anno_data[seq_id]
    #     sort_seq_data = sorted(seq_data.items(), key=lambda x: int(x[0]))
        
    #     if args.hand_type == 'L':
    #         # if sort_seq_data[0][1]['left'] is None: print(f'id {seq_id} has not left hand'); continue
    #         pose = np.array([i[1]['left']['pose'] for i in sort_seq_data if i[1]['left'] is not None])
    #     if args.hand_type == 'R':
    #         # if sort_seq_data[0][1]['right'] is None: print(f'id {seq_id} has not left hand'); continue
    #         pose = np.array([i[1]['right']['pose'] for i in sort_seq_data if i[1]['right'] is not None])
                        
    #     fill_data(pose)
    # print(f'END process InterHand2.6M')
        

        
    # # <====== ReInterHand
    # for capture_id_dir in tqdm(ReInterHand_data_root.iterdir()):
    #     if args.hand_type == 'L':
    #         json_path = sorted([pth for pth in capture_id_dir.rglob("params/*_left.json")], key=lambda x: int(x.stem.split('_')[0]))
    #     if args.hand_type == 'R':
    #         json_path = sorted([pth for pth in capture_id_dir.rglob("params/*_right.json")], key=lambda x: int(x.stem.split('_')[0]))
        
    #     if len(json_path)==0: continue
    #     # import ipdb;ipdb.set_trace()
    #     pose = np.array([json.load(open(str(p),'r'))['pose'] for p in json_path]).reshape(-1,16,3)
    #     # Chop the data into evenly splitted sequence clips.
    #     fill_data(pose)
    #     # break

    # assert len(data_poses) != 0
    # print(f'END process InterHand-RE')
    # print(f'Total {len(data_poses)} poses')
        
        
    # import ipdb;ipdb.set_trace()
        
    # <===== GRAB dataset
    amass_dir = '/apdcephfs/share_1330077/dataset/amass_raw/'
    datasets = ['GRAB', 'TCD_handMocap']
    for ds_name in datasets:
        npz_fnames = glob.glob(os.path.join(amass_dir, ds_name, '*/*.npz'))
        logger('processing data points from %s.' % (ds_name))
        for npz_fname in tqdm(npz_fnames):
            cdata = np.load(npz_fname,allow_pickle=True)
            cdata = dict(cdata)
            
            try:
                mocap_fps = 30
                if 'mocap_frame_rate' in cdata:
                    mocap_fps = cdata['mocap_frame_rate']
                elif 'mocap_framerate' in cdata:
                    mocap_fps = cdata['mocap_framerate']
                    
                down_rate = int(mocap_fps/args.data.fps)
                pose = cdata['poses'].reshape(-1,52,3)[::down_rate]
                pose = pose[:,-30:,:]
                if args.hand_type == 'L':
                    pose = pose[:,:15]
                if args.hand_type == 'R':
                    pose = pose[:,15:]
                pose = np.concatenate([np.zeros_like(pose[:,:1,:]),pose],axis=1)
            except:
                # import ipdb;ipdb.set_trace()   
                continue 
            
            fill_data(pose)
                
                        
    print(f'Total {len(data_poses)} poses')
        
        

    # <=================== parse data
    # Compute necessary data for model training.
    # Choose the device to run the body model on.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")  # when GPU memory is limited
    pose = torch.from_numpy(np.asarray(data_poses, np.float32)).to(device).view(-1, clip_frames, 16, 3)  # axis-angle (N, T, J, 3)
    rotmat = axis_angle_to_matrix(pose)  # rotation matrix (N, T, J, 3, 3)
    
    # normal orientation
    identity = torch.zeros_like(rotmat[:, :, 0])  # (N, T, 3, 3)
    identity[:, :, 0, 0] = 1
    identity[:, :, 1, 1] = 1
    identity[:, :, 2, 2] = 1
    rotmat[:, :, 0] = identity
    # identity = matrix_to_rotation_6d(identity)
    # tmp = torch.cat([identity.view(-1, 1, 6), rot6d.view(-1, 15, 6)], dim=1)

    angular = estimate_angular_velocity(rotmat.clone(), dt=1.0 / args.data.fps)  # angular velocity of all the joints (N, T, J, 3)

    rot6d = matrix_to_rotation_6d(rotmat)  # 6D rotation representation (N, T, J, 6)

    # forward kinamatic
    # import ipdb;ipdb.set_trace()
    fk = ForwardKinematicsLayer(args, device=device)
    pos, global_xform = fk(rot6d.view(-1, 16, 6))
    pos = pos.contiguous().view(-1, clip_frames, 16, 3)  # local joint positions (N, T, J, 3)
    global_xform = global_xform.view(-1, clip_frames, 16, 4, 4)  # global transformation matrix for each joint (N, T, J, 4, 4)
    velocity = estimate_linear_velocity(pos, dt=1.0 / args.data.fps)  # linear velocity of all the joints (N, T, J, 3)
    # no camonical
    global_xform = global_xform[:, :, :, :3, :3]  # (N, T, J, 3, 3)
    global_xform = matrix_to_rotation_6d(global_xform)  # (N, T, J, 6)
    # <=================== 



    # <=================== save data
    torch.save(rotmat.detach().cpu(), out_posepath.replace('pose', 'rotmat'))  # (N, T, J, 3, 3)
    torch.save(pos.detach().cpu(), out_posepath.replace('pose', 'pos'))  # (N, T, J, 3)
    torch.save(rot6d.detach().cpu(), out_posepath.replace('pose', 'rot6d'))  # (N, T, J, 6)
    torch.save(angular.detach().cpu(), out_posepath.replace('pose', 'angular'))  # (N, T, J, 3)
    torch.save(global_xform.detach().cpu(), out_posepath.replace('pose', 'global_xform'))  # (N, T, J, 6)
    torch.save(velocity.detach().cpu(), out_posepath.replace('pose', 'velocity'))  # (N, T, J, 3)
    # <=================== 



    # <=================== normalize vars
    _, rotmat_mean, rotmat_std = normalize(rotmat)
    _, pos_mean, pos_std = normalize(pos)
    _, rot6d_mean, rot6d_std = normalize(rot6d)
    _, angular_mean, angular_std = normalize(angular)
    _, global_xform_mean, global_xform_std = normalize(global_xform)
    _, velocity_mean, velocity_std = normalize(velocity)

    mean = {}
    mean['rotmat'] = rotmat_mean.detach().cpu()
    mean['pos'] = pos_mean.detach().cpu()
    mean['rot6d'] = rot6d_mean.detach().cpu()
    mean['angular'] = angular_mean.detach().cpu()
    mean['global_xform'] = global_xform_mean.detach().cpu()
    mean['velocity'] = velocity_mean.detach().cpu()

    std = {}
    std['rotmat'] = rotmat_std.detach().cpu()
    std['pos'] = pos_std.detach().cpu()
    std['rot6d'] = rot6d_std.detach().cpu()
    std['angular'] = angular_std.detach().cpu()
    std['global_xform'] = global_xform_std.detach().cpu()
    std['velocity'] = velocity_std.detach().cpu()

    # if args.normalize:
    torch.save(mean, out_posepath.replace('pose', 'mean'))
    torch.save(std, out_posepath.replace('pose', 'std'))
    # <=================== 
    
    return len(data_poses)

def collect_amass_stats(logger=None):
    import matplotlib.pyplot as plt

    if logger is None:
        log_name = os.path.join('./data/ReInterHand', 're_interhand_stats.log')
        if os.path.exists(log_name):
            os.remove(log_name)
        logger = log2file(log_name)

    logger('Collecting stats for ReInterHand datasets:')

    gender_stats = {}
    fps_stats = {}
    durations = []
    
    for capture_id_dir in ReInterHand_data_root.iterdir():
        
        if args.hand_type == 'L':
            json_path = sorted([pth for pth in capture_id_dir.rglob("params/*_left.json")], key=lambda x: int(x.stem.split('_')[0]))
        if args.hand_type == 'R':
            json_path = sorted([pth for pth in capture_id_dir.rglob("params/*_right.json")], key=lambda x: int(x.stem.split('_')[0]))
        
        if len(json_path)==0: continue
        pose = np.array([json.load(open(str(p),'r'))['pose'] for p in json_path])[:,:45]
        fps = 30
        duration = len(pose)/fps
        durations.append(duration)
        if fps in fps_stats.keys():
            fps_stats[fps] += 1
        else:
            fps_stats[fps] = 1
        

    logger('\n')
    logger('Total motion sequences: {:,}'.format(len(durations)))
    logger('\tTotal Duration: {:,.2f}s'.format(sum(durations)))
    logger('\tMin Duration: {:,.2f}s'.format(min(durations)))
    logger('\tMax Duration: {:,.2f}s'.format(max(durations)))
    logger('\tAverage Duration: {:,.2f}s'.format(sum(durations) / len(durations)))
    logger('\tSequences longer than 5s: {:,} ({:.2f}%)'.format(sum(i > 5 for i in durations), sum(i > 5 for i in durations) / len(durations) * 100))
    logger('\tSequences longer than 10s: {:,} ({:.2f}%)'.format(sum(i > 10 for i in durations), sum(i > 10 for i in durations) / len(durations) * 100))
    logger('\n')
    logger('Gender:')
    for key, value in gender_stats.items():
        logger('\t{}: {:,} (Duration: {:,.2f}s)'.format(key, len(value), sum(value)))
    logger('\n')
    logger('FPS:')
    for key, value in fps_stats.items():
        logger('\t{}: {:,}'.format(key, value))

    # Plot histograms for duration distributions.
    fig, axes = plt.subplots(3, sharex=True)
    fig.tight_layout(pad=3.0)
    fig.suptitle('AMASS Data Duration Distribution')
    axes[0].hist(durations, density=False, bins=100)
    axes[0].set_title('total')
    for key, value in gender_stats.items():
        if key == 'female':
            f = axes[1]
        else:
            f = axes[2]
        f.hist(value, density=False, bins=100)
        f.set_title(f'{key}')
    # Add a big axis, hide frame.
    fig.add_subplot(111, frameon=False)
    # Hide tick and tick label of the big axis.
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Duration (second)")
    plt.ylabel("Counts", labelpad=10)
    fig.savefig('./data/reInterHand/duration_dist.pdf')


class ReInterHand(Dataset):
    def __init__(self, dataset_dir):
        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).split('-')[0]
            self.ds[k] = torch.load(data_fname)

    def __len__(self):
        return len(self.ds['rotmat'])

    def __getitem__(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys() if k not in ['mean', 'std']}

        return data


if __name__ == '__main__':
    args = Arguments('./configs', filename=sys.argv[1])
    work_dir = makepath(args.dataset_dir)

    log_name = os.path.join(work_dir, 'ReInterHand.log')
    if os.path.exists(log_name): os.remove(log_name)
    logger = log2file(log_name)
    logger('Re. InterHand Data Preparation Began.')
    
    ReInterHand_data_root = Path(args.raw_data_root)

    # collect_amass_stats(logger=logger)
    out_posepath = makepath(os.path.join(work_dir, f'pose-{args.hand_type}-{args.data.clip_length}-{args.data.fps}fps.pt'), isfile=True)
    dump_amass2pytroch(out_posepath, logger=logger)
