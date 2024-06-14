
import numpy as np
import joblib
from glob import glob
from tqdm import tqdm
import os
import sys
import argparse

sys.path.append(os.getcwd())

from embodied_pose.utils.motion_lib import MotionLib
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as joint_names
from uhc.smpllib.smpl_local_robot import Robot
from scipy.spatial.transform import Rotation as sRot

import torch
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default="data/motion_lib/amass_aug")
parser.add_argument('--num_motion_libs', type=int, default=8)
args = parser.parse_args()


info = joblib.load('data/misc/smpl_body_info.pkl')
robot_cfg = {
    "mesh": True,
    "model": "smpl",
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}

model_xml_path = f"/tmp/smpl/smpl_mesh_humanoid_v1_convert_{uuid.uuid4()}.xml"
smpl_local_robot = Robot(robot_cfg,data_dir= "data/smpl",model_xml_path=model_xml_path)
mujoco_joint_names = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
    'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
    'R_Elbow', 'R_Wrist', 'R_Hand'
]
smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]
os.makedirs(args.out_dir, exist_ok=True)

amass_occlusion = joblib.load("/apdcephfs/private_wallyliang/PLANT/motion_imitator/sample_data/amass_copycat_occlusion_v3.pkl")
raw_list = glob(f'/apdcephfs/share_1330077/dataset/amass_raw/*/*/*.npz')
print(f'len raw_list: {len(raw_list)}')#15805
npzs_list = []
for name in raw_list:
    raw_path = name.strip()
    clip_name = raw_path.split('/')[-1]
    seq_name = raw_path.split('/')[-2]
    collect_name = raw_path.split('/')[-3]
    
    # if ('skate' in raw_path):
    #     print(f'skate in filename')
    #     continue
                                
    # if 'BioMotionLab_NTroje'==collect_name:
    #     if ('treadmill' in raw_path) or ('normal' in raw_path):
    #         print(f'treamil in filename')
    #         continue
        
    # if 'MPI_HDM05'==collect_name:
    #     if ('dg' in raw_path):
    #         print(f'dg in filename')
    #         continue
                                            
    # if not os.path.exists(raw_path):
    #     print(f'{raw_path} not exists...')
    #     continue

    search_key = f'0-{collect_name}_{seq_name}_{clip_name[:-4]}'
    if search_key in amass_occlusion:
        issue = amass_occlusion[search_key]["issue"]
        if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[search_key]:
            bound = amass_occlusion[search_key]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
            if bound < 10:
                print("bound too small", search_key, bound)
                continue
        else:
            print("issue irrecoverable", search_key, issue)
            continue
        
    npzs_list.append(name)
        
print(f'len npzs_list: {len(npzs_list)}')#13451
num_motion_libs = args.num_motion_libs
motion_lib_seq_arr = np.array_split(npzs_list, num_motion_libs)
device = 'cuda'
import math
for i, motion_lib_seqs in enumerate((motion_lib_seq_arr)):
    for down_rate in [0.5,2]:#slow,normal,fast
        print(f'down_rate: {down_rate}')
        print(f'processing part {i}')
        motion_lib_input_dict = dict()
        for npz_path in tqdm(motion_lib_seqs):
            try:
                file_name = os.path.basename(npz_path)
                smpl_data_entry=np.load(npz_path)
                # import ipdb;ipdb.set_trace()
                
                
                # pose_aa = smpl_data_entry['poses'].copy()[:,:22].reshape(-1,66)[::down_rate]
                # trans = smpl_data_entry['trans'].copy()[::down_rate]
                
                fps = smpl_data_entry["mocap_framerate"]
                down1 = fps/30
                down2 = down1*down_rate
                down2 = math.ceil(down2)
                if down2<1: down2=1
                print(f'down2:{down2}')
                pose_aa = smpl_data_entry['poses'].copy().reshape(-1,52,3)[:,:22].reshape(-1,66)[::down2]
                trans = smpl_data_entry['trans'].copy()[::down2]
                
                                
                beta = smpl_data_entry['betas'][:10].copy()
                gender = smpl_data_entry['gender']
                fps = 30.0

                if isinstance(gender, np.ndarray):
                    gender = gender.item()
                if isinstance(gender, bytes):
                    gender = gender.decode("utf-8")
                if gender == "neutral":
                    gender_number = [0]
                    smpl_parser = smpl_local_robot.smpl_parser_n
                elif gender == "male":
                    gender_number = [1]
                    smpl_parser = smpl_local_robot.smpl_parser_m
                elif gender == "female":
                    gender_number = [2]
                    smpl_parser = smpl_local_robot.smpl_parser_f
                else:
                    import ipdb
                    ipdb.set_trace()
                    raise Exception("Gender Not Supported!!")

                batch_size = pose_aa.shape[0]
                pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)  # TODO: need to extract correct handle rotations instead of zero
                pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)[..., smpl_2_mujoco, :]
                smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None, ]), gender=gender_number)
                # TODO: check betas
                smpl_local_robot.write_xml()
                skeleton_tree = SkeletonTree.from_mjcf(model_xml_path)
                # self.sk_tree = skeleton_tree
                # self.smpl_robot = smpl_local_robot

                root_trans = trans + skeleton_tree.local_translation[0].numpy()
                verts, joints = smpl_parser.get_joints_verts(pose=torch.from_numpy(pose_aa),th_betas=torch.from_numpy(beta[None, ]),th_trans=torch.from_numpy(trans))
                # min_verts_h = verts[..., 2].min().item()
                min_verts_h = verts[..., 2].min(dim=-1)[0].mean().item()

                new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree,torch.from_numpy(pose_quat),torch.from_numpy(root_trans),is_local=True)
                new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=fps)

                new_motion_out = new_motion.to_dict()
                new_motion_out['seq_name'] = 'test'
                new_motion_out['seq_idx'] = 0
                new_motion_out['trans'] = trans
                new_motion_out['root_trans'] = root_trans
                new_motion_out['pose_aa'] = pose_aa
                new_motion_out['beta'] = beta
                new_motion_out['beta_idx'] = 0
                new_motion_out['gender'] = gender
                new_motion_out['min_verts_h'] = min_verts_h
                new_motion_out['body_scale'] = 1.0
                new_motion_out['__name__'] = "SkeletonMotion"
                motion_lib_input_dict[f'{down_rate}-'+file_name] = new_motion_out
            except:
                # import traceback; traceback.print_exc()
                print(f'err at {npz_path}')
                pass

        motion_lib = MotionLib(motion_file=motion_lib_input_dict,
            dof_body_ids=info['dof_body_ids'],
            dof_offsets=info['dof_offsets'],
            key_body_ids=info['key_body_ids'],
            device=device,
            clean_up=True
        )
        
        torch.save(motion_lib, f"{args.out_dir}/mlib_part_{i:05d}_speed{down_rate}.pth")
        del motion_lib_input_dict
        del motion_lib
        
smpl_local_robot.clean_up()

