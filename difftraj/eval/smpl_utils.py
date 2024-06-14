import os
import sys

import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    quaternion_to_matrix,
)

main_code_path = os.path.join(os.path.dirname(__file__), "../..")
model_files_path = os.path.join(main_code_path, "model_files")
third_party_path = os.path.join(main_code_path, "third-party")


sys.path.append(main_code_path)
sys.path.append(third_party_path)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.loco.trajdiff import (
    traj_local2global_heading,
)
from human_body_prior.body_model.body_model import BodyModel
import utils.rotation_conversions as geometry
from quaternion import *
from constants import *


class AnyRep2SMPLjoints:
    def __init__(self):
        self.male_bm = BodyModel(
            bm_fname=male_bm_path, num_betas=10, num_dmpls=8, dmpl_fname=male_dmpl_path
        ).cuda()

        self.zup_to_yup_trans_matrix = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        ).cuda()  # 这里从amass默认的z轴向上变成y轴向上

    def traj_to_joints(self, denormed_traj_motion, remove_float=True, zup_to_yup=False):
        # import ipdb;ipdb.set_trace()
        betas = torch.zeros_like(denormed_traj_motion[:,:10])
        if denormed_traj_motion.shape[-1] == 203:
            pred_local_traj, pred_body_pose, _ = torch.split(
                denormed_traj_motion.cuda(),
                split_size_or_sections=[11, 21 * 6, 22 * 3],
                dim=-1,
            )
        elif denormed_traj_motion.shape[-1] == 137:
            pred_local_traj, pred_body_pose = torch.split(
                denormed_traj_motion.cuda(), split_size_or_sections=[11, 21 * 6], dim=-1
            )
        elif denormed_traj_motion.shape[-1] == 147:
            pred_local_traj, pred_body_pose, betas = torch.split(
                denormed_traj_motion.cuda(),
                split_size_or_sections=[11, 21 * 6, 10],
                dim=-1,
            )
        else:
            raise ValueError(f"Error input shape: {denormed_traj_motion.shape=}")
            return denormed_traj_motion

        rec_trans, rec_orient = traj_local2global_heading(pred_local_traj)
        local_pose = pred_body_pose.reshape(-1, 21, 6)
        rotations = geometry.rotation_6d_to_matrix(local_pose)
        rotations = matrix_to_axis_angle(rotations).reshape(-1, 63)

        # local_joints = self.male_bm(pose_body = rotations).Jtr
        # # import ipdb;ipdb.set_trace()
        # # global_pose[:,0,:]
        # src_orient = quaternion_to_matrix(rec_orient)
        # global_pose=torch.einsum('tij,tbj->tbi',src_orient,local_joints)
        # global_pose=global_pose+rec_trans.unsqueeze(1)

        global_pose = self.male_bm(
            pose_body=rotations,
            root_orient=matrix_to_axis_angle(quaternion_to_matrix(rec_orient)),
            trans=rec_trans,
            betas=betas,
        ).Jtr

        if remove_float:
            foot_offset = global_pose[:, :, 2].min()
            print(f"foot_offset: {foot_offset}")
            global_pose[:, :, 2] -= foot_offset

        if zup_to_yup:
            global_pose = torch.einsum(
                "ij,tbj->tbi", self.zup_to_yup_trans_matrix, global_pose
            )

        return global_pose

    def traj_to_npz(self, denormed_traj_motion, out_path=f"test7.npz"):
        betas = None
        if denormed_traj_motion.shape[-1] == 203:
            pred_local_traj, pred_body_pose, _ = torch.split(
                denormed_traj_motion.cuda(),
                split_size_or_sections=[11, 21 * 6, 22 * 3],
                dim=-1,
            )
        elif denormed_traj_motion.shape[-1] == 137:
            pred_local_traj, pred_body_pose = torch.split(
                denormed_traj_motion.cuda(), split_size_or_sections=[11, 21 * 6], dim=-1
            )
        elif denormed_traj_motion.shape[-1] == 147:
            pred_local_traj, pred_body_pose, betas = torch.split(
                denormed_traj_motion.cuda(),
                split_size_or_sections=[11, 21 * 6, 10],
                dim=-1,
            )
        else:
            return denormed_traj_motion

        rec_trans, rec_orient = traj_local2global_heading(pred_local_traj)
        local_pose = pred_body_pose.reshape(-1, 21, 6)
        local_pose = geometry.rotation_6d_to_matrix(local_pose)
        local_pose_axis = geometry.matrix_to_axis_angle(local_pose)
        rec_orient_axis = geometry.quaternion_to_axis_angle(rec_orient)

        # body = male_bm(pose_body=local_pose_axis.reshape(-1,63), root_orient=rec_orient_axis)
        # pose_seq = body.Jtr + rec_trans.unsqueeze(1)
        # pose_seq_np = pose_seq.detach().cpu().numpy()
        # trans_matrix = np.array([[1.0, 0.0, 0.0],[0.0, 0.0, 1.0],[0.0, 1.0, 0.0]]) # 这里从amass默认的z轴向上变成y轴向上
        # pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)

        poses = torch.cat([rec_orient_axis.unsqueeze(1), local_pose_axis], dim=1)
        poses = poses.cpu().numpy()

        if betas is None:
            betas = (np.zeros(10),)  # 10
        else:
            betas = betas.cpu().numpy()
        to_save = {
            "betas": betas,
            "trans": rec_trans.cpu().numpy(),  # B,3
            "poses": np.concatenate(
                [poses[:, :22, :], np.zeros((poses.shape[0], 33, 3))], axis=1
            ).reshape(-1, 55, 3)[::1],
            "gender": "male",
            "mocap_framerate": 30,
        }
        np.savez(out_path, **to_save)
        print(f"saved to {out_path}")
