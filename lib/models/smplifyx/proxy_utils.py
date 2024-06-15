import os
import sys

"""
NeMF dependence
"""

NeMF_path = "third-party/NeMF"
sys.path.insert(0, f"{NeMF_path}/src")
from NeMF.src.nemf_arguments import Arguments
from NeMF.src.nemf.fk import ForwardKinematicsLayer
from NeMF.src.nemf.generative import Architecture
from NeMF.src.nemf_utils import estimate_angular_velocity, estimate_linear_velocity

import torch

torch.set_default_dtype(torch.float32)

from lib.loco.trajdiff import *

const_crist = torch.nn.MSELoss()


class BodyProxy:
    def __init__(self):
        with path_enter(NeMF_path):
            nemf_args = Arguments(f"configs", filename="application.yaml")
            nemf_args.data.root_transform = False
            nemf_args.data.overlap_len = 16

            nemf_fk = ForwardKinematicsLayer(nemf_args)

            nemf_model = Architecture(nemf_args, 1)
            nemf_model.load(optimal=True)
            nemf_model.eval()

            self.nemf_fk = nemf_fk
            self.nemf_args = nemf_args
            self.nemf_model = nemf_model

            self.clip_length = self.nemf_args.data.clip_length
            self.overlap_len = self.nemf_args.data.overlap_len
            self.dt = 1.0 / self.nemf_args.data.fps

    def prepare_nemf_inp_data(self, body_pose):
        """
        body_pose: T, 21, 3
        """
        # import ipdb;ipdb.set_trace()
        body_pose = body_pose.reshape(-1, 63)
        poses = torch.cat(
            [
                torch.zeros(body_pose.shape[0], 3).type_as(I),
                body_pose,
                torch.zeros(body_pose.shape[0], 6).type_as(I),
            ],
            dim=1,
        )
        pose = split_tensor(poses, self.clip_length, self.overlap_len)
        pose = pose.view(-1, self.clip_length, 24, 3)  # axis-angle (N, T, J, 3)
        rotmat = a2m(pose)  # rotation matrix (N, T, J, 3, 3)

        identity = torch.zeros_like(rotmat[:, :, 0])  # (N, T, 3, 3)
        identity[:, :, 0, 0] = 1
        identity[:, :, 1, 1] = 1
        identity[:, :, 2, 2] = 1

        rotmat[:, :, 0] = identity
        rot6d = m2s(rotmat)  # 6D rotation representation (N, T, J, 6)
        pos, global_xform = self.nemf_fk(rot6d.view(-1, 24, 6))
        pos = pos.contiguous().view(
            -1, self.clip_length, 24, 3
        )  # local joint positions (N, T, J, 3)

        velocity = estimate_linear_velocity(
            pos, dt=self.dt
        )  # linear velocity of all the joints (N, T, J, 3)
        angular = estimate_angular_velocity(
            rotmat.clone(), dt=self.dt
        )  # angular velocity of all the joints (N, T, J, 3)

        global_xform = global_xform.view(
            -1, self.clip_length, 24, 4, 4
        )  # global transformation matrix for each joint (N, T, J, 4, 4)
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
        pose: 1, T, 144
        """
        pose = s2a(pose.squeeze().reshape(-1, 24, 6))
        org_T = pose.shape[0]
        orient = pose[:, :1]
        body_pose = pose[:, 1 : 1 + 21, :]
        nemf_input_data = self.prepare_nemf_inp_data(body_pose)
        self.nemf_model.set_input(nemf_input_data)
        z_l, _, _ = self.nemf_model.encode_local()
        self._proxy_var = {
            "org_T": org_T,
            "cvt_vars": {
                "orient": a2s(orient),
                "z_l": z_l,
            },
        }
        return self._proxy_var

    def dec(self, proxy_var, return_consit=False):
        """
        B, 512
        """
        org_T = self._proxy_var["org_T"]
        orient = proxy_var["cvt_vars"]["orient"]
        z_l = proxy_var["cvt_vars"]["z_l"]
        nemf_output = self.nemf_model.decode(
            z_l, None, length=self.clip_length, step=1.0
        )
        rotmat = nemf_output["rotmat"]  # (B, T, J, 3, 3)
        b_size, _, n_joints = rotmat.shape[:3]
        local_rotmat = self.nemf_fk.global_to_local(
            rotmat.view(-1, n_joints, 3, 3)
        )  # (B x T, J, 3, 3)
        local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)
        # import ipdb;ipdb.set_trace()
        local_rotmat = merge_tensor_w_avg(
            (m2s(local_rotmat)), self.overlap_len, org_T
        )  # (T, J, 3)
        rec_pose = (
            torch.cat([orient, local_rotmat[..., 1:, :]], dim=1)
            .unsqueeze(0)
            .reshape(1, -1, 24 * 6)
        )
        # rec_pose: 1,T,24,6

        if return_consit:
            ret = get_overlap_pair(m2s(rotmat), self.overlap_len, org_T)
            consit = sum([const_crist(ret[i][0], ret[i][1]) for i in range(len(ret))])
        else:
            consit = torch.tensor(0)

        return rec_pose, consit

    def get_proxy(self):
        return self._proxy_var
