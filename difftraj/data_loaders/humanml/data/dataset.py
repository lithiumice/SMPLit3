import pandas
import torch
from torch.utils import data
import numpy as np
import os
import sys
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import joblib
import pandas

from quaternion import *
from constants import *


def t2t(x):
    return torch.from_numpy(x)


def norm_glamr_traj(local_traj):
    # local_traj: T,11
    local_traj[0, :2] = 0
    local_traj[0, -2] = 1
    local_traj[0, -1] = 0
    return local_traj


def transl_xy_to_tangent(transl_xy_global):
    B = transl_xy_global.size(0)
    tangent_vectors = torch.zeros_like(transl_xy_global)
    tangent_vectors[:-1] = transl_xy_global[1:] - transl_xy_global[:-1]
    tangent_vectors[-1] = tangent_vectors[-2]
    tangent_scale = torch.norm(tangent_vectors, dim=1, keepdim=True)
    tangent_vectors = tangent_vectors / (tangent_scale + 1e-8)
    return tangent_vectors, tangent_scale


def torch_safe_atan2(y, x, eps: float = 1e-6):
    y = y.clone()
    y[(y.abs() < eps) & (x.abs() < eps)] += eps
    return torch.atan2(y, x)


def rot_2d(xy, theta):
    rot_x = xy[..., 0] * torch.cos(theta) - xy[..., 1] * torch.sin(theta)
    rot_y = xy[..., 0] * torch.sin(theta) + xy[..., 1] * torch.cos(theta)
    rot_xy = torch.stack([rot_x, rot_y], dim=-1)
    return rot_xy


def heading_to_vec(h_theta):
    v = torch.stack([torch.cos(h_theta), torch.sin(h_theta)], dim=-1)
    return v


def vec_to_heading(h_vec):
    h_theta = torch_safe_atan2(h_vec[..., 1], h_vec[..., 0])
    return h_theta


def traj2d_global_to_local(global_xy, global_fw):
    heading = torch.arctan2(global_fw[:, 1], global_fw[:, 0])
    d_heading = heading[1:] - heading[:-1]
    d_heading = torch.cat([heading[[0]], d_heading])
    d_heading_vec = heading_to_vec(d_heading)

    xy = global_xy
    d_xy = xy[1:] - xy[:-1]
    d_xy_yawcoord = rot_2d(d_xy, -heading[:-1])
    d_xy_yawcoord = torch.cat([xy[[0]], d_xy_yawcoord])
    local_traj = torch.cat([d_xy_yawcoord, d_heading_vec], dim=-1)
    return local_traj


def traj2d_local_to_global(local_traj):
    d_xy_yawcoord, d_heading_vec = local_traj[..., :2], local_traj[..., -2:]
    d_heading = vec_to_heading(d_heading_vec)
    heading = torch.cumsum(d_heading, dim=0)
    d_heading_vec = heading_to_vec(d_heading)

    d_xy = d_xy_yawcoord.clone()
    d_xy[1:] = rot_2d(d_xy_yawcoord[1:], heading[:-1])
    xy = torch.cumsum(d_xy, dim=0)

    return xy, d_heading_vec


def plot_vectors(transl_xy, forward_vec, title, save_path="./test.png"):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.quiver(
        transl_xy[:, 0],
        transl_xy[:, 1],
        forward_vec[:, 0],
        forward_vec[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="r",
    )
    plt.plot(transl_xy[:, 0], transl_xy[:, 1], color="y")
    plt.title(title)
    plt.grid()
    plt.axis("equal")
    # plt.show()
    plt.savefig(save_path)
    plt.close()


style_onehot_save_path = os.path.join(
    os.path.dirname(__file__), "../../../style_str_to_onehot.pt"
)


def load_from_jpkl(train_data_path):
    lengths = []
    data = []
    styles = []
    ctrl_data = []
    for file_path in train_data_path:
        if os.path.exists(file_path):
            print(f"loading {file_path}")
            load_con = joblib.load(file_path)
            lens = load_con["lengths"]
            print(f"len: {len(lens)}")
            lengths.extend(load_con["lengths"])
            data.extend(load_con["data"])
            styles.extend(load_con["styles"])
            ctrl_data.extend(load_con["ctrl_data"])
        else:
            print(f"wrong file: {file_path}")
    return lengths, data, styles, ctrl_data

import ipdb

class diffgen_dataset(data.Dataset):
    """
    autoregressive generation with 100STYLE
    """

    def __init__(self, split, args, **kwargs):
        self.args = args
        self.split = split
        self.ex_fps = self.args.train_fps
        self.pastMotion_len = self.args.p_len
        self.inp_len = self.args.f_len

        if "test" in split and "window_size" in self.args:
            self.win_size = self.pastMotion_len + self.args.window_size
            self.eval_mode = True
        else:
            self.win_size = self.pastMotion_len + self.inp_len
            self.eval_mode = False

        self.lengths, self.data, self.styles, self.ctrl_data = load_from_jpkl(
            self.args.train_data_path
        )

        self.style_str_to_onehot = torch.load(style_onehot_save_path)

        # 35 is hard coded
        self.lengths = [
            self.lengths[idx] + 35 - self.win_size
            for idx in range(len(self.lengths))
        ]

        def relist_data(ds):
            return [
                ds[idx] for idx in range(len(self.lengths)) if self.lengths[idx] > 0
            ]

        self.data = relist_data(self.data)
        self.styles = relist_data(self.styles)
        self.ctrl_data = relist_data(self.ctrl_data)
        self.lengths = relist_data(self.lengths)

        self.cumsum = np.cumsum([0] + self.lengths)
        print(f"data length: {len(self.data)}")
        print(
            "Total number of motions {}, snippets {}".format(
                len(self.data), self.cumsum[-1]
            )
        )

        # load mean std
        train_mean_std = self.args.load_mean_path
        load_con = np.load(train_mean_std)
        print(f"load std mean from {train_mean_std}")
        for k, v in load_con.items():
            print(f"set key {k}, shape {v.shape}")
            setattr(self, k, v)

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if "test" in self.split:
            item = 0

        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0

        style = self.styles[motion_id]
        style_code = self.style_str_to_onehot[style].cpu().numpy()
        motoin_info_all = self.data[motion_id][idx : idx + self.win_size]
        ctrl_signal = self.ctrl_data[motion_id][idx : idx + self.win_size]

        motoin_info_all = (
            motoin_info_all - self.motoin_info_all_list_mean
        ) / self.motoin_info_all_list_std
        
        if self.args.normlize_ctrl_traj:
            ctrl_signal = (ctrl_signal - self.ctrl_signal_mean) / self.ctrl_signal_std

        if self.eval_mode:
            target_motion = motoin_info_all[self.pastMotion_len :]
            past_motion = motoin_info_all[: self.pastMotion_len]
            ctrl_traj = ctrl_signal[self.pastMotion_len :]
            # import ipdb;ipdb.set_trace()
            assert target_motion.shape[0] == self.args.window_size
        else:
            target_motion = motoin_info_all[self.pastMotion_len :]  # after
            past_motion = motoin_info_all[
                : self.pastMotion_len
            ]  # prev as past motion condition
            # ipdb.set_trace()
            ctrl_traj = ctrl_signal[self.pastMotion_len :]
            assert target_motion.shape[0] == self.inp_len
            assert ctrl_traj.shape[0] == self.inp_len

        return (style_code, target_motion, past_motion, ctrl_traj)


class difftraj_dataset(data.Dataset):
    """
    for trajectory prediction:
        all args used are mainly store in `args`.
    """

    def __init__(self, split, args=None):
        self.args = args
        self.split = split

        if self.args.use_ar:
            self.win_size = self.args.p_len + self.args.f_len
        else:
            self.win_size = self.args.seq_len

        self.lengths, self.data, self.styles, self.ctrl_data = load_from_jpkl(
            self.args.train_data_path
        )

        self.style_str_to_onehot = torch.load(style_onehot_save_path)

        # 35 is hard coded
        self.lengths = [
            self.lengths[idx] + 35 - self.win_size for idx in range(len(self.lengths))
        ]

        def relist_data(ds):
            return [
                ds[idx] for idx in range(len(self.lengths)) if self.lengths[idx] > 0
            ]

        self.data = relist_data(self.data)
        self.styles = relist_data(self.styles)
        self.ctrl_data = relist_data(self.ctrl_data)
        self.lengths = relist_data(self.lengths)

        self.cumsum = np.cumsum([0] + self.lengths)
        print(f"data length: {len(self.data)}")
        print(
            "Total number of motions {}, snippets {}".format(
                len(self.data), self.cumsum[-1]
            )
        )

        # load mean std
        train_mean_std = self.args.load_mean_path
        print(f"load std mean from {train_mean_std}")

        load_con = np.load(train_mean_std)
        for k, v in load_con.items():
            print(f"set key {k}, shape {v.shape}")
            setattr(self, k, v)

        if data_len := self.__len__() < 1:
            raise Exception(f"Error data length.")

        print(f"[DATA] {data_len=}")

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0

        style = self.styles[motion_id]
        style_code = self.style_str_to_onehot[style].cpu().numpy()
        motoin_info_all = self.data[motion_id][idx : idx + self.win_size]
        motoin_info_all = (
            motoin_info_all - self.motoin_info_all_list_mean
        ) / self.motoin_info_all_list_std

        # You can not debug with pdb along with torch.dataloader(num_workers>1)
        # import ipdb;ipdb.set_trace()

        if self.args.use_ar:
            # past motion
            p_motion = motoin_info_all[: self.args.p_len]
            # predict motion
            f_motion = motoin_info_all[self.args.p_len :]
            assert f_motion.shape[0] == self.args.f_len
            f_pose = np.concatenate([f_motion[:, 11:], f_motion[:, 9 : 9 + 2]], axis=-1)
            return (style_code, p_motion, f_pose, f_motion)
        else:
            # only output trajectory
            target = motoin_info_all[:, :11]
            condition = np.concatenate(
                [motoin_info_all[:, 11 + 21 * 6 :], motoin_info_all[:, 9 : 9 + 2]],
                axis=-1,
            )  # (200, 194)
            assert condition.shape[1] == 22 * 3 + 2
            return (style_code, condition, target, motoin_info_all)


class diffpose_dataset(data.Dataset):
    """
    smpl pose prediction
    """

    def __init__(self, split, args=None):
        self.args = args
        self.split = split
        self.win_size = self.args.seq_len

        self.image_points = []
        # self.norm_kp2d = []
        self.cam_angvel = []
        self.data = []
        for file_path in self.args.train_data_path:
            if os.path.exists(file_path):
                print(f"loading {file_path}")
                load_con = joblib.load(file_path)
                self.image_points.extend(load_con["image_points"])
                self.cam_angvel.extend(load_con["cam_angvel"])
                self.data.extend(load_con["motoin_info_all"])
            else:
                print(f"wrong file: {file_path}")

        self.image_points = np.stack(self.image_points, axis=0)
        self.cam_angvel = np.stack(self.cam_angvel, axis=0)
        self.data = np.stack(self.data, axis=0)

        print(f"data shape: {self.data.shape}")

        train_mean_std = self.args.load_mean_path
        load_con = np.load(train_mean_std)
        print(f"load std mean from {train_mean_std}")
        for k, v in load_con.items():
            print(f"set key {k}, shape {v.shape}")
            setattr(self, k, v)

        self.keypoints_normalizer = Normalizer(None)

    def __len__(self):
        # return self.cumsum[-1]
        return len(self.data)

    def __getitem__(self, item):
        idx = item
        # import ipdb;ipdb.set_trace()
        # norm_kp2d = self.norm_kp2d[idx][:,:17*2]
        res = torch.tensor([2048, 2048])
        image_points = t2t(self.image_points[idx])

        # add_mask = 0
        T = image_points.shape[0]
        all_mask = np.ones((T, 17, 1))
        if self.args.add_mask:
            main_p = 0.2
            lower_p = 0.3
            mask = np.random.binomial(1, main_p, size=(T, 17))
            ad_mask = np.ones((T, 17))
            ad_mask[:, [11, 12, 13, 14, 15, 16]] = np.random.binomial(
                1, lower_p, size=(T, 6)
            )
            all_mask = np.logic_and(mask, ad_mask)[..., None]
            image_points = image_points[all_mask]

        norm_kp2d, norm_bbox = self.keypoints_normalizer(
            image_points.cpu().clone(),
            res,
            cam_intrinsics=None,
            patch_width=224,
            patch_height=224,
            bbox=None,
        )
        norm_kp2d = norm_kp2d[:, :-3].cpu().numpy()

        if self.args.add_mask:
            new_norm_kp2d = np.zeros(T, 17)
            new_norm_kp2d[all_mask] = norm_kp2d
            norm_kp2d = new_norm_kp2d

        cam_angvel = self.cam_angvel[idx]
        target = (self.data[idx] - self.motoin_info_all_mean) / self.motoin_info_all_std

        if self.args.diffpose_body_only:
            target = target[:,11:]

        return (norm_kp2d, cam_angvel, target, all_mask)


class CommanDataloader(data.Dataset):
    def __init__(self, dataset_type, split="train", args=None, **kwargs):
        self.t2m_dataset = eval(dataset_type)(split, args=args)

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()
