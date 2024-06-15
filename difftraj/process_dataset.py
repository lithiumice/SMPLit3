import os
import sys
import pandas
import numpy as np
import random
import codecs as cs
from tqdm import tqdm
from easydict import EasyDict
import joblib
import clip
import math
import click


import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate

from quaternion import *
from constants import *

main_code_path = os.path.join(os.path.dirname(__file__), "..")
model_files_path = os.path.join(main_code_path, "model_files")
third_party_path = os.path.join(main_code_path, "third-party")

sys.path.append(main_code_path)
sys.path.append(third_party_path)

from lib.loco.trajdiff import (
    cvt_rot,
    traj_global2local_heading,
    traj_local2global_heading,
    angle_axis_to_quaternion,
    a2m,
    m2s,
)
from data_loaders.humanml.data.dataset import traj2d_global_to_local

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

from dataclasses import dataclass
from typing import Callable, List, Optional, Union


def norm_glamr_traj(local_traj):
    # local_traj: T,11
    local_traj[0, :2] = 0
    local_traj[0, -2] = 1
    local_traj[0, -1] = 0
    return local_traj


cut_frames_info = pandas.read_csv("dataset/100styles_meta_Frame_Cuts.csv")
AMASS_OCC_META_PATH = "amass_copycat_occlusion_v3.pkl"
amass_occlusion = joblib.load(AMASS_OCC_META_PATH)


def judge_is_normal(search_key):
    if search_key in amass_occlusion:
        issue = amass_occlusion[search_key]["issue"]
        if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[
            search_key
        ]:
            bound = amass_occlusion[search_key]["idxes"][
                0
            ]  # This bounded is calucaled assuming 30 FPS.....
            if bound < 10:
                print("bound too small", search_key, bound)
                return False
        else:
            print("issue irrecoverable", search_key, issue)
            return False
    return True


def judge_is_normal2(raw_path):
    clip_name = raw_path.split("/")[-1]
    seq_name = raw_path.split("/")[-2]
    collect_name = raw_path.split("/")[-3]

    if "skate" in raw_path:
        print(f"skate in filename")
        return False

    if "BioMotionLab_NTroje" == collect_name:
        if ("treadmill" in raw_path) or ("normal" in raw_path):
            print(f"treamil in filename")
            return False

    if "MPI_HDM05" == collect_name:
        if "dg" in raw_path:
            print(f"dg in filename")
            return False

    if not os.path.exists(raw_path):
        print(f"{raw_path} not exists...")
        return False

    return True


@dataclass
class ParseNpzOutput:
    root_orient: Union[torch.Tensor, np.ndarray]
    pose_body: Union[torch.Tensor, np.ndarray]
    pose_hand: Union[torch.Tensor, np.ndarray]
    trans: Union[torch.Tensor, np.ndarray]


def parse_100style_npz(name, bdata, down_sample, device, debug=False):
    # cut frames
    tmp = name.split("/")[-1].split("_")
    style = tmp[0]
    dir_type = tmp[1]
    infos = cut_frames_info[cut_frames_info["STYLE_NAME"] == style]
    st = infos[f"{dir_type.upper()}_START"].item()
    et = infos[f"{dir_type.upper()}_STOP"].item()
    st = int(st)
    et = int(et)
    if debug:
        print(f"st:{st},et:{et}")
    poses = torch.Tensor(bdata["poses"])[st:et]
    trans = torch.Tensor(bdata["trans"])[st:et]
    root_orient = (poses[::down_sample, 0]).to(device)
    pose_body = (poses[::down_sample, 1:22, :]).to(device).reshape(-1, 63)
    pose_hand = (poses[::down_sample, -30:, :]).to(device).reshape(-1, 90)
    trans = (trans[::down_sample]).to(device)
    # return ParseNpzOutput(root_orient, pose_body, pose_hand, trans)
    return root_orient, pose_body, pose_hand, trans, style


def ours_npz(bdata, down_sample, device):
    """Parse scrape data motion npz, which store in blender smplex-addon format."""
    poses = torch.Tensor(bdata["poses"])
    trans = torch.Tensor(bdata["trans"])
    root_orient = (poses[::down_sample, 0]).to(device)
    pose_body = (poses[::down_sample, 1:22, :]).to(device).reshape(-1, 63)
    pose_hand = (poses[::down_sample, -30:, :]).to(device).reshape(-1, 90)
    trans = (trans[::down_sample]).to(device)
    # return ParseNpzOutput(root_orient, pose_body, pose_hand, trans)
    return root_orient, pose_body, pose_hand, trans


def parse_amass_npz(bdata, down_sample, device):
    """Process raw npz format of AMASS."""
    root_orient = torch.Tensor(bdata["poses"][::down_sample, :3]).to(device)
    pose_body = torch.Tensor(bdata["poses"][::down_sample, 3:66]).to(device)
    pose_hand = torch.Tensor(bdata["poses"][::down_sample, 66:]).to(device)
    trans = torch.Tensor(bdata["trans"][::down_sample]).to(device)
    # return ParseNpzOutput(root_orient,pose_body,pose_hand,trans)
    return root_orient, pose_body, pose_hand, trans


male_bm = None
female_bm = None


def get_body_model(
    device,
    type="male",
):
    if type == "male":
        global male_bm
        if male_bm is None:
            male_bm = BodyModel(
                bm_fname=male_bm_path,
                num_betas=10,
                num_dmpls=8,
                dmpl_fname=male_dmpl_path,
            )
            male_bm = male_bm.to(device)
        return male_bm
    elif type == "female":
        global female_bm
        if female_bm is None:
            female_bm = BodyModel(
                bm_fname=female_bm_path,
                num_betas=10,
                num_dmpls=8,
                dmpl_fname=female_dmpl_path,
            )
            female_bm = female_bm.to(device)
        return female_bm
    else:
        raise ValueError(f"Not type defined for {type=}")


@click.command()
@click.option("--save_train_mean_std", help="ouput path")
@click.option("--save_train_jpkl", help="ouput path")
@click.option("--in_npz_list", help="input path")
@click.option(
    "--in_data_type",
    type=click.Choice(["amass", "100styles", 'scrape']),
    help="Currently, only support amass dataset.",
)
@click.option(
    "--save_data_fps",
    type=int,
    default=30,
    help="train on 30 fps, but actually can work on other fps when inference.",
)
@click.option(
    "--win_size", type=int, default=35, help="Depend on your difftraj windows size."
)
@click.option("--max_process_npz_num", type=int, default=-1, help="if -1, no limit.")
def parse_dataset(
    save_train_mean_std,
    save_train_jpkl,
    in_npz_list,
    in_data_type="amass",
    save_data_fps=30,
    win_size=35,
    split_file="train",
    device="cuda",
    min_motion_frames=60,
    use_betas=False,
    max_process_npz_num=-1,
):

    id_list = open(in_npz_list, "r").readlines()
    print(f'{len(id_list)=}')

    load_pred_process_jpks = {
        in_data_type: {
            "train": save_train_jpkl,
        }
    }

    for in_data_type, data_info in load_pred_process_jpks.items():
        file_path = data_info[split_file]
        print(f"loading dataloader {in_data_type} and will save to {file_path}")

        ctrl_data = []
        data = []
        lengths = []
        styles = []
        accl_judge_list = []

        for idx, name in tqdm(enumerate(id_list), desc="Processing npz data:"):
            if max_process_npz_num != -1:
                if idx > max_process_npz_num:
                    break

            raw_path = name.strip()
            clip_name = raw_path.split("/")[-1]
            seq_name = raw_path.split("/")[-2]
            collect_name = raw_path.split("/")[-3]

            if in_data_type=='amass':
                # We filter out motions that are not flat-ground. this is a limitaiton.
                if not judge_is_normal(search_key=f"0-{collect_name}_{seq_name}_{clip_name[:-4]}"):
                    continue

                if not judge_is_normal2(raw_path):
                    continue

            bdata = np.load(raw_path, allow_pickle=True)
            
            try:
                fps = bdata["mocap_framerate"]
            except:
                continue

            if bdata["gender"] == "female":
                bm = get_body_model(
                    device,
                    type="female",
                )
            else:
                bm = get_body_model(
                    device,
                    type="male",
                )
                
            # template_joints = bm().Jtr

            down_sample = int(fps / save_data_fps)
            if down_sample < 1:
                down_sample = 1

            # import ipdb;ipdb.set_trace()

            if in_data_type=='100styles':
                root_orient, pose_body, pose_hand, trans, style = parse_100style_npz(name, bdata, down_sample, device)
            elif in_data_type=='scrape':
                raise NotImplementedError
                # blender smplx npz format
                style = "Neutral"
                ours_npz()
            else:
                # amsss npz format
                style = "Neutral"
                root_orient, pose_body, pose_hand, trans = parse_amass_npz(
                    bdata, down_sample, device
                )

            bs = trans.shape[0]
            if bs < win_size:
                print(f"[Data] shorter than win_size {win_size}: {bs}")
                continue

            if bs < min_motion_frames:
                print(
                    f"[Data] shorter than min_motion_frames {min_motion_frames}: {bs}"
                )
                continue

            # normalize traj: tranform to direction that face x+
            orient_q = cvt_rot(root_orient, "aa", "quat")
            local_traj = traj_global2local_heading(
                trans, orient_q, local_orient_type="6d"
            )  # B,11
            local_traj = norm_glamr_traj(local_traj)
            T, R = traj_local2global_heading(local_traj)
            root_orient = cvt_rot(R, "quat", "aa")
            trans = T

            # Calculate the projection of a person facing the direction on the xy plane
            if use_betas:
                betas = (
                    torch.Tensor(bdata["betas"][:10][np.newaxis])
                    .to(device)
                    .expand(bs, -1)
                )
                body = bm(
                    pose_body=pose_body,
                    pose_hand=pose_hand,
                    betas=betas,
                    root_orient=root_orient,
                )
            else:
                # we use this setting.
                # take betas into account did perform well
                body = bm(
                    pose_body=pose_body, pose_hand=pose_hand, root_orient=root_orient
                )
            positions = body.Jtr + trans.unsqueeze(1)
            positions = positions.cpu().numpy()
            across1 = positions[:, r_hip] - positions[:, l_hip]
            across2 = positions[:, sdr_r] - positions[:, sdr_l]
            across = across1 + across2
            across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]
            # forward (3,), rotate around y-axis
            forward_vec = np.cross(np.array([[0, 0, 1]]), across, axis=-1)
            forward_vec = forward_vec[:, [0, 1]]
            forward_vec = (
                forward_vec / np.sqrt((forward_vec**2).sum(axis=-1))[..., np.newaxis]
            )

            transl_xy_global = trans[:, [0, 1]]
            forward_vec_global = torch.from_numpy(forward_vec).type_as(trans)

            transl_xy_global = transl_xy_global[::1]
            forward_vec_global = forward_vec_global[::1]
            ctrl_signal_all = traj2d_global_to_local(
                transl_xy_global, forward_vec_global
            )

            # Calculate the motion representation of the target
            body = bm(pose_body=pose_body)
            pose_seq = body.Jtr
            pose_seq = pose_seq[:, :22, :]
            local_pose = pose_seq.reshape(bs, -1)  # 22*3
            orient_q = angle_axis_to_quaternion(root_orient)
            local_traj = traj_global2local_heading(
                trans, orient_q, local_orient_type="6d"
            )  # B,11

            pose_body_axis = pose_body.reshape(-1, 21, 3)
            pose_body_mat = a2m(pose_body_axis)
            pose_body_6d = m2s(pose_body_mat).reshape(bs, -1)  # 21*6

            motoin_info_all = torch.cat([local_traj, pose_body_6d], dim=-1)
            motoin_info_all = torch.cat([motoin_info_all, local_pose], dim=-1)
            # The final motoin_info_all contatins:
            # [local_traj(11 dim), pose_body_6d(21*6), local_pose(22*3)]
            # attention!: not all of them are used in training.

            total_frames = motoin_info_all.shape[0]
            lengths.append(motoin_info_all.shape[0] - win_size)
            data.append(motoin_info_all.cpu().numpy())
            ctrl_data.append(ctrl_signal_all.cpu().numpy())
            styles.append(style)

            # measure diversity by joints
            # This is used to calulate data statistic of dataset.
            # not nessesary.
            tmp = local_pose.reshape(-1, 22, 3)
            accl_judge = (tmp[1:] - tmp[:-1]).norm(dim=-1).mean(dim=-1).mean(dim=-1)
            accl_judge_list.append(accl_judge)

        print(f'After processing, we get {len(data)=}')

        joblib.dump(
            {
                "styles": styles,
                "data": data,
                "ctrl_data": ctrl_data,
                "lengths": lengths,
                "accl_judge_list": accl_judge_list,
            },
            file_path,
        )
        # cumsum = np.cumsum([0] + lengths)
        # cumsum[-1]/30/3600

        print(f"[Save] save mean and std for normalization to {save_train_mean_std}")
        tmp1 = np.concatenate(data, axis=0)
        tmp2 = np.concatenate(ctrl_data, axis=0)
        np.savez(
            save_train_mean_std,
            **{
                "motoin_info_all_list_mean": np.mean(tmp1, axis=0),
                "motoin_info_all_list_std": np.std(tmp1, axis=0),
                "ctrl_signal_mean": np.mean(tmp2, axis=0),
                "ctrl_signal_std": np.std(tmp2, axis=0),
            },
        )


if __name__ == "__main__":
    parse_dataset()
