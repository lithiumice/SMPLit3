import sys
import torch
from torch import nn
from easydict import EasyDict
from glob import glob
import json
import os
import numpy as np
import math

from lib.loco.traj_utils import *

from human_body_prior.body_model.body_model import BodyModel
from loguru import logger

from difftraj.utils.model_util import create_gaussian_diffusion
from difftraj.data_loaders.get_data import get_model_args, collate
from difftraj.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from difftraj.model.difftraj_model import MDM
from difftraj.model.cfg_sampler import ClassifierFreeSampleModel


SHOW_to_AMASS_cvt = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
AMASS_to_SHOW_cvt = SHOW_to_AMASS_cvt.mT
difftraj_overlap_len = 32

device = "cuda"
dtype = torch.float32
I = torch.eye(3)[None].to(device=device).to(dtype=dtype)


class path_enter(object):
    def __init__(self, target_path=None):
        self.origin_path = None
        self.target_path = target_path

    def __enter__(self):
        if sys.path[0] != self.target_path:
            sys.path.insert(0, self.target_path)

        if self.target_path:
            self.origin_path = os.getcwd()
            os.chdir(self.target_path)
            logger.info(f"entered: {self.target_path}; origin_path: {self.origin_path}")

    def __exit__(self, exc_type, exc_value, trace):
        if self.origin_path:
            os.chdir(self.origin_path)
            logger.info(f"exit to origin_path: {self.origin_path}")


# convert a function into recursive style to
# handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tonp(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy()
    else:
        return vars


@make_recursive_func
def toth(vars):
    if isinstance(vars, np.ndarray):
        return torch.from_numpy(vars).float()
    elif isinstance(vars, (int, float)):
        return torch.tensor(vars).float()
    else:
        return vars


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        return vars


def tt(x):
    return x.float().cuda()


def t2t(x):
    return torch.from_numpy(x).float().cuda()


# def toth(x): return torch.from_numpy(x).float()
# def tonp(x): return x.detach().cpu().numpy()


def cvt_rot_torch(R, in_type, out_type):
    if in_type == "aa":
        cvted_rot = a2m(R)
    elif in_type == "quat":
        cvted_rot = q2m(R)
    elif in_type == "6d":
        cvted_rot = s2m(R)
    elif in_type == "mat":
        cvted_rot = R
    if out_type == "aa":
        cvted_rot = m2a(cvted_rot)
    elif out_type == "quat":
        cvted_rot = m2q(cvted_rot)
    elif in_type == "6d":
        cvted_rot = m2s(cvted_rot)
    elif out_type == "mat":
        cvted_rot = cvted_rot
    return cvted_rot


def cvt_rot(R, *args, **kwargs):
    if isinstance(R, np.ndarray):
        return cvt_rot_torch(torch.from_numpy(R), *args, **kwargs).numpy()
    elif isinstance(R, torch.Tensor):
        return cvt_rot_torch(R, *args, **kwargs)
    else:
        raise ValueError("Invalid input type. Expected NumPy array or PyTorch tensor.")


def apply_cvt_R_tensor(cvt, R, in_type, out_type):
    cvt = cvt.type_as(R)
    cvted_rot = cvt_rot_torch(R, in_type, "mat")
    cvted_rot = torch.einsum("ij,...tjk->...tik", cvt, cvted_rot)
    cvted_rot = cvt_rot_torch(cvted_rot, "mat", out_type)
    return cvted_rot


def apply_cvt_T_tensor(cvt, T):
    cvt = cvt.type_as(T)
    return torch.einsum("ij,...tj->...ti", cvt, T)


def apply_cvt_R(cvt, R, *args, **kwargs):
    if isinstance(R, np.ndarray):
        return apply_cvt_R_tensor(cvt, torch.from_numpy(R), *args, **kwargs).numpy()
    elif isinstance(R, torch.Tensor):
        return apply_cvt_R_tensor(cvt, R, *args, **kwargs)
    else:
        raise ValueError("Invalid input type. Expected NumPy array or PyTorch tensor.")


def apply_cvt_T(cvt, T, *args, **kwargs):
    if isinstance(T, np.ndarray):
        return apply_cvt_T_tensor(cvt, torch.from_numpy(T), *args, **kwargs).numpy()
    elif isinstance(T, torch.Tensor):
        return apply_cvt_T_tensor(cvt, T, *args, **kwargs)
    else:
        raise ValueError("Invalid input type. Expected NumPy array or PyTorch tensor.")


def get_overlap_pair(splits, overlap_len, B):
    split_num = splits.shape[0]
    clip_len = splits.shape[1]
    step = clip_len - overlap_len
    pairs = []
    for idx in range(1, split_num):
        pairs.append([splits[idx - 1][-overlap_len:], splits[idx][:overlap_len]])
    return pairs


def split_tensor(tensor, clip_len, overlap_len):
    B = tensor.shape[0]
    stepB = clip_len - overlap_len
    nB = math.ceil((B - overlap_len) / stepB)
    m = nB * stepB - (B - overlap_len)
    tmp = tensor
    if m > 0:
        tmp = torch.cat((tmp, tensor[-1:].repeat(m, 1)), dim=0)
    # assert (nB-1) * step + clip_len >= tmp.shape[0]
    splits = [tmp[i * stepB : i * stepB + clip_len] for i in range(nB)]
    # import ipdb;ipdb.set_trace()
    return torch.stack(splits)


def merge_tensor(splits, overlap_len, B):
    split_num = splits.shape[0]
    clip_len = splits.shape[1]
    step = clip_len - overlap_len
    merged_tensor = torch.cat(
        [
            splits[idx][:step] if idx != split_num - 1 else splits[idx]
            for idx in range(split_num)
        ],
        dim=0,
    )[:B]
    return merged_tensor


def merge_tensor_w_avg(splits, overlap_len, B):
    # splits = torch.stack(splits)
    split_num = splits.shape[0]
    clip_len = splits.shape[1]
    stepN = clip_len - overlap_len
    pairs = get_overlap_pair(splits, overlap_len, B)

    # import ipdb;ipdb.set_trace()

    merged_tensor = []
    if len(pairs) > 0:
        merged_tensor.append(splits[0][:stepN])
    else:
        merged_tensor.append(splits[0])

    linear_weight = torch.linspace(0, 1, overlap_len).type_as(splits)
    for ii in range(len(splits.shape) - 2):
        linear_weight = linear_weight.unsqueeze(-1)

    for pairs_idx in range(len(pairs)):
        avg_part = (
            linear_weight * pairs[pairs_idx][1]
            + (1 - linear_weight) * pairs[pairs_idx][0]
        )
        merged_tensor.append(avg_part)
        if (pairs_idx + 1) != (split_num - 1):
            merged_tensor.append(splits[pairs_idx + 1][overlap_len:-overlap_len])
        else:
            merged_tensor.append(splits[pairs_idx + 1][overlap_len:])
    merged_tensor = torch.cat(merged_tensor, dim=0)[:B]
    return merged_tensor


# save blender npz
def save_to_blender_smplx_addon_npz(
    results,
    suffix="",
    trans_name="difftraj_trans",
    root_name="difftraj_root",
    pose_var_name="pred_body_pose",
    output_pth=None,
    fps=30,
    save_prefix="",
    save_ex_keys=[],
    ex_params_to_save={},
):
    device = "cuda"
    dtype = torch.float32
    I = torch.eye(3)[None].to(device=device).to(dtype=dtype)

    for _id in results.keys():
        betas = results[_id]["betas"].squeeze()
        if isinstance(betas, torch.Tensor):
            betas = betas.detach().cpu().numpy()
        if len(betas.shape) > 1:
            betas = betas[0]

        trans_world = results[_id][trans_name]
        global_orient = results[_id][root_name]
        pose_var = results[_id][pose_var_name]

        if isinstance(pose_var, np.ndarray):
            pose_var = torch.from_numpy(pose_var)

        pose_var = pose_var.reshape(-1, 23, 3)[:, :21, :].reshape(-1, 63)
        poses = (
            torch.cat(
                [
                    global_orient,
                    pose_var,
                    torch.zeros((pose_var.shape[0], 6), device=pose_var.device),
                ],
                dim=1,
            )
            .clone()
            .reshape(-1, 24, 3)
            .cpu()
        )
        orient = poses[:, 0, :]
        transl = apply_cvt_T(SHOW_to_AMASS_cvt, trans_world).detach().cpu().numpy()
        poses[:, 0, :] = apply_cvt_R(
            SHOW_to_AMASS_cvt, orient, in_type="aa", out_type="aa"
        )
        poses = poses.detach().cpu().numpy()

        if "hand_pose" in results[_id]:
            print(f"has hands...")
            hand_pose = toth(results[_id]["hand_pose"])
            Lh = hand_pose[:, 0]
            Rh = hand_pose[:, 1]
            hand_pose = torch.cat([Lh, Rh], axis=1)
            hand_pose = m2a(hand_pose).cpu().numpy()
        else:
            hand_pose = np.zeros((poses.shape[0], 30, 3))

        if "init_param_th" in results[_id]:
            exp = tonp(results[_id]["init_param_th"]["exp"].squeeze())
        else:
            exp = np.zeros((poses.shape[0], 50))

        jaw = np.zeros((poses.shape[0], 1, 3))
        leye = jaw.copy()
        reye = jaw.copy()

        to_save = {
            # 'betas': np.zeros(10), # 10
            "betas": betas,  # 10
            "trans": transl,  # B,3
            "exp": exp,  # T,50
            "poses": np.concatenate(
                [poses[:, :22, :], jaw, leye, reye, hand_pose], axis=1
            ).reshape(-1, 55, 3)[::1],
            "gender": "male",
            "mocap_framerate": fps,
        }

        to_save |= ex_params_to_save

        for k in save_ex_keys:
            if k in results[_id]:
                to_save[k] = results[_id][k]

        if output_pth.endswith(".npz"):
            out_path = output_pth
        else:
            fname = f"ID{_id}.npz"
            if len(save_prefix) > 0:
                fname = f"{save_prefix}_{fname}"
            out_path = os.path.join(output_pth, fname)
        np.savez(out_path, **to_save)
        print(f"saved to {out_path}")


class DiffTraj:
    def __init__(self):
        device = "cuda"
        dtype = torch.float32
        I = torch.eye(3)[None].to(device=device).to(dtype=dtype)

        # ---->
        train_mean_std = "model_files/DiffTraj_models/difftraj_old_model0/amass_glamr_dump_dataloader_load_txt_traj_feet_trajRot_train_mean_std.npz"
        model_path = (
            "model_files/DiffTraj_models/traj_feet_trajRot_12_8/model000100000.pt"
        )
        use_old_model = True
        seq_len = 200

        # model_path = 'model_files/DiffTraj_models/difftraj_amass_100styles_seq128_step1000_addEmb_allAMASS/model000602325.pt'
        # train_mean_std = '/apdcephfs/share_1330077/wallyliang/ar_diff_loco_gen_save/dataloader_pts/amassData_mean_std.npz'

        # # model_path = 'model_files/DiffTraj_models/difftraj_amass_100styles_seq128_step1000_addEmb/model000612480.pt'
        # model_path = 'model_files/DiffTraj_models/difftraj_amass_100styles_seq128_step1000_addEmb_amassOnly/model000605280.pt'
        # train_mean_std = 'model_files/DiffTraj_models/amassData_noOcclusion_mean_std.npz'
        # # train_mean_std = '/apdcephfs/share_1330077/wallyliang/ar_diff_loco_gen_save/dataloader_pts/amassData_noOcclusion_mean_std.npz'
        # use_old_model = False
        # seq_len = None

        male_bm_path = "model_files/uhc_data/smpl/SMPLH_MALE.npz"
        male_dmpl_path = "model_files/uhc_data/dmpls/male/model.npz"

        self.male_bm = BodyModel(
            bm_fname=male_bm_path, num_betas=10, num_dmpls=8, dmpl_fname=male_dmpl_path
        )
        difftraj_args_path = os.path.join(os.path.dirname(model_path), "args.json")
        self.difftraj_args = json.load(open(difftraj_args_path, "r"))
        self.difftraj_args = EasyDict(self.difftraj_args)
        self.difftraj_args.dataset = "difftraj"
        self.difftraj_args.guidance_param = 2.5
        self.difftraj_args.use_old_model = use_old_model
        if seq_len is not None:
            self.difftraj_args.seq_len = seq_len

        if "use_ar" not in self.difftraj_args:
            self.difftraj_args.use_ar = False

        if self.difftraj_args.diffusion_steps > 200:
            use_ddim = 1
        else:
            use_ddim = 0
        if use_ddim:
            self.difftraj_args.diffusion_steps = 100
        self.load_con = np.load(train_mean_std)
        self.seq_len = self.difftraj_args.seq_len
        self.difftraj_model = MDM(**get_model_args(self.difftraj_args, None))
        load_model_wo_clip(
            self.difftraj_model, torch.load(model_path, map_location="cpu")
        )

        # import ipdb;ipdb.set_trace()
        if self.difftraj_args.guidance_param != 1:
            self.difftraj_model = ClassifierFreeSampleModel(
                self.difftraj_model
            )  # wrapping self.difftraj_model with the classifier-free sampler
        self.difftraj_model.cuda()
        self.male_bm = self.male_bm.to(device)

        self.difftraj_model.eval()  # disable random masking
        diffusion = create_gaussian_diffusion(self.difftraj_args)
        if use_ddim:
            self.sample_fn = diffusion.ddim_sample_loop
        else:
            self.sample_fn = diffusion.p_sample_loop
        # <----

    def pose_to_traj(self, pred_root_world, pred_body_pose, pred_trans_world):
        device = "cuda"
        dtype = torch.float32
        I = torch.eye(3)[None].to(device=device).to(dtype=dtype)

        if self.difftraj_args.use_old_model:
            local_pose_mean = self.load_con["local_pose_mean"]
            local_pose_std = self.load_con["local_pose_std"]
            local_traj_mean = self.load_con["local_traj_mean"]
            local_traj_std = self.load_con["local_traj_std"]
            # #traj(9)+feet(4)+66(pose)+vec(2)
        else:
            motoin_info_all_list_mean = self.load_con["motoin_info_all_list_mean"]
            motoin_info_all_list_std = self.load_con["motoin_info_all_list_std"]
            mean = torch.from_numpy(motoin_info_all_list_mean).type_as(I)
            std = torch.from_numpy(motoin_info_all_list_std).type_as(I)

        if pred_body_pose.shape[-2] == 21 or pred_body_pose.shape[-1] == 63:
            pose_body_all = torch.cat(
                [
                    pred_body_pose.detach().reshape(-1, 21, 3),
                    torch.zeros(pred_body_pose.shape[0], 2, 3).to(device),
                ],
                dim=1,
            )
        else:
            pose_body_all = pred_body_pose.reshape(-1, 23, 3)

        # pose_body_all = torch.cat([nemf_smooth_pose.detach().reshape(-1,21,3), torch.zeros(nemf_smooth_pose.shape[0],2,3).to(device)],dim=1)
        origin_length = pose_body_all.shape[0]
        batch = []

        has_pose = "pose" in self.difftraj_args.in_type
        has_feet = "feet" in self.difftraj_args.in_type

        orient = apply_cvt_R(
            SHOW_to_AMASS_cvt, pred_root_world, in_type="aa", out_type="aa"
        )
        trans = apply_cvt_T(SHOW_to_AMASS_cvt, pred_trans_world)

        # import ipdb;ipdb.set_trace()
        pose_seq = self.male_bm(
            pose_body=pose_body_all[:1, :21, :].reshape(-1, 63),
            root_orient=orient[:1],
            trans=trans[:1],
        ).Jtr[:, :22, :]
        min_z_first = pose_seq[0, :, 2].min()
        trans[:, 2] -= min_z_first

        local_rep_slam_cam_to_world = traj_global2local_heading(
            trans, a2q(orient.contiguous())
        )
        local_rep_slam_cam_to_world[0, :2] = 0
        local_rep_slam_cam_to_world[0, -2] = 1
        local_rep_slam_cam_to_world[0, -1] = 0
        traj_vec = local_rep_slam_cam_to_world[:, -2:]
        # import ipdb;ipdb.set_trace()

        if self.difftraj_args.use_ar:
            pose_body = pose_body_all[:, :21, :].reshape(-1, 63)
            body = self.male_bm(pose_body=pose_body)
            pose_seq = body.Jtr[:, :22, :]

            pose_body_axis = pose_body.reshape(-1, 21, 3)
            pose_body_mat = a2m(pose_body_axis)
            pose_body_6d = m2s(pose_body_mat).reshape(pose_seq.shape[0], -1)  # 21*6

            local_pose = pose_seq.reshape(pose_seq.shape[0], -1)
            precompute_motions = torch.cat(
                [local_rep_slam_cam_to_world, pose_body_6d, local_pose], dim=-1
            )
            precompute_motions = (precompute_motions - mean) / std

            # cur_point = 0
            p_motion = precompute_motions[
                self.difftraj_args.p_len * 0 : self.difftraj_args.p_len * 1
            ]
            f_motion = precompute_motions[
                self.difftraj_args.p_len * 1 : self.difftraj_args.p_len * 1
                + self.difftraj_args.f_len
            ]
            f_pose = torch.cat([f_motion[:, 11:], f_motion[:, 9 : 9 + 2]], axis=-1)
            batch = [
                {
                    "inp": torch.zeros(self.difftraj_args.f_len, 203).T,
                    "p_motion": p_motion.T.float().unsqueeze(1),
                    "f_pose": f_pose.T.float().unsqueeze(1),
                    "lengths": torch.tensor(self.difftraj_args.f_len),
                }
            ]
            _, model_kwargs_custom = collate(batch)
            model_kwargs_custom["y"]["scale"] = (
                torch.ones(1, device=device) * self.difftraj_args.guidance_param
            )
            ar_model_output_list = model_kwargs_custom["y"]["p_motion"].type_as(I)
            infer_step = 3
            for t in range(self.difftraj_args.p_len, origin_length, infer_step):
                print(f"time: {t}")

                f_motion = precompute_motions[t : t + self.difftraj_args.f_len]
                f_pose = torch.cat([f_motion[:, 11:], f_motion[:, 9 : 9 + 2]], axis=-1)

                if f_pose.shape[0] < self.difftraj_args.f_len:
                    miss_len = self.difftraj_args.f_len - f_pose.shape[0]
                    f_pose = torch.cat([f_pose, f_pose[-1:].tile(miss_len, 1)])
                    # f_pose = torch.cat([f_pose, f_pose[-miss_len:][::-1,...]])

                model_kwargs_custom["y"]["f_pose"] = (
                    f_pose[None].permute(0, 2, 1).unsqueeze(2)
                )
                model_kwargs_custom["y"]["p_motion"] = ar_model_output_list[
                    :, :, :, t - self.difftraj_args.p_len : t
                ]

                sample_custom = self.sample_fn(
                    self.difftraj_model,
                    (
                        len(batch),
                        self.difftraj_model.njoints,
                        self.difftraj_model.nfeats,
                        self.difftraj_args.f_len,
                    ),
                    clip_denoised=False,
                    model_kwargs=model_kwargs_custom,
                    skip_timesteps=0,
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
                ar_model_output_list = torch.cat(
                    [ar_model_output_list, sample_custom[:, :, :, :infer_step]], dim=-1
                )  # [1, 203, 1, T]

                # model_kwargs['y']['p_motion'] = ar_model_output_list[:,:,:,-past_motion_offset:]
            sample_custom = ar_model_output_list[:, :, :, :origin_length]
            sample_rep_pred_list = [
                sample_custom[idx, :, 0, :].permute(1, 0)
                for idx in range(sample_custom.shape[0])
            ]

        else:
            pose_chunk = split_tensor(
                pose_body_all.reshape(-1, 69), self.seq_len, difftraj_overlap_len
            )
            vec_chunk = split_tensor(traj_vec, self.seq_len, difftraj_overlap_len)

            for chunk_idx in range(len(pose_chunk)):
                pose_body = (
                    pose_chunk[chunk_idx].reshape(-1, 23, 3)[:, :21, :].reshape(-1, 63)
                )
                body = self.male_bm(pose_body=pose_body)
                pose_seq = body.Jtr[:, :22, :]

                pose_body_axis = pose_body.reshape(-1, 21, 3)
                pose_body_mat = a2m(pose_body_axis)
                pose_body_6d = m2s(pose_body_mat).reshape(pose_seq.shape[0], -1)  # 21*6

                if self.difftraj_args.use_old_model:
                    # import ipdb;ipdb.set_trace()
                    local_pose = pose_seq.reshape(pose_seq.shape[0], -1)
                    local_pose = torch.cat([local_pose, vec_chunk[chunk_idx]], dim=-1)
                    cond_mean = torch.from_numpy(local_pose_mean).type_as(I)
                    cond_std = torch.from_numpy(local_pose_std).type_as(I)
                    local_pose = (local_pose - cond_mean) / cond_std
                    batch.append(
                        {
                            "inp": torch.zeros(local_pose.shape[0], 11).T,
                            "local_pose": local_pose.T.float().unsqueeze(1),
                            "lengths": torch.tensor(200),
                        }
                    )

                else:
                    local_pose = pose_seq.reshape(pose_seq.shape[0], -1)
                    cond = torch.cat([local_pose, vec_chunk[chunk_idx]], dim=-1)
                    cond_mean = torch.cat([mean[11 + 21 * 6 :], mean[9 : 9 + 2]])
                    cond_std = torch.cat([std[11 + 21 * 6 :], std[9 : 9 + 2]])
                    cond = (cond - cond_mean) / cond_std
                    # import ipdb;ipdb.set_trace()
                    batch.append(
                        {
                            "inp": torch.zeros(cond.shape[0], 203).T,
                            "condition": cond.T.float().unsqueeze(1),
                            "lengths": torch.tensor(200),
                        }
                    )

            _, model_kwargs_custom = collate(batch)
            model_kwargs_custom["y"]["scale"] = (
                torch.ones(1, device=device) * self.difftraj_args.guidance_param
            )
            sample_custom = self.sample_fn(
                self.difftraj_model,
                (
                    len(batch),
                    self.difftraj_model.njoints,
                    self.difftraj_model.nfeats,
                    local_pose.shape[0],
                ),
                clip_denoised=False,
                model_kwargs=model_kwargs_custom,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            sample_rep_pred_list = [
                sample_custom[idx, :, 0, :].permute(1, 0)
                for idx in range(sample_custom.shape[0])
            ]

        sample_rep_pred_list = torch.stack(sample_rep_pred_list)
        sample_rep_pred = merge_tensor_w_avg(
            sample_rep_pred_list, difftraj_overlap_len, origin_length
        )

        if self.difftraj_args.use_old_model:
            cond_mean = torch.from_numpy(local_traj_mean).type_as(I)
            cond_std = torch.from_numpy(local_traj_std).type_as(I)
            sample_rep_pred = sample_rep_pred * cond_std + cond_mean
            traj_local_pred = torch.cat([sample_rep_pred[:, :9], traj_vec], dim=-1)
        else:
            traj_local_pred = sample_rep_pred * std[:11] + mean[:11]

        transl_amass_pred, orient_q_amass_pred = traj_local2global_heading(
            traj_local_pred
        )
        root = apply_cvt_R(
            AMASS_to_SHOW_cvt, orient_q_amass_pred, in_type="quat", out_type="aa"
        )
        trans = apply_cvt_T(AMASS_to_SHOW_cvt, transl_amass_pred)

        return root, trans


if __name__ == "__main__":
    traj_predictor = DiffTraj()

    def pose_to_traj(npz_file, output_pth):
        data = np.load(npz_file)
        pred_body_pose = tt(torch.from_numpy(data["poses"][:, 1 : 22 + 2, :]))
        pred_root_world = tt(torch.from_numpy(data["poses"][:, 0, :]))
        pred_trans_world = tt(torch.from_numpy(data["trans"]))

        difftraj_root, difftraj_trans = traj_predictor.pose_to_traj(
            pred_root_world, pred_body_pose, pred_trans_world
        )
        results = {
            "0": {
                "pred_body_pose": tt(pred_body_pose),
                "difftraj_root": tt(difftraj_root),
                "difftraj_trans": tt(difftraj_trans),
                "betas": tt(torch.zeros(3, 10)),
            }
        }
        save_to_blender_smplx_addon_npz(results, "difTraj_raw", output_pth=output_pth)

    inp = sys.argv[1]

    if inp.endswith(".npz"):
        pose_to_traj(inp, inp.replace(".npz", "_cvt.npz"))
    else:
        save_npz_root = inp + "_pose2Traj"
        os.makedirs(save_npz_root, exist_ok=True)
        for npz_file in glob(f"{inp}/*.npz"):
            print(f"npz_file: {npz_file}")
            output_pth = os.path.join(save_npz_root, os.path.basename(npz_file))
            pose_to_traj(npz_file, output_pth)
