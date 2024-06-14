# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os
import sys

__cur__ = os.path.join(os.path.dirname(__file__))
sys.path.append(__cur__)

from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from utils.model_util import create_gaussian_diffusion
from data_loaders.get_data import get_model_args, collate
import glob
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from scipy.interpolate import interp1d
import argparse
import yaml
import random

from data_loaders.humanml.data.dataset import *
from lib.loco.trajdiff import *
import utils.rotation_conversions as geometry


def generate_spline(num_points, step_length, max_curvature):
    points = []
    angle = 0
    x = 0
    y = 0

    for i in range(num_points):
        curvature = random.uniform(-max_curvature, max_curvature)
        angle += curvature
        x += step_length * math.cos(angle)
        y += step_length * math.sin(angle)
        points.append((x, y))

    return points


if __name__ == "__main__":
    args = generate_args()
    args.dataset = "AMASS_GLAMR_taming_style"
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    dist_util.setup_dist(args.device)

    user_transl = args.user_transl
    autogressive = "pastMotion" in args.in_type
    print(f"autogressive: {autogressive}")
    args.batch_size = args.num_samples
    # args.batch_size = 1
    use_ddim = 0
    save_gt_videos = 1
    if args.diffusion_steps > 200:
        use_ddim = 1
    if use_ddim:
        args.diffusion_steps = 100
    if user_transl:
        print(f"[WARN] use user define translation")
        after_fix = "replaceTransl"
        save_gt_videos = 0
    else:
        after_fix = "dataTransl"

    print("Loading dataset...")
    args.dataset = "diffgen"
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=1,
        split="train",
        args=args,
    )
    # data = get_dataset_loader(name=args.dataset,batch_size=args.batch_size,num_frames=1,split='test',args=args)
    # max_frames = data.dataset.t2m_dataset.fixed_length = data.dataset.t2m_dataset.inp_len
    fps = data.dataset.t2m_dataset.ex_fps
    ex_fps = data.dataset.t2m_dataset.ex_fps
    inp_len = data.dataset.t2m_dataset.inp_len
    B = args.batch_size

    print("Creating model and diffusion...")
    if args.eval_cmdm_base:
        args.train_cmdm_base = True
        from model.cmdm_style import MDM
    elif args.eval_cmdm_finetune:
        args.train_cmdm_base = False
        from model.cmdm_style import MDM
    else:
        from model.mdm_taming_style import MDM
    model = MDM(**get_model_args(args, data), inp_len=data.dataset.t2m_dataset.inp_len)
    diffusion = create_gaussian_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.cuda()
    model.eval()  # disable random masking
    if use_ddim:
        sample_fn = diffusion.ddim_sample_loop
    else:
        sample_fn = diffusion.p_sample_loop
    # from model.smpl import SMPL, JOINTSTYPE_ROOT
    # smpl_model = SMPL().eval().to('cuda')
    I = torch.ones(1).to(dist_util.dev())
    dataloader_infer_dir = f"{args.vis_save_dir}/{args.in_type}_{niter}_{after_fix}"
    os.makedirs(dataloader_infer_dir, exist_ok=True)
    kinematic_tree = skeleton = paramUtil.t2m_kinematic_chain
    trans_matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    ).type_as(I)
    mean = torch.from_numpy(data.dataset.t2m_dataset.motoin_info_all_list_mean).type_as(
        I
    )
    std = torch.from_numpy(data.dataset.t2m_dataset.motoin_info_all_list_std).type_as(I)

    iterator = iter(data)
    gt_motion, model_kwargs = next(iterator)
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        )

    # model.model.inp_len = inp_len

    style_str_to_onehot = data.dataset.t2m_dataset.style_str_to_onehot
    sytle_onhot_to_str = {v: k for k, v in style_str_to_onehot.items()}
    style_name_list = list(style_str_to_onehot.keys())
    sytle_int_to_str = {v.argmax().item(): k for k, v in style_str_to_onehot.items()}
    print(f"style_name_list: {style_name_list}")

    # motionclip
    if user_transl:
        if "motionclip" in args.in_type:
            print(f"set motionclip exmaple motion")
            from data_loaders.humanml.data.dataset import smplh52_to_smpl24
            from human_body_prior.body_model.body_model import BodyModel

            male_bm_path = "/apdcephfs/private_wallyliang/PLANTmodels/models_argol/sfiles/smplh/male/model.npz"
            male_dmpl_path = "/apdcephfs/private_wallyliang/PLANTmodels/models_argol/sfiles/dmpls/male/model.npz"
            male_bm = BodyModel(
                bm_fname=male_bm_path,
                num_betas=10,
                num_dmpls=8,
                dmpl_fname=male_dmpl_path,
            ).cuda()
            example_motion_npz = (
                "/root/apdcephfs/private_wallyliang/putin_ID0_origin_st280_et340.npz"
            )
            bdata = np.load(example_motion_npz, allow_pickle=True)
            fps = bdata["mocap_framerate"]
            down_sample = int(fps / ex_fps)
            comp_device = "cuda"
            poses = torch.Tensor(bdata["poses"])
            trans = torch.Tensor(bdata["trans"])
            root_orient = (poses[::down_sample, 0]).to(comp_device)
            pose_body = (poses[::down_sample, 1:22, :]).to(comp_device).reshape(-1, 63)
            pose_hand = (poses[::down_sample, -30:, :]).to(comp_device).reshape(-1, 90)
            trans = (trans[::down_sample]).to(comp_device)

            positions_woTrans = male_bm(
                pose_body=pose_body, pose_hand=pose_hand, root_orient=root_orient
            )
            joints3D = positions_woTrans.Jtr
            joints3D = joints3D - joints3D[0, 0, :]
            all_poses = torch.cat(
                [
                    root_orient.unsqueeze(1),
                    pose_body.reshape(-1, 21, 3),
                    pose_hand.reshape(-1, 30, 3),
                ],
                dim=1,
            )
            pose = all_poses[:, smplh52_to_smpl24, :]
            pose = m2s(a2m(pose))
            padded_tr = torch.zeros((pose.shape[0], pose.shape[2])).type_as(pose)
            padded_tr[:, :3] = joints3D[:, 0, :]
            motion_clip_inp = torch.cat((pose, padded_tr[:, None]), 1)

            if "preCompute" in args.in_type:
                motoin_clip_model = get_motionclip_model()
                example_motoin = motion_clip_inp.permute(1, 2, 0).unsqueeze(
                    0
                )  # 1, 25, 6, 60
                batch = {
                    "x": example_motoin,
                    "y": torch.zeros(1).long().to(example_motoin.device),
                    "mask": torch.ones(1, 60).bool().to(example_motoin.device),
                    "lengths": torch.ones(1, 60)
                    .bool()
                    .to(example_motoin.device)
                    .data.fill_(60),
                }
                batch = motoin_clip_model(batch)
                example_motoin_embedding = batch["z"].unsqueeze(2)
                # import ipdb;ipdb.set_trace()
                model_kwargs["y"]["example_motoin"] = example_motoin_embedding.repeat(
                    args.batch_size, 1, 1
                )  # bs, 512, 1
            else:
                # model_kwargs['y']['example_motoin'] #torch.Size([10, 6, 1, 25, 60])
                model_kwargs["y"]["example_motoin"] = (
                    motion_clip_inp.permute(2, 1, 0)
                    .unsqueeze(1)
                    .repeat(args.batch_size, 1, 1, 1, 1)
                )

    ar_model_output_list = model_kwargs["y"]["past_motion"].type_as(I)
    past_motion_offset = data.dataset.t2m_dataset.pastMotion_len
    gen_target_T = ex_fps * args.gen_time_length
    save_gt_videos = 0
    displacement_per_step = args.infer_walk_speed / ex_fps
    infer_step = args.infer_step
    difine_traj = args.difine_traj

    if difine_traj == "line":
        # # <-------------
        after_fix = "straintLine"
        target_transl = torch.zeros(gen_target_T, 2).type_as(I)
        for i in range(gen_target_T):
            target_transl[i, 0] = displacement_per_step * i
        # transl = target_transl[None,...]

    if difine_traj == "rec":
        # <-------------
        after_fix = "Rectangle"
        len_meter = displacement_per_step * (gen_target_T / 4)

        # 生成矩形的四个角点
        rect_corners = torch.tensor(
            [[0, 0], [len_meter, 0], [len_meter, len_meter], [0, len_meter]]
        )

        # 计算每条边的插值步数
        num_steps_per_edge = gen_target_T // 4
        transl = torch.zeros(1, gen_target_T, 2).type_as(I)

        for i in range(4):
            start_corner = rect_corners[i]
            end_corner = rect_corners[(i + 1) % 4]

            # 计算每步的位移
            step_displacement = (end_corner - start_corner) / num_steps_per_edge

            # 插值
            for j in range(num_steps_per_edge):
                transl[:, i * num_steps_per_edge + j] = (
                    start_corner + step_displacement * j
                )

        target_transl = transl[0]
        # ----->
    # # <-------------
    if difine_traj == "circle":
        after_fix = "Circle"
        run_angle = 2 * math.pi
        circle_radius = displacement_per_step * gen_target_T / run_angle
        angular_speed = run_angle / gen_target_T  # 每个时间步的角速度
        target_transl = torch.zeros(gen_target_T, 2).type_as(I)

        # 遍历每个时间步，计算圆形轨迹上的点
        for i in range(gen_target_T):
            angle = i * angular_speed
            x = circle_radius * math.cos(angle)
            y = circle_radius * math.sin(angle)
            target_transl[i] = torch.tensor([x, y])

        # transl = target_transl[None, ...]

    # import ipdb;ipdb.set_trace()
    if difine_traj == "spline":
        after_fix = "spline"
        max_curvature = 0.4
        target_transl = generate_spline(
            gen_target_T, displacement_per_step, max_curvature
        )
        target_transl = torch.tensor(target_transl)
        # transl = target_transl[None, ...]

    if args.traj_rev_dir:
        target_transl[:, 1] *= -1
        after_fix += "_dir1"
    else:
        after_fix += "_dir0"

    after_fix += f"_speed{args.infer_walk_speed:.1f}"

    if gen_target_T > 25:
        from scipy.signal import savgol_filter

        trajectory_np = target_transl.cpu().numpy()
        smoothed_trajectory_np = savgol_filter(
            trajectory_np, window_length=25, polyorder=3, axis=0
        )
        smoothed_trajectory = torch.tensor(smoothed_trajectory_np)
        target_transl = smoothed_trajectory
    # ----->

    look_ahead_len = past_motion_offset * infer_step
    target_transl = target_transl.cuda()
    target_transl = torch.cat(
        [target_transl, target_transl[-1].repeat(look_ahead_len, 1)], dim=0
    )

    if args.traj_face_type:
        # face fixed 1,0 vector
        after_fix += "_Face0"
        target_transl_traj = traj2d_global_to_local(
            target_transl,
            torch.tensor([1.0, 0])
            .repeat(target_transl.shape[0], 1)
            .type_as(target_transl),
        )
    else:
        # face traj foreward
        after_fix += "_Face1"
        tangent_vectors, tangent_scale = transl_xy_to_tangent(target_transl)
        target_transl_traj = traj2d_global_to_local(
            target_transl, tangent_vectors.type_as(target_transl)
        )

    target_transl_traj[0, :2] = 0
    target_transl_traj[0, 2] = 1
    target_transl_traj[0, 3] = 0
    target_transl, _ = traj2d_local_to_global(target_transl_traj)

    cur_target_pos_idx = look_ahead_len
    ar_model_output_list = torch.cat(
        [ar_model_output_list, ar_model_output_list[:, :, :, -infer_step:]], dim=-1
    )  # [1, 203, 1, T]
    future_traj = (
        (model_kwargs["y"]["past_motion"][0, :11, 0, :])
        .mean(-1)
        .tile(past_motion_offset, 1)
        .permute(1, 0)
        .cuda()
    )  # 11, （T）
    future_traj[:2, :] = 0
    prev_cur2target_trans = None
    seted_debug = False

    for t in range(0, gen_target_T, infer_step):
        print(f"time: {t}")
        ctrl_traj = target_transl_traj[t : t + inp_len][None, ...].repeat(
            args.batch_size, 1, 1
        )
        if ctrl_traj.shape[1] != inp_len:
            break

        # import ipdb;ipdb.set_trace()
        if "normCtrlTraj" in args.in_type:
            ctrl_traj = (
                ctrl_traj
                - torch.from_numpy(data.dataset.t2m_dataset.ctrl_signal_mean)[
                    None, None, ...
                ].type_as(ctrl_traj)
            ) / torch.from_numpy(data.dataset.t2m_dataset.ctrl_signal_std)[
                None, None, ...
            ].type_as(
                ctrl_traj
            )

        model_kwargs["y"]["ctrl_traj"] = ctrl_traj.permute(0, 2, 1).unsqueeze(2)
        model_kwargs["y"]["past_motion"] = ar_model_output_list[
            :, :, :, -past_motion_offset:
        ]
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, inp_len),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        ar_model_output_list = torch.cat(
            [ar_model_output_list, sample[:, :, :, :infer_step]], dim=-1
        )  # [1, 203, 1, T]
        future_traj = sample[0, :11, 0, :]  # inp_len

    ar_model_output_list = ar_model_output_list[:, :, :, past_motion_offset:]
    sample = ar_model_output_list
    gt_motion = gt_motion.to(dist_util.dev())
    std = std[:137]
    mean = mean[:137]
    sample = sample[:, :137, :, :]
    gt_motion = gt_motion[:, :137, :, :]
    # import ipdb;ipdb.set_trace()
    from eval.smpl_utils import AnyRep2SMPLjoints

    traj2joints = AnyRep2SMPLjoints()

    for bs_idx in range(sample.shape[0]):
        # import ipdb;ipdb.set_trace()
        if "example_motoin" in model_kwargs["y"]:
            caption = sytle_int_to_str[
                model_kwargs["y"]["example_motoin"][bs_idx].argmax().item()
            ]
        else:
            caption = "None"

        print(f"generating sample videos")
        sample_rep_pred = sample[bs_idx, :, 0, :].permute(1, 0)
        sample_rep_pred = sample_rep_pred * std + mean

        if 1:
            temp_traj = torch.zeros_like(sample_rep_pred[:, :11])
            temp_traj[:, -2] = 1
            sample_rep_pred = torch.cat([temp_traj, sample_rep_pred[:, 11:]], dim=-1)

        global_pose = traj2joints.traj_to_joints(sample_rep_pred, zup_to_yup=True)
        save_path = f"{dataloader_infer_dir}/{caption}_pred_{bs_idx}_{after_fix}.mp4"
        plot_3d_motion(
            save_path,
            skeleton,
            global_pose.cpu().numpy(),
            dataset="humanml",
            title=f"style: {caption}, predict",
            fps=fps,
        )
        print(f"{save_path}")
        out_path = f"{dataloader_infer_dir}/{caption}_pred_{bs_idx}_{after_fix}.npz"
        traj2joints.traj_to_npz(sample_rep_pred, out_path)

        # get GT motions
        if save_gt_videos:
            print(f"generating GT videos")
            sample_rep_gt = gt_motion[bs_idx, :, 0, :].permute(1, 0)
            motion_len = model_kwargs["y"]["lengths"][bs_idx]
            sample_rep_gt = sample_rep_gt[:motion_len]
            sample_rep_gt = sample_rep_gt * std + mean

            if 1:
                sample_rep_gt = torch.cat([temp_traj, sample_rep_gt[:, 11:]], dim=-1)

            global_pose = traj2joints.traj_to_joints(sample_rep_pred, zup_to_yup=True)
            save_path = f"{dataloader_infer_dir}/{caption}_gt_{bs_idx}.mp4"
            plot_3d_motion(
                save_path,
                skeleton,
                global_pose.cpu().numpy(),
                dataset="humanml",
                title=f"style: {caption}, gt",
                fps=fps,
            )
            print(f"{save_path}")
