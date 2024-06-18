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
import numpy as np
import torch
import shutil
import glob
import math
import numpy as np
import random
from einops import rearrange

from utils.parser_util import generate_args
from utils.model_util import load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from utils.model_util import create_gaussian_diffusion
from data_loaders.get_data import get_model_args
from utils.fixseed import fixseed
from eval.smpl_utils import AnyRep2SMPLjoints
from glob import glob

import ipdb


def get_motion_clip_inp():
    print(f"set motionclip exmaple motion")
    from data_loaders.dataset import smplh52_to_smpl24
    from human_body_prior.body_model.body_model import BodyModel
    male_bm = BodyModel(
        bm_fname=male_bm_path,
        num_betas=10,
        num_dmpls=8,
        dmpl_fname=male_dmpl_path,
    ).cuda()
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
    return motion_clip_inp


def generate_line(gen_target_T, displacement_per_step):
    target_transl = torch.zeros(gen_target_T, 2).type_as(I)
    for i in range(gen_target_T):
        target_transl[i, 0] = displacement_per_step * i
    return target_transl

def generate_rectangle(gen_target_T, displacement_per_step):
    len_meter = displacement_per_step * (gen_target_T / 4)

    rect_corners = torch.tensor(
        [[0, 0], [len_meter, 0], [len_meter, len_meter], [0, len_meter]]
    )

    num_steps_per_edge = gen_target_T // 4
    transl = torch.zeros(1, gen_target_T, 2).type_as(I)

    for i in range(4):
        start_corner = rect_corners[i]
        end_corner = rect_corners[(i + 1) % 4]

        step_displacement = (end_corner - start_corner) / num_steps_per_edge

        for j in range(num_steps_per_edge):
            transl[:, i * num_steps_per_edge + j] = (
                start_corner + step_displacement * j
            )

    target_transl = transl[0]
    return target_transl
    
def geenrate_circle(gen_target_T, displacement_per_step):
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
    return target_transl


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

def generate_spline2(gen_target_T, displacement_per_step):
    max_curvature = 0.4
    target_transl = generate_spline(
        gen_target_T, displacement_per_step, max_curvature
    )
    target_transl = torch.tensor(target_transl)
    return target_transl
    
def smooth_trajectory(target_transl):
    from scipy.signal import savgol_filter

    trajectory_np = target_transl.cpu().numpy()
    smoothed_trajectory_np = savgol_filter(
        trajectory_np, window_length=25, polyorder=3, axis=0
    )
    smoothed_trajectory = torch.tensor(smoothed_trajectory_np)
    target_transl = smoothed_trajectory
    return target_transl
    
if __name__ == "__main__":
    args = generate_args()
    print(args)

    fixseed(args.seed)
    out_path = args.output_dir

    # args.model_path can be root dir or model.pt
    if not str(args.model_path).endswith(".pt"):
        checkpoints = glob(f"{args.model_path}/model-*.pt")
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )
        args.model_path = checkpoints[-1]
        print(f"{args.model_path=} is set.")

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    dist_util.setup_dist(args.device)

    args.batch_size = args.num_samples

    use_ddim = 0
    save_gt_videos = 1
    if args.diffusion_steps > 200:
        use_ddim = 1

    if use_ddim:
        args.diffusion_steps = 100

    print("Loading dataset...")
    args.dataset = "difftraj"
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=1,
        split="test",
        args=args,
    )
    B = args.batch_size
    fps = 30

    print("Creating model and diffusion...")
    if args.dataset == "difftraj":
        from model.mdm_traj import MDM

        model = MDM(**get_model_args(args, data))
    elif args.dataset == "diffpose":
        from model.model_diffpose import MODEL_DIFFPOSE

        model = MODEL_DIFFPOSE(**get_model_args(args, data))
    elif args.dataset == "diffgen":
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

    I = torch.ones(1).to(dist_util.dev())
    dataloader_infer_dir = args.vis_save_dir
    os.makedirs(dataloader_infer_dir, exist_ok=True)
    kinematic_tree = skeleton = paramUtil.t2m_kinematic_chain
    trans_matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    ).type_as(I)
    mean_203 = torch.from_numpy(
        data.dataset.t2m_dataset.motoin_info_all_list_mean
    ).type_as(I)
    std_203 = torch.from_numpy(
        data.dataset.t2m_dataset.motoin_info_all_list_std
    ).type_as(I)

    iterator = iter(data)
    gt_motion, model_kwargs = next(iterator)
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        )

    style_str_to_onehot = data.dataset.t2m_dataset.style_str_to_onehot
    sytle_onhot_to_str = {v: k for k, v in style_str_to_onehot.items()}
    style_name_list = list(style_str_to_onehot.keys())
    print(f"style_name_list: {style_name_list}")
    sytle_int_to_str = {v.argmax().item(): k for k, v in style_str_to_onehot.items()}


    # motionclip
    if False:
        if "motionclip" in args.in_type:
            motion_clip_inp = get_motion_clip_inp()
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
                
                
    if args.dataset == "diffgen":
    # ipdb.set_trace()
        from data_loaders.dataset import (
            transl_xy_to_tangent,
            traj2d_global_to_local,
            traj2d_local_to_global
        )
        inp_len = data.dataset.t2m_dataset.inp_len
        ex_fps = data.dataset.t2m_dataset.ex_fps 
        gen_target_T = ex_fps * args.gen_time_length
        displacement_per_step = args.infer_walk_speed / ex_fps
        target_transl = generate_line(gen_target_T, displacement_per_step)
        past_motion_offset = data.dataset.t2m_dataset.p_len
        infer_step = args.infer_step
        
        if args.traj_rev_dir:
            target_transl[:, 1] *= -1
        
        look_ahead_len = past_motion_offset * infer_step
        target_transl = target_transl.cuda()
        target_transl = torch.cat(
            [target_transl, target_transl[-1].repeat(look_ahead_len, 1)], dim=0
        )

        if args.traj_face_type:
            # face fixed 1,0 vector
            target_transl_traj = traj2d_global_to_local(
                target_transl,
                torch.tensor([1.0, 0])
                .repeat(target_transl.shape[0], 1)
                .type_as(target_transl),
            )
        else:
            # face traj foreward
            tangent_vectors, tangent_scale = transl_xy_to_tangent(target_transl)
            target_transl_traj = traj2d_global_to_local(
                target_transl, tangent_vectors.type_as(target_transl)
            )

        target_transl_traj[0, :2] = 0
        target_transl_traj[0, 2] = 1
        target_transl_traj[0, 3] = 0
        target_transl, _ = traj2d_local_to_global(target_transl_traj)

        ar_model_output_list = model_kwargs["y"]["past_motion"].type_as(I)
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
            if args.normlize_ctrl_traj:
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

        sample = ar_model_output_list[:, :, :, past_motion_offset:]
        
    elif args.dataset == "difftraj":
        # runing forward
        sample = sample_fn(
            model,
            (
                args.batch_size,
                model.njoints,
                model.nfeats,
                args.f_len if args.use_ar else 200,
            ),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        
    
    gt_motion = gt_motion.to(dist_util.dev())
    std_137 = std_203[:137]
    mean_137 = mean_203[:137]
    std_11 = std_203[:11]
    mean_11 = mean_203[:11]
    sample = sample[:, :137, :, :]
    gt_motion = gt_motion[:, :137, :, :]

    # just visualize
    traj2joints = AnyRep2SMPLjoints()

    for bs_idx in range(sample.shape[0]):
        if "example_motoin" in model_kwargs["y"]:
            # for 100styles
            caption = sytle_int_to_str[
                model_kwargs["y"]["example_motoin"][bs_idx].argmax().item()
            ]
        else:
            caption = "None"

        print(f"generating sample videos")
        motoin_info_all = model_kwargs["y"]["motoin_info_all"][bs_idx].clone()
        motoin_info_all = rearrange(motoin_info_all, "d 1 f -> f d")
        motoin_info_all = motoin_info_all.cuda() * std_203 + mean_203
        sample_rep_gt = motoin_info_all.clone()

        # ipdb.set_trace()
        sample_rep_pred = sample[bs_idx, :, 0, :].permute(1, 0)
        sample_rep_pred = sample_rep_pred * std_11 + mean_11
        sample_rep_pred_full = motoin_info_all.clone()
        sample_rep_pred_full[:, :11] = sample_rep_pred

        global_pose = traj2joints.traj_to_joints(sample_rep_pred_full, zup_to_yup=True)
        save_path = f"{dataloader_infer_dir}/{caption}_pred_{bs_idx}.mp4"
        plot_3d_motion(
            save_path,
            skeleton,
            global_pose.cpu().numpy(),
            dataset="humanml",
            title=f"style: {caption}, predict",
            fps=fps,
        )
        print(f"{save_path}")
        out_path = f"{dataloader_infer_dir}/{caption}_pred_{bs_idx}.npz"
        traj2joints.traj_to_npz(sample_rep_pred, out_path)

        # get GT motions
        if save_gt_videos:
            print(f"generating GT videos")
            global_pose = traj2joints.traj_to_joints(sample_rep_gt, zup_to_yup=True)
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
