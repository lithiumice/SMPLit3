# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import sys

__cur__ = os.path.join(os.path.dirname(__file__))
sys.path.append(__cur__)

import os
import torch
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

from einops import rearrange
import ipdb


if __name__ == "__main__":
    args = generate_args()
    # import ipdb;ipdb.set_trace()
    print(args)
    fixseed(args.seed)
    out_path = args.output_dir
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
    from model.mdm_traj import MDM

    model = MDM(**get_model_args(args, data))
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
    mean_203 = torch.from_numpy(data.dataset.t2m_dataset.motoin_info_all_list_mean).type_as(I)
    std_203 = torch.from_numpy(data.dataset.t2m_dataset.motoin_info_all_list_std).type_as(I)

    iterator = iter(data)
    gt_motion, model_kwargs = next(iterator)
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        )

    style_str_to_onehot = data.dataset.t2m_dataset.style_str_to_onehot
    sytle_onhot_to_str = {v: k for k, v in style_str_to_onehot.items()}
    style_name_list = list(style_str_to_onehot.keys())
    sytle_int_to_str = {v.argmax().item(): k for k, v in style_str_to_onehot.items()}
    print(f"style_name_list: {style_name_list}")

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
        sample_rep_pred_full[:,:11] = sample_rep_pred

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
