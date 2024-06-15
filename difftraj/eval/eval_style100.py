from collections import OrderedDict
from diffusion import logger
import torch
import joblib
from datetime import datetime
from eval.smpl_utils import AnyRep2SMPLjoints
from copy import deepcopy
from easydict import EasyDict
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from glob import glob
import json
import sys

from utils import dist_util
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from model.mdm_taming_style import MDM
from utils.model_util import create_gaussian_diffusion
from model.cfg_sampler import ClassifierFreeSampleModel
from utils.model_util import load_model_wo_clip


from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from data_loaders.humanml.motion_loaders.comp_v6_model_dataset import (
    TamingGeneratedDataset,
)
from data_loaders.get_data import get_model_args, collate, get_dataset_loader

from eval.t2m.networks.modules import (
    StylePredictorFromRawPose,
    MovementConvEncoder2,
    MovementConvDecoder2,
    MovementConvEncoder3,
    MovementConvDecoder3,
)

torch.multiprocessing.set_sharing_strategy("file_system")
pd.options.display.float_format = "{:.4f}".format


# for 100 styles evaluation
def get_taming_loader(
    model,
    diffusion,
    batch_size,
    ground_truth_loader,
    mm_num_samples,
    mm_num_repeats,
    max_motion_length,
    num_samples_limit,
    scale,
    args,
):
    dataset = TamingGeneratedDataset(
        model,
        diffusion,
        ground_truth_loader,
        mm_num_samples,
        mm_num_repeats,
        max_motion_length,
        num_samples_limit,
        scale,
        args,
    )
    motion_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=default_collate,
        drop_last=True,
        num_workers=0,
    )
    print("Generated Dataset Loading Completed!!!")
    return motion_loader


# # FGD AE, window size 24
# ckpt_file_path = 't2m_checkpoints/style100_AE_FMD_win60_fps30+latest.tar'

# # FID AE triplet loss, window size 24
# ckpt_file_path = 't2m_checkpoints/style100_AE_FID_triplet_win60_fps30+latest.tar'

# FID AE triplet loss, window size 32
# opt.window_size = 32
# ckpt_file_path = 't2m_checkpoints/style100_AE_FID_triplet_win32_fps30+latest.tar'


class eval_agent:
    def init_style_pred(self, t2m_opt, dim_pose, device, style_predictor_model_path):
        style_predictor = StylePredictorFromRawPose(
            t2m_opt, dim_pose, 256, 256, latent_dim=512, classes_num=100
        ).to(device)
        # t2m_checkpoints/style_predictor_from_motion.pt
        style_predictor.load_state_dict(
            torch.load(
                style_predictor_model_path,
                map_location=device,
            )["style_predictor"]
        )
        style_predictor.eval()

    def init_fmd1(self):
        # FGD AE, window size 24
        dim_pose = 203
        opt = EasyDict()
        opt.window_size = 24
        movement_enc = MovementConvEncoder2(dim_pose, 256, 256)
        movement_dec = MovementConvDecoder2(256, 256, dim_pose)
        ckpt_file_path = "t2m_checkpoints/style100_AE_FMD_noStylePred+latest.tar"

    def init_fid1(self):
        # FID AE triplet loss, window size 24
        dim_pose = 203
        opt = EasyDict()
        opt.use_vae = False
        opt.window_size = 24
        ckpt_file_path = "t2m_checkpoints/style100_triplet_win24_FID_AE+latest.tar"
        movement_enc = MovementConvEncoder3(opt, dim_pose, 256, 256)
        movement_dec = MovementConvDecoder3(opt, 256, 256, dim_pose)

    def init_style_vae(self):
        # FID, VAE with style predict, window size 24
        dim_pose = 203
        opt = EasyDict()
        opt.use_vae = True
        opt.window_size = 24
        ckpt_file_path = "t2m_checkpoints/style100_VAE_tripletLoss_stylePred+latest.tar"
        movement_enc = MovementConvEncoder3(opt, dim_pose, 256, 256)
        movement_dec = MovementConvDecoder3(opt, 256, 256, dim_pose)

    def init_fgd(self):
        # FGD AE, window size 24
        dim_pose = 203
        opt = EasyDict()
        opt.use_vae = False
        opt.window_size = 60
        ckpt_file_path = "t2m_checkpoints/style100_AE_FMD_win60_fps30+latest.tar"
        movement_enc = MovementConvEncoder3(opt, dim_pose, 256, 256)
        movement_dec = MovementConvDecoder3(opt, 256, 256, dim_pose)

    def init_fid(self, device):
        dim_pose = 203
        opt = EasyDict()
        opt.use_vae = False
        opt.window_size = 24
        ckpt_file_path = (
            "style100/style100_triplet_win24_FID_AE/model/E0270.tar"
            # "t2m_checkpoints/style100_AE_FID_triplet_win24_fps30+latest.tar"
        )
        print(f"load from {ckpt_file_path}")
        checkpoint = torch.load(ckpt_file_path, map_location=device)
        movement_enc.load_state_dict(checkpoint["movement_enc"])
        movement_dec.load_state_dict(checkpoint["movement_dec"])
        movement_enc = MovementConvEncoder3(opt, dim_pose, 256, 256)
        movement_dec = MovementConvDecoder3(opt, 256, 256, dim_pose)
        movement_enc = movement_enc.to(device)
        movement_enc.eval()
        movement_dec = movement_dec.to(device)
        movement_dec.eval()

        ckpts_root = os.path.abspath(f"{ckpt_file_path}/../..")
        mean_std_path = glob(f"{ckpts_root}/*_train_mean_std.npz")
        load_con = np.load(mean_std_path[0])
        ae_mean = torch.from_numpy(load_con["ae_mean"]).cpu()
        ae_std = torch.from_numpy(load_con["ae_std"]).cpu()
        return movement_enc, movement_dec, ae_mean, ae_std

    def __init__(
        self,
        t2m_opt,
        ckpt_file_path,
        dim_pose=203,
        device="cuda",
        use_standalone_style_pred=False,
        style_predictor_model_path=None,
    ) -> None:
        self.use_standalone_style_pred = use_standalone_style_pred
        movement_enc, movement_dec, ae_mean, ae_std = self.init_fid(device)
        self.movement_enc = movement_enc
        self.movement_dec = movement_dec
        self.ae_mean = ae_mean
        self.ae_std = ae_std


def evaluation(
    eval_wrapper,
    gt_loader,
    eval_motion_loaders,
    log_file,
    replication_times,
    diversity_times,
    mm_num_times,
    run_mm=False,
):
    gen_mean = torch.from_numpy(
        gt_loader.dataset.t2m_dataset.motoin_info_all_list_mean
    ).cpu()
    gen_std = torch.from_numpy(
        gt_loader.dataset.t2m_dataset.motoin_info_all_list_std
    ).cpu()

    with open(log_file, "w") as f:

        def print_mod(s):
            print(s)
            print(
                s,
                file=f,
                flush=True,
            )

        print = print_mod
        metric_template = OrderedDict(
            {
                "fid": OrderedDict({}),
                "accel": OrderedDict({}),
                "fs": OrderedDict({}),
                "t_err": OrderedDict({}),
                "o_err": OrderedDict({}),
                "diversity": OrderedDict({}),
                "emb_dist": OrderedDict({}),
                "acc_radio": OrderedDict({}),
            }
        )

        all_metrics = deepcopy(metric_template)
        for replication in range(replication_times):
            print(
                f"==================== Replication {replication} ===================="
            )
            activation_dict = {}
            motion_loaders = {}
            motion_loaders["gt"] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader

            file = f
            all_eval_dict = deepcopy(metric_template)
            all_infos_dict = {}
            save_keys = [
                "motion_embeddings",
            ]
            if eval_joints_metrics:
                save_keys.extend(["denorm_target_motion", "pred_joint"])
            if eval_wrapper.use_standalone_style_pred:
                save_keys.extend(["acc_radio"])
            for motion_loader_name, motion_loader in motion_loaders.items():
                all_motion_embeddings = []
                # all_joints = []
                all_infos = {}
                score_list = []
                all_size = 0
                matching_score_sum = 0
                top_k_count = 0
                print(f"motion_loader_name: {motion_loader_name}")

                # if motion_loader_name == "gt":
                #     if persist_gt_emb and os.path.exists(persist_gt_emb_path):
                #         content = joblib.load(persist_gt_emb_path)
                #         activation_dict["gt"] = content["gt_acit"]
                #         all_infos_dict["gt"] = content["gt_infos"]
                #         print(f"load gt emb from {persist_gt_emb_path}")
                #         continue

                with torch.no_grad():
                    for idx, batch in tqdm(enumerate(motion_loader)):
                        # import ipdb;ipdb.set_trace()
                        # print(f'len(batch): {len(batch)}')
                        if len(batch) == 2:
                            motions, m_lens = batch
                            target_motion = motions
                        else:
                            (
                                example_motoin,
                                target_motion,
                                past_motion,
                                ctrl_traj,
                            ) = batch

                        # import ipdb;ipdb.set_trace()
                        denorm_target_motion = target_motion * gen_std + gen_mean
                        ae_motions = (
                            denorm_target_motion - eval_wrapper.ae_mean
                        ) / eval_wrapper.ae_std
                        ae_motions = ae_motions.cuda().float()

                        if eval_wrapper.use_standalone_style_pred:
                            motion_embeddings, pred_style_prob = (
                                eval_wrapper.style_predictor(ae_motions)
                            )
                            # import ipdb;ipdb.set_trace()
                            pred_style = torch.argmax(pred_style_prob, dim=1).cpu()
                            gt_style = torch.argmax(example_motoin, dim=1).cpu()
                            same_indices = gt_style == pred_style
                            acc_radio = torch.nonzero(same_indices).size(
                                0
                            ) / torch.numel(same_indices)
                            acc_radio = torch.tensor([acc_radio])
                        else:
                            motion_embeddings, _, _, pred_style_prob = (
                                eval_wrapper.movement_encoder(ae_motions)
                            )

                        if eval_joints_metrics:
                            pred_joint = torch.stack(
                                [
                                    traj2joints.traj_to_joints(
                                        denormed_traj_motion, remove_float=True
                                    )
                                    for denormed_traj_motion in denorm_target_motion
                                ]
                            )
                            pred_joint = pred_joint[:, :, :22, :].cpu()

                        for k in save_keys:
                            if not k in all_infos.keys():
                                all_infos[k] = []
                            all_infos[k].append(locals()[k])

                        all_motion_embeddings.append(motion_embeddings.cpu().numpy())

                    for k in all_infos.keys():
                        all_infos[k] = torch.cat(all_infos[k], dim=0)
                    all_infos_dict[motion_loader_name] = all_infos

                    all_motion_embeddings = np.concatenate(
                        all_motion_embeddings, axis=0
                    )
                    activation_dict[motion_loader_name] = all_motion_embeddings

            # # save activation_dict['gt'] to file
            # if persist_gt_emb and not os.path.exists(persist_gt_emb_path):
            #     joblib.dump(
            #         {
            #             "gt_acit": activation_dict["gt"],
            #             "gt_infos": all_infos_dict["gt"],
            #         },
            #         persist_gt_emb_path,
            #     )
            #     print(f"save gt acit to {persist_gt_emb_path}")

            # import ipdb;ipdb.set_trace()
            if "vald" in activation_dict:
                num_samples_limit = 1000
                gt_emb = activation_dict["gt"][:num_samples_limit]
                vald_emb = activation_dict["vald"][:num_samples_limit]
                gt_emb = torch.from_numpy(gt_emb)
                vald_emb = torch.from_numpy(vald_emb)
                dist = torch.norm(gt_emb - vald_emb, dim=-1)
                dist = dist.mean()
                print(f"emb dist: {dist}")
                # all_eval_dict['emb_dist']['gt'] = dist
                all_eval_dict["emb_dist"]["vald"] = dist

            print("========== Evaluating FID ==========")
            gt_mu, gt_cov = calculate_activation_statistics(activation_dict["gt"])
            for model_name, motion_embeddings in activation_dict.items():
                mu, cov = calculate_activation_statistics(motion_embeddings)
                # print(f'gt_mu: {gt_mu}')
                # print(f'mu: {mu}')
                fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
                print(f"---> [{model_name}] FID: {fid:.4f}")
                # if not 'fid' in all_eval_dict.keys(): all_eval_dict['fid'] = {}
                all_eval_dict["fid"][model_name] = fid

            print("========== Evaluating Diversity ==========")
            for model_name, motion_embeddings in activation_dict.items():
                diversity = calculate_diversity(motion_embeddings, diversity_times)
                # eval_dict[model_name] = diversity
                print(f"---> [{model_name}] Diversity: {diversity:.4f}")
                all_eval_dict["diversity"][model_name] = diversity

            if eval_wrapper.use_standalone_style_pred:
                # import ipdb;ipdb.set_trace()
                for model_name, model_infos in all_infos_dict.items():
                    all_eval_dict["acc_radio"][model_name] = (
                        model_infos["acc_radio"].mean().item()
                    )

            if eval_joints_metrics:
                print("========== Evaluating Foot Skating ==========")
                gt_traj = all_infos_dict["gt"]["denorm_target_motion"]
                for model_name, model_infos in all_infos_dict.items():
                    # import ipdb;ipdb.set_trace()
                    joints = model_infos["pred_joint"]
                    accel = joints[:, :-1] - joints[:, 1:]
                    dist = torch.norm(accel, dim=-1) * (20)
                    accel = dist.mean()
                    line = f"---> [{model_name}] accel: {accel}"
                    print(line)
                    all_eval_dict["accel"][model_name] = accel.item()

                    CONTACT_TOE_HEIGHT_THRESH = 0.04
                    CONTACT_ANKLE_HEIGHT_THRESH = 0.08
                    FOOT_IDX = [7, 8, 10, 11]

                    dt = 1.0 / 20
                    data_seq = joints
                    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / dt
                    middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2 * dt)
                    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / dt
                    vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1)
                    vel = vel_seq

                    vel = vel[:, :, FOOT_IDX]  # L_Ankle, R_Ankle, L_Foot, R_Foot
                    vel_norm = vel[:, :, :, :2].norm(dim=-1, p=2)
                    vel_norm *= dt

                    h = joints[:, :, FOOT_IDX, 2]
                    ankle_contact = h[:, :, [0, 1]] < CONTACT_ANKLE_HEIGHT_THRESH
                    toe_contact = h[:, :, [2, 3]] < CONTACT_TOE_HEIGHT_THRESH
                    contact = torch.cat(
                        (
                            ankle_contact[:, :, 0, None],
                            toe_contact[:, :, 0, None],
                            ankle_contact[:, :, 1, None],
                            toe_contact[:, :, 1, None],
                        ),
                        dim=-1,
                    )

                    selected = list(range(joints.shape[1]))
                    v = vel_norm[:, selected] * contact[:, selected]
                    h = h[:, selected] * contact[:, selected]
                    h[:, :, [0, 1]] /= CONTACT_ANKLE_HEIGHT_THRESH
                    h[:, :, [2, 3]] /= CONTACT_TOE_HEIGHT_THRESH
                    h = torch.clamp(h, 0, 1)  # clamp the exponent between 0 and 1
                    s = v * (2 - torch.pow(2, h)) * 1
                    fs = s.mean()
                    line = f"---> [{model_name}] fs: {fs:.4f}"
                    print(line)
                    all_eval_dict["fs"][model_name] = fs.item()
                    # import ipdb;ipdb.set_trace()

                    vald_traj = model_infos["denorm_target_motion"]
                    real_bs_len = min(len(gt_traj), len(vald_traj))
                    gt_traj = gt_traj[:real_bs_len]
                    vald_traj = vald_traj[:real_bs_len]
                    trans_err = (vald_traj[:, :, :2] - gt_traj[:, :, :2]).norm(dim=-1)
                    orient_err = (vald_traj[:, :, 3:9] - gt_traj[:, :, 3:9]).norm(
                        dim=-1
                    )
                    t_err = trans_err.mean()
                    o_err = orient_err.mean()
                    line = f"---> [{model_name}] t_err: {t_err}"
                    print(line)
                    line = f"---> [{model_name}] o_err: {o_err}"
                    print(line)
                    all_eval_dict["t_err"][model_name] = t_err.item()
                    all_eval_dict["o_err"][model_name] = o_err.item()

            print(
                f"==================== Replication {replication} ===================="
            )
            print(
                tabulate(pd.DataFrame(all_eval_dict), headers="keys", tablefmt="grid")
            )
            for eval_metric_name, model_value_dict in all_eval_dict.items():
                for key, item in model_value_dict.items():
                    if key not in all_metrics[eval_metric_name]:
                        all_metrics[eval_metric_name][key] = [item]
                    else:
                        all_metrics[eval_metric_name][key] += [item]

        mean_dict = {}
        conf_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print("========== %s Summary ==========" % metric_name)
            for model_name, values in metric_dict.items():

                def get_metric_statistics(values, replication_times):
                    mean = np.mean(values, axis=0)
                    std = np.std(values, axis=0)
                    conf_interval = 1.96 * std / np.sqrt(replication_times)
                    return mean, conf_interval

                mean, conf_interval = get_metric_statistics(
                    np.array(values), replication_times
                )
                mean_dict[metric_name + "_" + model_name] = mean
                conf_dict[metric_name + "_" + model_name] = conf_interval

                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(
                        f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}"
                    )
                elif isinstance(mean, np.ndarray):
                    line = f"---> [{model_name}]"
                    for i in range(len(mean)):
                        line += "(top %d) Mean: %.4f CInt: %.4f;" % (
                            i + 1,
                            mean[i],
                            conf_interval[i],
                        )
                    print(line)

        print(f"mean_dict: {mean_dict}")
        print(f"conf_dict: {conf_dict}")
        print(tabulate(pd.DataFrame(mean_dict), headers='keys', tablefmt='grid'))
        return mean_dict, conf_dict


if __name__ == "__main__":
    split = "test"
    device = dist_util.dev()
    # opt.infer_step = 2

    args = evaluation_parser()
    fixseed(args.seed)

    eval_wrapper = eval_agent()
    traj2joints = AnyRep2SMPLjoints()

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")

    log_file = os.path.join(
        os.path.dirname(args.model_path), "eval_humanml_{}_{}".format(name, niter)
    )

    if args.guidance_param != 1.0:
        log_file += f"_gscale{args.guidance_param}"

    args.eval_mode = "debug"
    log_file += f"_{args.eval_mode}"
    log_json = log_file + ".json"
    log_file += ".log"

    print(f"Will save to log file [{log_file}]")
    print(f"Eval mode [{args.eval_mode}]")

    if args.eval_mode == "debug":
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = args.replication_times
        eval_joints_metrics = args.eval_joints_metrics
    elif args.eval_mode == "debug2":
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 1  # about 3 Hrs
    elif args.eval_mode == "wo_mm":
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 20  # about 12 Hrs
    elif args.eval_mode == "mm_short":
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 300
        replication_times = 5  # about 15 Hrs
    else:
        raise ValueError()

    dist_util.setup_dist(args.device)
    logger.configure()
    logger.log("creating data loader...")

    for k, v in opt.items():
        setattr(args, k, v)
    gt_loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=None,
        split=split,
        hml_mode="gt",
        args=args,
    )
    gen_loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=None,
        split=split,
        hml_mode="eval",
        args=args,
    )
    print(args.in_type)

    if args.eval_cmdm_base:
        args.train_cmdm_base = True
        from model.cmdm_style import MDM
    elif args.eval_cmdm_finetune:
        args.train_cmdm_base = False
        from model.cmdm_style import MDM
    else:
        from model.mdm_taming_style import MDM
    model = MDM(
        **get_model_args(args, gen_loader),
        inp_len=gen_loader.dataset.t2m_dataset.inp_len,
    )
    diffusion = create_gaussian_diffusion(args)
    # import ipdb;ipdb.set_trace()

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    eval_motion_loaders = {
        "vald": lambda: get_taming_loader(
            model,
            diffusion,
            args.batch_size,
            gen_loader,
            mm_num_samples,
            mm_num_repeats,
            gt_loader.dataset.opt.max_motion_length,
            num_samples_limit,
            args.guidance_param,
            args,
        )
    }
    mean_dict, conf_dict = evaluation(
        eval_wrapper,
        gt_loader,
        eval_motion_loaders,
        log_file,
        replication_times,
        diversity_times,
        mm_num_times,
        run_mm=run_mm,
    )

    # Function to convert float32 values in a dictionary to float
    def convert_float32_values(d):
        for key, value in d.items():
            if isinstance(value, dict):
                convert_float32_values(value)
            elif isinstance(value, np.float32):
                d[key] = float(value)

    # Convert float32 values in mean_dict and conf_dict
    convert_float32_values(mean_dict)
    convert_float32_values(conf_dict)

    with open(log_json, "w") as f:
        f.write(
            json.dumps(
                {
                    "mean_dict": mean_dict,
                    "conf_dict": conf_dict,
                }
            )
        )
