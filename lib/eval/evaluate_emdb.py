import os
import time
import os.path as osp
from glob import glob
from collections import defaultdict

import torch
import pickle
import numpy as np
from smplx import SMPL
from loguru import logger
from progress.bar import Bar

from configs import constants as _C
from configs.config import parse_args
from lib.data.dataloader import setup_eval_dataloader
from lib.models import build_network, build_body_model
from lib.eval.eval_utils import (
    compute_error_accel,
    batch_align_by_pelvis,
    batch_compute_similarity_transform_torch,
    compute_jpe,
    first_align_joints,
    global_align_joints,
)
from lib.utils import transforms
from lib.utils.utils import prepare_output_dir
from lib.utils.utils import prepare_batch

"""
This is a tentative script to evaluate WHAM on EMDB dataset.
Current implementation requires EMDB dataset downloaded at ./datasets/EMDB/
"""

m2mm = 1e3


@torch.no_grad()
def main(cfg, args):
    if args.post_fitting:
        import sys

        sys.path.append("/root/apdcephfs/private_wallyliang/PLANT/Thirdparty/WHAM")
        sys.path.insert(0, "/apdcephfs/private_wallyliang/PLANT")
        from show_libs.utils.decorator import tensor2numpy, to_detach
        from DiffTraj import DiffTraj

        traj_predictor = DiffTraj()

    logger.info(f"GPU name -> {torch.cuda.get_device_name()}")
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    # ========= Dataloaders ========= #
    eval_loader = setup_eval_dataloader(
        cfg, "emdb", args.eval_split, cfg.MODEL.BACKBONE
    )
    logger.info(f"Dataset loaded")

    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()

    # Build SMPL models with each gender
    smpl = {
        k: SMPL(_C.BMODEL.FLDR, gender=k).to(cfg.DEVICE)
        for k in ["male", "female", "neutral"]
    }

    # Load vertices -> joints regression matrix to evaluate
    J_regressor_eval = smpl["neutral"].J_regressor.clone().unsqueeze(0)
    pelvis_idxs = [1, 2]

    # WHAM uses Y-down coordinate system, while EMDB dataset uses Y-up one.
    yup2ydown = (
        transforms.axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]]))
        .float()
        .to(cfg.DEVICE)
    )

    # To torch tensor function
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)
    accumulator = defaultdict(list)
    bar = Bar("Inference", fill="#", max=len(eval_loader))

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):

            # <======= Prepare groundtruth data
            subj, seq = batch["vid"][0][:2], batch["vid"][0][3:]

            # import ipdb;ipdb.set_trace()
            annot_pth = glob(osp.join(_C.PATHS.EMDB_PTH, seq, "*_data.pkl"))
            # annot_pth = glob(osp.join(_C.PATHS.EMDB_PTH, subj, seq, '*_data.pkl'))

            if len(annot_pth) == 0:
                print(f"{subj} {seq} not exists")
                continue

            annot_pth = annot_pth[0]

            # if not os.path.exists(annot_pth):
            #     print(f'{annot_pth} not exists')
            #     continue

            annot = pickle.load(open(annot_pth, "rb"))

            masks = annot["good_frames_mask"]
            gender = annot["gender"]
            poses_body = annot["smpl"]["poses_body"]
            poses_root = annot["smpl"]["poses_root"]
            betas = np.repeat(
                annot["smpl"]["betas"].reshape((1, -1)),
                repeats=annot["n_frames"],
                axis=0,
            )
            trans = annot["smpl"]["trans"]
            extrinsics = annot["camera"]["extrinsics"]

            # # Map to camear coordinate
            poses_root_cam = transforms.matrix_to_axis_angle(
                tt(extrinsics[:, :3, :3])
                @ transforms.axis_angle_to_matrix(tt(poses_root))
            )

            # Groundtruth global motion
            target_glob = smpl[gender](
                body_pose=tt(poses_body),
                global_orient=tt(poses_root),
                betas=tt(betas),
                transl=tt(trans),
            )
            target_j3d_glob = torch.matmul(
                smpl[gender].J_regressor.unsqueeze(0), target_glob.vertices
            )[masks]

            # Groundtruth local motion
            target_cam = smpl[gender](
                body_pose=tt(poses_body), global_orient=poses_root_cam, betas=tt(betas)
            )
            target_verts_cam = target_cam.vertices[masks]
            target_j3d_cam = torch.matmul(
                smpl[gender].J_regressor.unsqueeze(0), target_verts_cam
            )
            # =======>

            x, inits, features, gt = prepare_batch(batch, cfg.DEVICE, True)

            # Align with groundtruth data to the first frame
            cam2yup = batch["R"][0][:1].to(cfg.DEVICE)
            cam2ydown = cam2yup @ yup2ydown
            cam2root = transforms.rotation_6d_to_matrix(inits[1][:, 0, 0])
            ydown2root = cam2ydown.mT @ cam2root
            ydown2root = transforms.matrix_to_rotation_6d(ydown2root)
            gt["init_root"][:, 0] = ydown2root

            # <======= Inference
            pred = network(x, inits, features, **gt)
            # =======>

            stability = np.zeros((1, 1))
            # <======= TODO: add fitting
            # torch.save(pred, 'pred.pt')
            if args.post_fitting:
                # result = run_post_fiiting_single(pred)
                # pred['poses_body'] = result['person0']['poses_body']
                # pred['trans_world'] = result['person0']['trans_world']
                # pred['poses_root_world'] = result['person0']['poses_root_world']
                # stability = result['person0']['stability_metric'].cpu().numpy()[None,...]

                def tt(x):
                    return x.float().cuda()

                def t2t(x):
                    return torch.from_numpy(x).float().cuda()

                print("[WARN] this is old codebase......")
                # import ipdb;ipdb.set_trace()
                pred_root_world = (
                    transforms.matrix_to_axis_angle(pred["poses_root_world"])
                    .cpu()
                    .numpy()
                    .reshape(-1, 3)
                )
                pred_body_pose = (
                    transforms.matrix_to_axis_angle(pred["poses_body"])
                    .cpu()
                    .numpy()
                    .reshape(-1, 69)
                )
                pred_root_world = t2t(pred_root_world)
                pred_body_pose = t2t(pred_body_pose)
                pred_trans_world = tt(pred["trans_world"][0])
                difftraj_root, difftraj_trans = traj_predictor.pose_to_traj(
                    pred_root_world, pred_body_pose, pred_trans_world
                )
                pred["poses_root_world"] = transforms.axis_angle_to_matrix(
                    difftraj_root
                )
                pred["trans_world"] = difftraj_trans
            # =======>

            # Convert WHAM global orient to Y-up coordinate
            poses_root = pred["poses_root_world"].squeeze(0)
            trans = pred["trans_world"].squeeze(0)
            poses_root = yup2ydown.mT @ poses_root
            trans = (yup2ydown.mT @ trans.unsqueeze(-1)).squeeze(-1)

            # <======= Build predicted motion
            # Predicted global motion
            pred_glob = smpl["neutral"](
                body_pose=pred["poses_body"],
                global_orient=poses_root.unsqueeze(1),
                betas=pred["betas"].squeeze(0),
                transl=trans,
                pose2rot=False,
            )
            pred_j3d_glob = torch.matmul(
                smpl["neutral"].J_regressor.unsqueeze(0), pred_glob.vertices
            )

            # Predicted local motion
            pred_cam = smpl["neutral"](
                body_pose=pred["poses_body"],
                global_orient=pred["poses_root_cam"],
                betas=pred["betas"].squeeze(0),
                pose2rot=False,
            )
            pred_verts_cam = pred_cam.vertices
            pred_j3d_cam = torch.matmul(
                smpl["neutral"].J_regressor.unsqueeze(0), pred_verts_cam
            )
            # =======>

            # <======= Evaluation on the local motion
            pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam = (
                batch_align_by_pelvis(
                    [pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam],
                    pelvis_idxs,
                )
            )
            S1_hat = batch_compute_similarity_transform_torch(
                pred_j3d_cam, target_j3d_cam
            )
            pa_mpjpe = (
                torch.sqrt(((S1_hat - target_j3d_cam) ** 2).sum(dim=-1))
                .mean(dim=-1)
                .cpu()
                .numpy()
                * m2mm
            )
            mpjpe = (
                torch.sqrt(((pred_j3d_cam - target_j3d_cam) ** 2).sum(dim=-1))
                .mean(dim=-1)
                .cpu()
                .numpy()
                * m2mm
            )
            pve = (
                torch.sqrt(((pred_verts_cam - target_verts_cam) ** 2).sum(dim=-1))
                .mean(dim=-1)
                .cpu()
                .numpy()
                * m2mm
            )
            accel = compute_error_accel(
                joints_pred=pred_j3d_cam.cpu(), joints_gt=target_j3d_cam.cpu()
            )[1:-1]
            accel = accel * (30**2)  # per frame^s to per s^2

            summary_string = f'{batch["vid"][0]} | PA-MPJPE: {pa_mpjpe.mean():.1f}   MPJPE: {mpjpe.mean():.1f}   PVE: {pve.mean():.1f}'
            bar.suffix = summary_string
            bar.next()
            # =======>

            # <======= Evaluation on the global motion
            chunk_length = 100
            w_mpjpe, wa_mpjpe = [], []
            for start in range(0, masks.sum() - chunk_length, chunk_length):
                end = start + chunk_length
                if start + 2 * chunk_length > masks.sum():
                    end = masks.sum() - 1

                target_j3d = target_j3d_glob[start:end].clone().cpu()
                pred_j3d = pred_j3d_glob[start:end].clone().cpu()

                w_j3d = first_align_joints(target_j3d, pred_j3d)
                wa_j3d = global_align_joints(target_j3d, pred_j3d)

                w_jpe = compute_jpe(target_j3d, w_j3d)
                wa_jpe = compute_jpe(target_j3d, wa_j3d)
                w_mpjpe.append(w_jpe)
                wa_mpjpe.append(wa_jpe)

            w_mpjpe = np.concatenate(w_mpjpe) * m2mm
            wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm
            # =======>

            # <======= Accumulate the results over entire sequences
            accumulator["pa_mpjpe"].append(pa_mpjpe)
            accumulator["mpjpe"].append(mpjpe)
            accumulator["pve"].append(pve)
            accumulator["accel"].append(accel)
            accumulator["wa_mpjpe"].append(wa_mpjpe)
            accumulator["w_mpjpe"].append(w_mpjpe)
            accumulator["stability"].append(stability)
            # =======>

    for k, v in accumulator.items():
        accumulator[k] = np.concatenate(v).mean()

    print("")
    log_str = f"Evaluation on EMDB {args.eval_split}, "
    log_str += " ".join([f"{k.upper()}: {v:.4f}," for k, v in accumulator.items()])
    logger.info(log_str)


if __name__ == "__main__":
    cfg, cfg_file, args = parse_args(test=True)
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg, args)
