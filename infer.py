import os
import sys

__cur__ = os.path.join(os.path.dirname(__file__))
sys.path.append(__cur__ + "/third-party")
sys.path.append(__cur__ + "/difftraj")


import time
import colorsys
import argparse
import cv2
import torch
import joblib
import imageio
import numpy as np
from smplx import SMPL
from loguru import logger
from progress.bar import Bar
import os.path as osp
from glob import glob
from collections import defaultdict

from configs.config import get_cfg_defaults
from lib.data._custom import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model, build_smplx_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify
from lib.models.smplifyx import TemporalSMPLifyX

from lib.models.layers import (
    MotionEncoder,
    MotionDecoder,
    TrajectoryDecoder,
    TrajectoryRefiner,
    Integrator,
    rollout_global_motion,
    compute_camera_pose,
    reset_root_velocity,
    compute_camera_motion,
)

from easydict import EasyDict
from lib.utils.bbox import get_move_area, scale_bbox
from lib.utils.decorator import tensor2numpy, to_detach
from lib.utils.video import pySceneDetect
from lib.vis.run_vis import run_vis_on_demo
from lib.loco.trajdiff import *
from copy import deepcopy

from pathlib import Path


print = logger.info

try:
    from lib.models.preproc.slam import SLAMModel

    _run_global = True
    # print(f'[IMPORTANT] DPVO enabled')
except:
    logger.info("DPVO is not properly installed. Only estimate in local coordinates !")
    _run_global = False


def insert_pred_root(pred, root_world):
    # root_world: T,3
    pose = pred["pose"]
    pose = pose.reshape(pose.shape[1], -1, 6)
    pose = s2a(pose)
    # pose = make_it_stand(pose)
    pose[:, 0, :] = root_world
    pose = a2s(pose)
    pose = pose.reshape(-1, 24 * 6).unsqueeze(0)
    pred["pose"] = pose
    return pred


def make_it_stand(pose):
    # pose: T,24,N
    pose[:, np.array([0, 1, 3, 4, 6, 7, 9, 10]) + 1] = 0
    pose[
        :,
        np.array(
            [
                2,
                5,
                8,
                11,
            ]
        )
        + 1,
    ] = 0
    pose[
        :,
        np.array(
            [
                14,
            ]
        )
        + 1,
    ] = 0
    return pose


def get_res_init(T):
    jaw = tonp(torch.tensor([1, 0, 0, 0, 1, 0.0]).float().repeat(T, 1).unsqueeze(0))
    leye = jaw.copy()
    reye = jaw.copy()
    exp = np.zeros((1, T, 50))
    return jaw, leye, reye, exp


def cvt_hamer_hands(hand_pose):
    # 左手怎么处理？
    # cvt_rot = torch.tensor([
    #     [-1.,  0.,  0.],
    #     [0.,  1.,  0.],
    #     [0,  0.,  -1.],
    # ]).float()
    Lh = hand_pose[:, 0]
    Rh = hand_pose[:, 1]
    # import ipdb;ipdb.set_trace()
    euler = m2e(Lh, "ZYX")
    euler[..., 0] *= -1
    euler[..., 1] *= -1
    euler[..., 2] *= -1
    Lh = e2m(euler, "ZYX")
    # Lh = torch.matmul(cvt_rot, Lh)
    # Lh = Lh.inverse()
    hand_pose = torch.stack([Lh, Rh], dim=1)
    return hand_pose


def run(
    cfg,
    video,
    output_pth,
    network,
    detector=None,
    calib=None,
    run_global=True,
    save_pkl=False,
    visualize=False,
    args=None,
):

    smplx = network.smplx

    if not args.debug:
        logger.remove()
        # 添加一个仅输出 info 级别及以上日志的控制台处理器
        logger.add(sys.stderr, level="INFO")

    traj_predictor = DiffTraj()

    # TODO: mod video framerate here
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f"Faild to load video file {video}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.runing_fps == -1:
        origin_fps = fps
        stride = 1
    else:
        assert abs(fps - 30) < 0.1, f"fps must be 30..."
        origin_fps = 30
        stride = 1

    print(f"{origin_fps=}, {stride=}, {fps=}")

    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )

    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global

    frame_idx = 0

    # TODO: here
    if args.anno_pkl_path is not None:
        import config

        anno_data = joblib.load(args.anno_pkl_path)
        persons_ids = list(anno_data["tracklet_list_all"].keys())

    st = time.time()

    # <---------
    # PySceneDetect
    if args.inp_body_pose is None:
        scene_list_path = osp.join(output_pth, "scene_list.pth")
        # import ipdb;ipdb.set_trace()
        if not osp.exists(scene_list_path):
            scene_list = pySceneDetect(video)
            torch.save(scene_list, scene_list_path)
        else:
            scene_list = torch.load(scene_list_path)

        scene_change_pos = []
        for i in scene_list:
            scene_change_pos.append(i[0].frame_num)
            scene_change_pos.append(i[1].frame_num)
        scene_change_pos = list(set(scene_change_pos))
        scene_change_pos = [int(i / stride) for i in scene_change_pos]
    else:
        scene_change_pos = []
    # <---------

    # Preprocess
    if not (
        osp.exists(osp.join(output_pth, "tracking_results.pth"))
        and osp.exists(osp.join(output_pth, "slam_results.pth"))
    ):
        if detector is None:
            detector = DetectionModel(cfg.DEVICE.lower(), args)
        # extractor = FeatureExtractor(cfg.DEVICE.lower())
        extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)

        # TODO: mod video framerate here
        if run_global:
            slam = SLAMModel(video, output_pth, width, height, calib, stride=stride)
        else:
            slam = None

        bar = Bar("Preprocess: 2D detection and SLAM", fill="#", max=length // stride)

        current_frame = 0

        print(f"{args.only_one_person=}")

        while cap.isOpened():
            # flag, img = cap.read()
            # if not flag: break
            # Capture frame-by-frame
            for _ in range(stride):
                flag, img = cap.read()
                # if frame is read correctly flag is True
                if not flag:
                    break

            if not flag:
                break

            current_frame += 1

            # 2D detection and tracking
            # import ipdb;ipdb.set_trace()
            if args.anno_pkl_path is not None:
                # 得到一帧内所有人的bbox
                pre_def_bboxes = []
                for person_id in persons_ids:
                    person_data = anno_data["tracklet_list_all"][person_id]
                    cur_idx = frame_idx - person_data["st_frame"]
                    if (
                        frame_idx < person_data["st_frame"]
                        or frame_idx > person_data["et_frame"]
                    ):
                        continue
                    if cur_idx >= len(person_data["track_bbox"]):
                        continue
                    bboxes = person_data["track_bbox"][cur_idx]
                    bboxes = [
                        bboxes[0],
                        bboxes[1],
                        bboxes[0] + bboxes[2],
                        bboxes[1] + bboxes[3],
                    ]
                    contact_detect_result = person_data["contact_detect_result"][
                        cur_idx
                    ]
                    pre_def_bboxes.append(
                        {"bbox": bboxes, "contact": contact_detect_result}
                    )

                detector.track(img, fps, length, pre_def_bboxes)
            else:
                detector.track(
                    img,
                    fps,
                    length,
                    scene_change_pos=scene_change_pos,
                    current_frame=current_frame,
                    only_one_person=args.only_one_person,
                    # only_one_person=True if args.inp_body_pose is not None else False,
                )

            # SLAM
            if slam is not None:
                slam.track()

            # for _ in range(stride):
            bar.next()

            frame_idx += 1

        tracking_results = detector.process(fps)

        if slam is not None:
            slam_results = slam.process()
        else:
            slam_results = np.zeros((length, 7))
            slam_results[:, 3] = 1.0  # Unit quaternion

        # Extract image features
        # TODO: Merge this into the previous while loop with an online bbox smoothing.
        # TODO: mod video framerate here
        tracking_results = extractor.run(video, tracking_results, stride=stride)
        logger.info("Complete Data preprocessing!")

        # Save the processed data
        joblib.dump(tracking_results, osp.join(output_pth, "tracking_results.pth"))
        joblib.dump(slam_results, osp.join(output_pth, "slam_results.pth"))
        logger.info(f"Save processed data at {output_pth}")

    # If the processed data already exists, load the processed data
    else:
        tracking_results = joblib.load(osp.join(output_pth, "tracking_results.pth"))
        slam_results = joblib.load(osp.join(output_pth, "slam_results.pth"))
        logger.info(f"Already processed data exists at {output_pth} ! Load the data .")

    print(f"preprocess used time: {time.time()-st} seconds")

    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)

    # run WHAM

    # prepare
    device = cfg.DEVICE
    dtype = torch.float32
    I = torch.eye(3)[None].to(device=device).to(dtype=dtype)
    results = EasyDict()

    st = time.time()
    # 《=========
    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                (
                    _id,
                    x,
                    inits,
                    features,
                    mask,
                    init_root,
                    cam_angvel,
                    frame_id,
                    kwargs,
                ) = flipped_batch
                flipped_pred = network(
                    x,
                    inits,
                    features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    **kwargs,
                )

                # import ipdb;ipdb.set_trace()
                if len(frame_id) < 128:
                    # if len(frame_id) < clip_frames:
                    print(f"too short ID{_id}, len {len(frame_id)}")
                    continue

                # Forward pass with normal input
                batch = dataset.load_data(subj)
                (
                    _id,
                    x,
                    inits,
                    features,
                    mask,
                    init_root,
                    cam_angvel,
                    frame_id,
                    kwargs,
                ) = batch
                pred = network(
                    x,
                    inits,
                    features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    **kwargs,
                )

                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred["pose"].squeeze(
                    0
                ), flipped_pred["betas"].squeeze(0)
                pose, shape = pred["pose"].squeeze(0), pred["betas"].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(
                    -1, 24, 6
                )
                avg_pose, avg_shape = avg_preds(
                    pose, shape, flipped_pose, flipped_shape
                )
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (
                    flipped_pred["contact"][..., [2, 3, 0, 1]] + pred["contact"]
                ) / 2

                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

            else:
                # data
                batch = dataset.load_data(subj)
                (
                    _id,
                    x,
                    inits,
                    features,
                    mask,
                    init_root,
                    cam_angvel,
                    frame_id,
                    kwargs,
                ) = batch

                # inference
                pred = network(
                    x,
                    inits,
                    features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    **kwargs,
                )

        def update_wham_vars():
            nonlocal pred
            nonlocal network
            with torch.no_grad():
                network.pred_pose = pred["pose"]
                network.pred_shape = pred["betas"]
                network.pred_cam = pred["cam"]
                pred = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(pred, cam_angvel, return_y_up=True)

        result_id_path = str(Path(output_pth, f"ID{subj}_tmp.jpkl"))

        if args.save_opt_tmp:
            if Path(result_id_path).exists():
                results[str(_id)] = joblib.load(result_id_path)
                print(f"{result_id_path=} exists, continue")
                continue

        # <===========================
        """
        步骤:
        SMPLX参数初始化:
            身体:WHAM，body pose
            手部:双手HAMER,L R hands
            脸部:DECA, exp jaw
        长时间序列分段处理（重叠优化，重叠的部分优化consistent的平滑）
            src参数:
                prior proxy:
                    body: NeMF
                    hands: HMP
                    face: None
            dst参数:
                data:
                conf:
                mask:
            loss weight
            loss type: 
                reg(one params, two params)
                jitter
                prior
        """

        T = batch_size = pred["cam"].shape[1]
        dst_data = dataset.tracking_results[_id]
        jaw, leye, reye, exp = get_res_init(T)

        if args.vis_kpts:
            save_path = osp.join(output_pth, f"render_ID{_id}_kpts.mp4")
            if not os.path.exists(save_path):
                from lib.vis.run_vis import save_kpts2d

                # save_kpts2d(toth(dst_data['keypoints']), save_path, fps, int(width), int(height))
                tmp = dst_data["keypoints"]
                tmp = np.concatenate(
                    [
                        tmp,
                        np.zeros((tmp.shape[0], 6, 3)),
                        dst_data["face_keyp"],
                        dst_data["left_hand_keyp"],
                        dst_data["right_hand_keyp"],
                    ],
                    axis=1,
                )
                save_kpts2d(
                    toth(tmp),
                    save_path,
                    fps,
                    int(width),
                    int(height),
                    dataset="TopDownCocoWholeBodyDataset",
                    bg_gray=0,
                )
                print(f"2d kpts saved to {save_path}")

        if (dst_data["keypoints"][:, [13, 14, 15, 16], -1] < 0.4).all():
            args.replace_lower = True
            # args.run_smplify = True
            # args.use_hmr2 = True
            print(f"[INFO] set args.replace_lower = True")
        else:
            args.replace_lower = False
            # args.run_smplify = False
            # args.use_hmr2 = False

        if args.inp_body_pose is not None:
            inp_data = dict(np.load(args.inp_body_pose))
            poses = a2s(toth(inp_data["poses"]))
            body_pose = poses[:, :24, :]
            hand_pose = poses[:, 25:, :]
            smplerx_hand_pose = torch.stack(
                [hand_pose[:, :15, :], hand_pose[:, 15:, :]], dim=1
            ).unsqueeze(
                0
            )  # (1, T, 2, 15, 6)

            pred["pose"] = body_pose.reshape(-1, 24 * 6).unsqueeze(0)
            pred["cam"] = toth(inp_data["trans"]).squeeze().unsqueeze(0)

        if args.use_hmr2:
            pred["pose"] = (
                a2s(
                    toth(
                        np.concatenate(
                            [dst_data["hmr_orient"], dst_data["hmr_body_pose"]], axis=1
                        )
                    )
                )
                .reshape(-1, 24 * 6)
                .unsqueeze(0)
            )
            pred["cam"] = toth(dst_data["hmr_pred_cam"]).squeeze().unsqueeze(0)
            pred["betas"] = toth(dst_data["hmr_betas"]).squeeze().unsqueeze(0)

        flame2_smplx_convt = toth(
            np.load("model_files/FLAME2020/flame2020to2019_exp_trafo.npy")
        )

        def cvt_exp(e_deca):
            e_deca = torch.nn.functional.pad(e_deca, (0, 50), "constant", 0)
            e_smplx = torch.matmul(e_deca, flame2_smplx_convt)[:, :50]
            exp = tonp(e_smplx.unsqueeze(0))
            return exp

        if args.use_deca:
            exp = cvt_exp(toth(dst_data["deca_exp"]))
            jaw = tonp(a2s(toth(dst_data["deca_jaw"][:, 3:])).unsqueeze(0))

        mica_vert = None
        if args.use_smirk:
            # exp = cvt_exp(toth(dst_data['smirk_exp']))
            exp = (toth(dst_data["smirk_exp"])).unsqueeze(0)
            jaw = tonp(a2s(toth(dst_data["smirk_jaw"])).unsqueeze(0))

            # import sys;sys.path.append('third-party/smirk')
            # from src.FLAME.FLAME import FLAME
            # with path_enter('third-party/smirk'):
            #     flame = FLAME()

            # f_shape = (toth(dst_data['smirk_shape'])) #[249, 300]
            # ret = flame({
            #     'shape_params': f_shape.mean(0).unsqueeze(0),
            #     'expression_params': torch.zeros((1,50)),
            #     'pose_params': torch.zeros((1,3)),
            #     'jaw_params': torch.zeros((1,3)),
            #     'eye_pose_params': torch.zeros((1,6)),
            #     'neck_pose_params': torch.zeros((1,3)),
            #     'eyelid_params': torch.zeros((1,6)),
            # })
            # mica_vert=ret['vertices']#([1, 5023, 3])

        # <=========
        if args.use_hamer:
            hand_pose = toth(dst_data["hand_pose"])
            hand_pose = cvt_hamer_hands(hand_pose)
            hand_pose = tonp(m2s(hand_pose).unsqueeze(0))  # (1, T, 2, 15, 6)
        elif args.use_acr:
            hand_pose = toth(dst_data["hand_pose"])
            hand_pose = tonp(m2s(hand_pose).unsqueeze(0))  # (1, T, 2, 15, 6)
        elif args.inp_body_pose is not None:
            hand_pose = tonp(smplerx_hand_pose)
        else:
            hand_pose = np.zeros((1, T, 2, 15, 6))
        # <=========

        init_param_np = {
            "pose": tonp(pred["pose"]),
            "betas": tonp(pred["betas"]),
            "cam": tonp(pred["cam"]),
            "hand_pose": hand_pose,
            "jaw": jaw,
            "leye": leye,
            "reye": reye,
            "exp": exp,
        }
        init_param_th = {k: toth(v).to(cfg.DEVICE) for k, v in init_param_np.items()}

        pred_body_pose_no_fit = matrix_to_axis_angle(pred["poses_body"]).reshape(-1, 69)
        pred_root_world_no_fit = matrix_to_axis_angle(pred["poses_root_world"]).reshape(
            -1, 3
        )
        pred_trans_world_no_fit = pred["trans_world"].squeeze(0)

        # 《=========
        #
        if args.run_smplify:
            print("run_smplify...")
            if args.use_smplx:
                smplify = TemporalSMPLifyX(
                    smplx, img_w=width, img_h=height, device=cfg.DEVICE
                )
                init_param_th = smplify.fit(
                    init_param_th=init_param_th,
                    dst_data=dataset.tracking_results[_id],
                    args=args,
                    **kwargs,
                )
                pred = deepcopy(init_param_th)
            else:
                smplify = TemporalSMPLify(
                    smpl, img_w=width, img_h=height, device=cfg.DEVICE
                )
                input_keypoints = dataset.tracking_results[_id]["keypoints"]
                pred = smplify.fit(pred, input_keypoints, **kwargs)

            update_wham_vars()
        # 《=========

        # ========= Store results ========= #
        pred_body_pose = matrix_to_axis_angle(pred["poses_body"]).reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred["poses_root_cam"]).reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred["poses_root_world"]).reshape(-1, 3)
        pred_pose = torch.cat((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = torch.cat((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = pred["trans_cam"] - network.output.offset
        pred_trans_world = pred["trans_world"].squeeze(0)
        pred_betas = pred["betas"].squeeze(0)

        # save vars
        _id = str(_id)
        results[_id] = EasyDict()
        results[_id]["pose_world"] = pred_pose_world
        results[_id]["trans_world"] = pred_trans_world
        results[_id]["frame_ids"] = frame_id
        results[_id]["betas"] = pred_betas
        results[_id]["pred"] = pred
        results[_id]["pose"] = pred_pose
        results[_id]["trans"] = pred_trans

        results[_id]["bbox"] = dst_data["bbox"]
        results[_id]["bbox_xyxy"] = dst_data["bbox_xyxy"]
        results[_id]["keypoints"] = dst_data["keypoints"]  # (299, 17, 3)

        # save torch tensor
        results[_id]["pred_body_pose_no_fit"] = pred_body_pose_no_fit
        results[_id]["pred_root_world_no_fit"] = pred_root_world_no_fit
        results[_id]["pred_trans_world_no_fit"] = pred_trans_world_no_fit
        results[_id]["pred_body_pose"] = pred_body_pose
        results[_id]["pred_root_world"] = pred_root_world
        results[_id]["pred_trans_world"] = pred_trans_world

        # numpy array for visualization
        params = init_param_th
        opt_output = smplx(
            betas=params["betas"],  # T,10
            Pose=params["pose"],  # T,24,6
            Lh=params["hand_pose"][:, :, 0, :, :],  # T,15,6
            Rh=params["hand_pose"][:, :, 1, :, :],
            Exp=params["exp"],  # T,50
            Jaw=params["jaw"],  # T,3
            Leye=params["leye"],  # T,3
            Reye=params["reye"],
            cam=params["cam"],
            cam_intrinsics=kwargs["cam_intrinsics"],
            bbox=kwargs["bbox"],
            res=kwargs["res"],
            return_full_pose=True,
        )
        pred["verts_cam"] = opt_output.vertices
        results[_id]["verts"] = (
            (pred["verts_cam"] + pred["trans_cam"].unsqueeze(1)).cpu().numpy()
        )
        tmp = opt_output.full_pose.reshape(-1, 55, 3)[:, 25 : 25 + 30, :]
        tmp = torch.stack([tmp[:, :15, :], tmp[:, 15:, :]], dim=1)
        results[_id]["hand_pose"] = tonp(a2m(tmp))  # T,2,15,3,3
        results[_id]["init_param_th"] = init_param_th

        # predict trajectory
        difftraj_root, difftraj_trans = traj_predictor.pose_to_traj(
            pred_root_world, pred_body_pose, pred_trans_world
        )
        results[_id]["difftraj_root"] = difftraj_root
        results[_id]["difftraj_trans"] = difftraj_trans

        if args.run_smplify and args.save_wham_npz:
            difftraj_root_no_fit, difftraj_trans_no_fit = traj_predictor.pose_to_traj(
                pred_root_world_no_fit, pred_body_pose_no_fit, pred_trans_world_no_fit
            )
            results[_id]["difftraj_root_no_fit"] = difftraj_root_no_fit
            results[_id]["difftraj_trans_no_fit"] = difftraj_trans_no_fit

        # parse bbox
        bbox_xyxy = results[_id]["bbox_xyxy"]
        xyxy = get_move_area(bbox_xyxy, fw=width, fh=height)
        xyxy = scale_bbox(xyxy, h=height, w=width, scale=1.3)
        results[_id]["merge_bbox_xyxy"] = xyxy

        results[_id]["height"] = height
        results[_id]["width"] = width
        results[_id]["stride"] = stride
        results[_id]["fps"] = fps
        results[_id]["origin_fps"] = origin_fps

        if (
            "face_keyp" in dst_data
            and "left_hand_keyp" in dst_data
            and "right_hand_keyp" in dst_data
        ):
            tmp = dst_data["keypoints"]
            whole_body_kpts = np.concatenate(
                [
                    tmp,
                    np.zeros((tmp.shape[0], 6, 3)),
                    dst_data["face_keyp"],
                    dst_data["left_hand_keyp"],
                    dst_data["right_hand_keyp"],
                ],
                axis=1,
            )
            results[_id]["whole_body_kpts"] = whole_body_kpts

        if args.save_opt_tmp:
            joblib.dump(
                deepcopy(results[_id]),
                result_id_path,
            )
            print(f"tmp to {result_id_path=}")

    if save_pkl:
        # joblib.dump(results, osp.join(output_pth, "wham_output.pkl"))
        # torch tensor to blender npz
        if args.run_smplify and args.save_wham_npz:
            save_to_blender_smplx_addon_npz(
                results,
                "whamPose+diffTraj",
                trans_name="difftraj_trans_no_fit",
                root_name="difftraj_root_no_fit",
                pose_var_name="pred_body_pose_no_fit",
                output_pth=output_pth,
                fps=fps,
                save_prefix=args.save_prefix,
            )

        save_to_blender_smplx_addon_npz(
            results,
            "difTraj_raw",
            trans_name="difftraj_trans",
            root_name="difftraj_root",
            pose_var_name="pred_body_pose",
            output_pth=output_pth,
            fps=fps,
            save_prefix=args.save_prefix,
            save_ex_keys=[
                "height",
                "width",
                "whole_body_kpts",
                "merge_bbox_xyxy",
                "bbox_xyxy",
                "stride",
                "fps",
                "origin_fps",
            ],
            # ex_params_to_save={
            #     'H': height,
            #     'W': width,
            #     'whole_body_kpts': whole_body_kpts,
            #     'bbox_xyxy': dst_data['bbox_xyxy'],
            #     'merge_bbox_xyxy': results[_id]['merge_bbox_xyxy']
            # },
            # tracking_results=dataset.tracking_results,
        )

    print(f"wham used time: {time.time()-st} seconds")

    # Visualize
    if visualize:
        st = time.time()
        results = tensor2numpy(results)
        run_vis_on_demo(
            cfg,
            video,
            results,
            output_pth,
            smplx,
            vis_global=run_global,
            stride=stride,
            faces=smplx.faces,
            save_prefix=args.save_prefix,
            save_ex=args.save_ex,
            args=args,
        )
        print(f"visualization used time: {time.time()-st} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        default="examples/demo_video.mp4",
        help="input video path or youtube link",
    )
    parser.add_argument(
        "--output_pth",
        type=str,
        default="/apdcephfs/private_wallyliang/wham_private_output",
        help="output folder to write results",
    )
    parser.add_argument(
        "--calib", type=str, default=None, help="Camera calibration file path"
    )
    parser.add_argument(
        "--estimate_local_only",
        action="store_true",
        help="Only estimate motion in camera coordinate if True",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the output mesh if True"
    )
    parser.add_argument(
        "--save_pkl", action="store_true", help="Save output as pkl file"
    )

    parser.add_argument(
        "--not_full_body",
        action="store_true",
        help="do not force filter out partial body frame.",
    )
    parser.add_argument("--anno_pkl_path", type=str, default=None, help="deprecated.")
    parser.add_argument(
        "--use_smplx",
        action="store_true",
        help="Use SMPLX model instead of original SMPL.",
    )
    parser.add_argument(
        "--use_hamer", action="store_true", help="HAMER for right hands."
    )
    parser.add_argument(
        "--use_deca", action="store_true", help="DECA for exp prediction."
    )
    parser.add_argument(
        "--use_smirk", action="store_true", help="SMRIK for exp prediction."
    )
    parser.add_argument(
        "--use_acr", action="store_true", help="ACR for interhands prediction."
    )
    parser.add_argument("--debug", action="store_true", help="debug.")

    parser.add_argument("--save_split_vid", action="store_true", help="debug.")
    parser.add_argument("--save_normal_map", action="store_true", help="debug.")
    parser.add_argument("--save_depth_map", action="store_true", help="debug.")

    parser.add_argument("--save_prefix", type=str, default="", help="for debug.")
    parser.add_argument(
        "--vitpose_model",
        type=str,
        default="ViTPose+-G",
        help="ViTPose+-G or ViTPose+-B",
    )

    parser.add_argument("--vis_kpts", action="store_true", help="debug.")
    parser.add_argument("--save_ex", action="store_true", help="debug.")
    parser.add_argument("--use_smplerx", action="store_true", help="debug.")
    parser.add_argument("--only_one_person", action="store_true", help="debug.")
    parser.add_argument("--use_opt_pose_reg", action="store_true", help="debug.")
    parser.add_argument("--skip_save_vis_if_exists", action="store_true", help="debug.")
    parser.add_argument("--save_opt_tmp", action="store_true", help="debug.")
    parser.add_argument("--opt_lr", type=float, default=1.0)
    parser.add_argument("--runing_fps", type=int, default=-1)

    # <==== AUTO
    parser.add_argument(
        "--save_wham_npz",
        action="store_true",
        help="save wham pose if run optimization.",
    )
    parser.add_argument(
        "--run_smplify",
        action="store_true",
        help="Run Temporal SMPLify for post processing, not used, we switch it automatically",
    )
    parser.add_argument(
        "--replace_lower",
        action="store_true",
        help="for speech scnerio, replace lower body with predifined motion. we switch it automatically",
    )
    parser.add_argument(
        "--use_hmr2",
        action="store_true",
        help="HMR2 for half body pose prdiction. we switch it automatically",
    )
    # <==== AUTO

    parser.add_argument(
        "--inp_body_pose",
        type=str,
        default=None,
        help="if set, we will use it as body pose initialization, only support single person video.",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    st = time.time()

    args = parser.parse_args()
    # import ipdb;ipdb.set_trace()

    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/yamls/demo.yaml")
    cfg.merge_from_list(args.opts)

    logger.info(f"GPU name -> {torch.cuda.get_device_name()}")
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()

    if args.use_smplx:
        smplx = build_smplx_body_model(cfg.DEVICE, smpl_batch_size, args)
        setattr(network, "smplx", smplx)

    # Output folder
    sequence = ".".join(args.video.split("/")[-1].split(".")[:-1])
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)

    logger.add(
        str(Path(output_pth) / "main.log"),
        rotation="10 MB",
        level="DEBUG",
        format="{time} {message}",
    )

    print(f"{cfg=}")
    print(f"{args=}")

    with torch.no_grad():
        run(
            cfg,
            args.video,
            output_pth,
            network,
            args.calib,
            run_global=not args.estimate_local_only,
            save_pkl=args.save_pkl,
            visualize=args.visualize,
            args=args,
        )
    et = time.time()
    print(f"used time: {et-st} seconds")
    logger.info("Done !")
