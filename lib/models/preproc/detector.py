from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal
from progress.bar import Bar

from ultralytics import YOLO
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    get_track_id,
    vis_pose_result,
)

ROOT_DIR = osp.abspath(f"{__file__}/../../../../")
VIT_DIR = osp.join(ROOT_DIR, "third-party/ViTPose")

VIS_THRESH = 0.3
BBOX_CONF = 0.5
TRACKING_THR = 0.2
MINIMUM_FRMAES = 30
from copy import deepcopy
from loguru import logger
import cv2

from skimage.transform import estimate_transform, warp, resize, rescale
from smirk.utils.mediapipe_utils import Mediapipe_detector, crop_face
from lib.loco.trajdiff import *
import ipdb

SMPLERX_PATH = "/apdcephfs/private_wallyliang/MotionCaption"
HAMER_PATH = "third-party/hamer"
HMR2_PATH = "third-party/4D-Humans"
ACR_PATH = "third-party/ACR"
SMIRK_PATH = "third-party/smirk"
DECA_PATH = None

print = logger.info


class SMPL_var_init_interface:
    def __init__(self):
        pass

    def get_var(self, in_var):
        pass


class Face_var_init_deca:
    def __init__(self):
        # INIT DECA
        # import ipdb;ipdb.set_trace()
        import sys

        sys.path.append(DECA_PATH)
        from decalib.deca import DECA
        from decalib.utils import util
        from decalib.utils.config import get_cfg_defaults

        deca_cfg = get_cfg_defaults()
        deca_cfg.model.use_tex = False
        deca_cfg.rasterizer_type = "pytorch3d"
        deca = DECA(config=deca_cfg, device=device)
        self.deca = deca

        self.vars_to_save = [
            "deca_exp",
            "deca_jaw",
        ]

        logger.debug("DECA")

    def get_result(self, xyxy, img, pose_result):
        # import ipdb;ipdb.set_trace()
        image = img.copy()
        image = image / 255.0

        left, top, right, bottom = list(map(int, xyxy[:4]))
        # xmin,ymin,xmax,ymax

        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        # h, w, _ = image.shape

        scale = 1.25
        crop_size = 224
        old_size = (right - left + bottom - top) / 2 * 1.1
        size = int(old_size * scale)

        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        new_bbox = [
            center[0] - size / 2,
            center[1] - size / 2,
            center[0] + size / 2,
            center[1] + size / 2,
        ]

        src_pts = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] - size / 2, center[1] + size / 2],
                [center[0] + size / 2, center[1] - size / 2],
            ]
        )
        DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
        tform = estimate_transform("similarity", src_pts, DST_PTS)
        dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))
        dst_image = dst_image.transpose(2, 0, 1)
        images = torch.tensor(dst_image).float().to(device)[None, ...]

        with torch.no_grad():
            codedict = self.deca.encode(images)
            opdict, visdict = self.deca.decode(codedict)
        pose_result["deca_exp"] = tonp(codedict["exp"].squeeze())
        pose_result["deca_jaw"] = tonp(codedict["pose"].squeeze())
        return pose_result


class Face_var_init_smirk:
    def __init__(self):
        # SMIRK
        import sys

        sys.path.append(SMIRK_PATH)
        from src.smirk_encoder import SmirkEncoder
        from src.FLAME.FLAME import FLAME
        from src.renderer.renderer import Renderer

        with path_enter(SMIRK_PATH):
            smirk_encoder = SmirkEncoder().to(device)
            checkpoint = torch.load("../../model_files/smirk/SMIRK_em1.pt")
            checkpoint_encoder = {
                k.replace("smirk_encoder.", ""): v
                for k, v in checkpoint.items()
                if "smirk_encoder" in k
            }  # checkpoint includes both smirk_encoder and smirk_generator

            smirk_encoder.load_state_dict(checkpoint_encoder)
            smirk_encoder.eval()

            flame = FLAME().to(device)
            renderer = Renderer().to(device)

            self.flame = flame
            self.renderer = renderer
            self.smirk_encoder = smirk_encoder
            # self.mpdet = Mediapipe_detector()

        self.vars_to_save = ["smirk_exp", "smirk_jaw", "smirk_shape"]
        logger.debug("SMIRK")

    def get_result(self, xyxy, img, pose_result):
        left, top, right, bottom = list(map(int, xyxy[:4]))
        image = img[top:bottom, left:right, :]

        if 1:
            left, top, right, bottom = list(xyxy[:4])
            tmp = pose_result["face_keyp"]
            tmp[:, 0] -= left
            tmp[:, 1] -= top
            kpt_mediapipe = tmp
        else:
            kpt_mediapipe = self.mpdet.infer(image)  # (478, 3)

        kpt_mediapipe = kpt_mediapipe[..., :2]

        image_size = 224
        tform = crop_face(image, kpt_mediapipe, scale=1.4, image_size=image_size)

        cropped_image = warp(
            image, tform.inverse, output_shape=(224, 224), preserve_range=True
        ).astype(np.uint8)

        cropped_kpt_mediapipe = np.dot(
            tform.params,
            np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T,
        ).T
        cropped_kpt_mediapipe = cropped_kpt_mediapipe[:, :2]
        # cv2.imwrite('test.png',cropped_image)

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224, 224))
        cropped_image = (
            torch.tensor(cropped_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        cropped_image = cropped_image.to(device)

        outputs = self.smirk_encoder(cropped_image)

        pose_result["smirk_exp"] = tonp(outputs["expression_params"].squeeze())
        pose_result["smirk_jaw"] = tonp(outputs["jaw_params"].squeeze())
        pose_result["smirk_shape"] = tonp(outputs["shape_params"].squeeze())

        return pose_result

    def vis(self, outputs):
        # import ipdb;ipdb.set_trace()
        flame_output = self.flame.forward(outputs)
        renderer_output = self.renderer.forward(flame_output['vertices'], outputs['cam'],
                                            landmarks_fan=flame_output['landmarks_fan'],
                                            landmarks_mp=flame_output['landmarks_mp'])

        rendered_img = renderer_output['rendered_img']
        grid = torch.cat([cropped_image, rendered_img], dim=3)
        cv2.imwrite('test.png',(grid.detach()[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))


class Body_var_init_hmr2:
    def __init__(self):
        # INIT 4D human
        import sys

        sys.path.append(HMR2_PATH)
        from hmr2.configs import CACHE_DIR_4DHUMANS
        from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT

        download_models(CACHE_DIR_4DHUMANS)
        # import ipdb;ipdb.set_trace()
        hmr2_model, hmr2_model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        hmr2_model = hmr2_model.to(device)
        hmr2_model.eval()
        self.hmr2_model = hmr2_model
        self.hmr2_model_cfg = hmr2_model_cfg

        self.vars_to_save = [
            "hmr_body_pose",
            "hmr_orient",
            "hmr_betas",
            "hmr_pred_cam",
            "hmr_cam_full",
        ]
        logger.debug("HMR2")

    def get_result(self, xyxy, img, pose_result):
        # import ipdb;ipdb.set_trace()
        from hmr2.utils import recursive_to
        from hmr2.datasets.vitdet_dataset import (
            ViTDetDataset,
            DEFAULT_MEAN,
            DEFAULT_STD,
        )
        from hmr2.utils.renderer import Renderer, cam_crop_to_full

        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(self.hmr2_model_cfg, img, xyxy[:4][None])
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0
        )
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = self.hmr2_model(batch)
                pred_cam = out["pred_cam"]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = (
                    self.hmr2_model_cfg.EXTRA.FOCAL_LENGTH
                    / self.hmr2_model_cfg.MODEL.IMAGE_SIZE
                    * img_size.max()
                )
                pred_cam_t_full = (
                    cam_crop_to_full(
                        pred_cam, box_center, box_size, img_size, scaled_focal_length
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                pose_result["hmr_orient"] = tonp(
                    m2a(out["pred_smpl_params"]["global_orient"][0])
                )  # 1,3
                pose_result["hmr_body_pose"] = tonp(
                    m2a(out["pred_smpl_params"]["body_pose"][0])
                )  # 23,3
                pose_result["hmr_betas"] = tonp(
                    out["pred_smpl_params"]["betas"]
                )  # 1,10
                pose_result["hmr_pred_cam"] = tonp(pred_cam)
                pose_result["hmr_cam_full"] = pred_cam_t_full

        return pose_result


class Body_var_init_smplerx:
    def __init__(self):
        import sys

        sys.path.append(SMPLERX_PATH)
        from main.mocap import init_model, mocap
        import ipdb

        ipdb.set_trace()
        with path_enter(SMPLERX_PATH):
            self.model = init_model()
        logger.debug("SMPLER-X")
        self.vars_to_save = [
            "smplerx_hand_pose",
            "smplerx_body_pose",
            "smplerx_orient",
            "smplerx_betas",
        ]

    def get_result(self, xyxy, img, pose_result):
        import ipdb

        ipdb.set_trace()
        demoer = self.model[0]
        from utils.preprocessing import load_img, process_bbox, generate_patch_image

        original_img_height, original_img_width = img.shape[:2]

        mmdet_box = xyxy
        mmdet_box_xywh = np.zeros((4))
        mmdet_box_xywh[0] = xyxy[0]
        mmdet_box_xywh[1] = xyxy[1]
        mmdet_box_xywh[2] = abs(xyxy[2] - xyxy[0])
        mmdet_box_xywh[3] = abs(xyxy[3] - xyxy[1])

        bbox = process_bbox(mmdet_box_xywh, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(
            original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape
        )
        img = transform(img.astype(np.float32)) / 255
        img = img.cuda()[None, :, :, :]
        inputs = {"img": img}
        targets = {}
        meta_info = {}

        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, "test")
        focal = [
            cfg.focal[0] / cfg.input_body_shape[1] * bbox[2],
            cfg.focal[1] / cfg.input_body_shape[0] * bbox[3],
        ]
        princpt = [
            cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0],
            cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1],
        ]
        global_orient = out["smplx_root_pose"].reshape(-1, 3).cpu().numpy()
        global_orient = global_orient * math.pi / 180.0

        # extract single person param
        smplx_pred = {}
        smplx_pred["global_orient"] = global_orient
        smplx_pred["body_pose"] = out["smplx_body_pose"].reshape(-1, 3).cpu().numpy()
        smplx_pred["left_hand_pose"] = (
            out["smplx_lhand_pose"].reshape(-1, 3).cpu().numpy()
        )
        smplx_pred["right_hand_pose"] = (
            out["smplx_rhand_pose"].reshape(-1, 3).cpu().numpy()
        )
        smplx_pred["jaw_pose"] = out["smplx_jaw_pose"].reshape(-1, 3).cpu().numpy()
        smplx_pred["leye_pose"] = np.zeros((1, 3))
        smplx_pred["reye_pose"] = np.zeros((1, 3))
        smplx_pred["betas"] = out["smplx_shape"].reshape(-1, 10).cpu().numpy()
        smplx_pred["expression"] = out["smplx_expr"].reshape(-1, 10).cpu().numpy()
        smplx_pred["transl"] = out["cam_trans"].reshape(-1, 3).cpu().numpy()
        smplx_pred["focal"] = np.array(focal)
        smplx_pred["princpt"] = np.array(princpt)

        for k, v in smplx_pred.items():
            pose_result["smplerx_" + k] = v

        return pose_result


class Hand_var_init_hamer:
    def __init__(self):
        # init HAMER
        import os

        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        import sys

        sys.path.insert(0, HAMER_PATH)
        from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
        from hamer.utils.renderer import Renderer, cam_crop_to_full

        with path_enter(HAMER_PATH):
            self.hamer_model, self.hamer_model_cfg = load_hamer(DEFAULT_CHECKPOINT)
            self.hamer_model = self.hamer_model.cuda().eval()

        self.vars_to_save = [
            "hand_pose",
        ]

        logger.debug("HAMER")

    def get_result(self, xyxy, img, pose_result):
        from hamer.utils import recursive_to
        from hamer.datasets.vitdet_dataset import (
            ViTDetDataset,
            DEFAULT_MEAN,
            DEFAULT_STD,
        )

        bboxes = []
        is_right = []

        # 左手，如果有效的kpts数量小于3则使用上一帧的结果作为image
        # TODO: 添加hands的帧的mask
        keyp = pose_result["left_hand_keyp"]
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bboxes.append(bbox)
            self._last_bbox_L = bbox
            pose_result["Lh_valid"] = 1
        else:
            bboxes.append(self._last_bbox_L)
            pose_result["Lh_valid"] = 0
            pose_result["left_hand_keyp"][:, -1] = 0
        is_right.append(0)

        # 右手
        keyp = pose_result["right_hand_keyp"]
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bboxes.append(bbox)
            self._last_bbox_R = bbox
            pose_result["Rh_valid"] = 1
        else:
            bboxes.append(self._last_bbox_R)
            pose_result["Rh_valid"] = 0
            pose_result["right_hand_keyp"][:, -1] = 0
        is_right.append(1)

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        # import ipdb;ipdb.set_trace()
        dataset = ViTDetDataset(
            self.hamer_model_cfg, img, boxes, right, rescale_factor=2.0
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0
        )
        for batch in dataloader:
            batch = recursive_to(batch, "cuda")
            with torch.no_grad():
                out = self.hamer_model(batch)

        # 得到mano参数
        pose_result["hand_pose"] = (
            out["pred_mano_params"]["hand_pose"].cpu().numpy()
        )  # [2, 15, 3, 3]
        return pose_result


class Hand_var_init_acr:
    def __init__(self):
        # init ACR
        import os

        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        import sys

        sys.path.append(ACR_PATH)
        from acr.model import ACR as ACR_v1
        from acr.utils import load_model as acr_load_model

        self.acr_model = ACR_v1().eval()
        acr_kpt_path = "model_files/acr_ckpt_wild.pkl"
        self.acr_model = acr_load_model(
            acr_kpt_path,
            self.acr_model,
            prefix="module.",
            drop_prefix="",
            fix_loaded=False,
        )
        self.acr_model = nn.DataParallel(self.acr_model.cuda())
        self.acr_model.eval()

        from acr.config import default_acr_cfg

        self.default_acr_cfg = default_acr_cfg

        self.vars_to_save = [
            "hand_pose",
        ]

        logger.debug("ACR")

    def get_result(self, xyxy, img, pose_result):
        from acr.utils import img_preprocess

        bgr_frame = img[int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2]), :]
        # import cv2; cv2.imwrite('t.png',bgr_frame)
        meta_data = img_preprocess(
            bgr_frame,
            "asdsadf.png",
            input_size=self.default_acr_cfg.input_size,
            single_img_input=True,
        )
        meta_data["batch_ids"] = torch.arange(len(meta_data["image"]))
        outputs = self.acr_model(meta_data, **{"mode": "parsing", "calc_loss": False})
        # outputs['detection_flag']
        # outputs = self.mano_regression(outputs, outputs['meta_data'])
        # outputs['params_dict']['poses'][sid], outputs['params_dict']['betas'][sid]
        pose_result["hand_pose"] = tonp(
            a2m(outputs["params_dict"]["poses"][:, 3:].reshape(2, 15, 3))
        )
        return pose_result


class DetectionModel(object):
    def __init__(self, device, args):
        self.args = args

        self.min_det_joints = self.args.min_det_joints

        self.var_initers = []

        if self.args.use_smplerx:
            self.var_initers.append(Body_var_init_smplerx())

        if self.args.use_deca:
            self.var_initers.append(Face_var_init_deca())

        if self.args.use_smirk:
            self.var_initers.append(Face_var_init_smirk())

        if self.args.use_hmr2:
            self.var_initers.append(Body_var_init_hmr2())

        if self.args.use_hamer:
            self.var_initers.append(Hand_var_init_hamer())

        if self.args.use_acr:
            self.var_initers.append(Hand_var_init_acr())

        # ViTPose
        if self.args.vitpose_model == "body_only":
            # body only
            self.pose_model = init_pose_model(
                osp.join(
                    VIT_DIR,
                    "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py",
                ),
                "model_files/vitpose_ckpts/vitpose-h-multi-coco.pth",
                device=device.lower(),
            )
        else:
            # get face hands body kpts
            from lib.models.vitpose_model import ViTPoseModel

            self.cpm = ViTPoseModel("cuda", model_name=self.args.vitpose_model)

        # YOLO
        self.bbox_model = YOLO("model_files/wham_ckpt/yolov8x.pt")

        self.device = device
        self.initialize_tracking()

        self.all_vars_to_save = sum(
            [getattr(ii, "vars_to_save") for ii in self.var_initers], []
        ) + [
            "contact",
            "face_keyp",
            "left_hand_keyp",
            "right_hand_keyp",
            "Lh_valid",
            "Rh_valid",
            "bbox_xyxy",
            "bbox_cxcys",
        ]
        logger.debug(f"{self.all_vars_to_save=}")

    def initialize_tracking(
        self,
    ):
        self.next_id = 0
        self.frame_id = 0
        self.pose_results_last = []
        self.tracking_results = {
            "id": [],
            "frame_id": [],
            "bbox": [],
            "keypoints": [],
        }
        self._last_bbox_L = None
        self._last_bbox_R = None

    def xyxy_to_cxcys(self, bbox, s_factor=1.05):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200 * s_factor
        return np.array([[cx, cy, scale]])

    def compute_bboxes_from_keypoints(self, s_factor=1.2):
        X = self.tracking_results["keypoints"].copy()
        mask = X[..., -1] > VIS_THRESH

        bbox = np.zeros((len(X), 3))
        for i, (kp, m) in enumerate(zip(X, mask)):
            bb = [kp[m, 0].min(), kp[m, 1].min(), kp[m, 0].max(), kp[m, 1].max()]
            cx, cy = [(bb[2] + bb[0]) / 2, (bb[3] + bb[1]) / 2]
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            s = np.stack((bb_w, bb_h)).max()
            bb = np.array((cx, cy, s))
            bbox[i] = bb

        bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
        self.tracking_results["bbox"] = bbox

    def track(
        self,
        img,
        fps,
        length,
        pre_def_bboxes=None,
        scene_change_pos=None,
        current_frame=0,
        only_one_person=False,
    ):
        # img: np array
        # bbox detection
        if pre_def_bboxes is not None:
            bboxes = pre_def_bboxes
        else:
            bboxes = (
                self.bbox_model.predict(
                    img,
                    device=self.device,
                    classes=0,
                    conf=BBOX_CONF,
                    save=False,
                    verbose=False,
                )[0]
                .boxes.xyxy.detach()
                .cpu()
                .numpy()
            )
            bboxes = [{"bbox": bbox} for bbox in bboxes]
            # bboxes might be empty

        if self.args.vitpose_model == "body_only":
            # keypoints detection
            pose_results, returned_outputs = inference_top_down_pose_model(
                self.pose_model,
                img,
                person_results=bboxes,
                format="xyxy",
                return_heatmap=False,
                outputs=None,
            )
        else:
            # Detect human keypoints for each person
            if len(bboxes) > 0:
                pred_bboxes = np.stack([ii["bbox"] for ii in bboxes])
                vitpose_results = self.cpm.predict_pose(
                    img,
                    [
                        np.concatenate(
                            [pred_bboxes, np.ones_like(pred_bboxes[:, :1])], axis=1
                        )
                    ],
                )
                pose_results = deepcopy(vitpose_results)
            else:
                pose_results = []

            for idx in range(len(pose_results)):
                kp = pose_results[idx]["keypoints"]  # 133
                ipdb.set_trace()
                pose_results[idx]["left_hand_keyp"] = kp[-42:-21].copy()  # 21
                pose_results[idx]["right_hand_keyp"] = kp[-21:].copy()
                pose_results[idx]["face_keyp"] = kp[23 : 23 + 68, :].copy()  # 68
                pose_results[idx]["keypoints"] = kp[:17]

        # person identification
        tracking_thr = TRACKING_THR
        if scene_change_pos is not None:
            if current_frame in scene_change_pos and current_frame > 0:
                print(f"frame pos {current_frame} is short change...")
                tracking_thr = 1.1

        # import ipdb;ipdb.set_trace()
        if only_one_person:
            self.next_id = 0
            if len(pose_results) > 0:
                pose_results = pose_results[:1]
                pose_results[0]["track_id"] = 0
        else:
            pose_results, self.next_id = get_track_id(
                pose_results,
                self.pose_results_last,
                self.next_id,
                use_oks=False,
                tracking_thr=tracking_thr,
                use_one_euro=True,
                fps=fps,
            )

        for pose_result in pose_results:
            n_valid = (pose_result["keypoints"][:, -1] > VIS_THRESH).sum()  # (17, 3)
            if n_valid < self.min_det_joints:
                continue

            _id = pose_result["track_id"]
            xyxy = pose_result["bbox"]
            bbox = self.xyxy_to_cxcys(xyxy)

            pose_result["bbox_xyxy"] = xyxy
            pose_result["bbox_cxcys"] = bbox

            self.tracking_results["id"].append(_id)
            self.tracking_results["frame_id"].append(self.frame_id)
            self.tracking_results["bbox"].append(bbox)
            self.tracking_results["keypoints"].append(pose_result["keypoints"])

            for initer in self.var_initers:
                pose_result = initer.get_result(xyxy, img, pose_result)

            for k_to_add in self.all_vars_to_save:
                if k_to_add in pose_result.keys():
                    if k_to_add not in self.tracking_results.keys():
                        self.tracking_results[k_to_add] = []
                    self.tracking_results[k_to_add].append(pose_result[k_to_add])

        self.frame_id += 1
        self.pose_results_last = pose_results

    def process(self, fps):
        # import ipdb;ipdb.set_trace()
        for k in self.tracking_results.keys():
            self.tracking_results[k] = np.array(self.tracking_results[k])

        self.compute_bboxes_from_keypoints()

        output = defaultdict(dict)
        ids = np.unique(self.tracking_results["id"])
        for _id in ids:
            output[_id]["features"] = []
            idxs = np.where(self.tracking_results["id"] == _id)[0]
            for key, val in self.tracking_results.items():
                if key == "id":
                    continue
                output[_id][key] = val[idxs]

        # Smooth bounding box detection
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]["bbox"]) < MINIMUM_FRMAES:
                del output[_id]
                continue

            kernel = int(int(fps / 2) / 2) * 2 + 1
            smoothed_bbox = np.array(
                [signal.medfilt(param, kernel) for param in output[_id]["bbox"].T]
            ).T
            output[_id]["bbox"] = smoothed_bbox

        return output
