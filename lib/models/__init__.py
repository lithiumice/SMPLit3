import os, sys
import yaml
import torch
from loguru import logger

from configs import constants as _C
from .smpl import SMPL
from .my_smplx import SMPLX
from .cvt_hand_pca import attach_smplx_pca_func


def build_body_model(device, batch_size=1, gender='neutral', **kwargs):
    sys.stdout = open(os.devnull, 'w')
    body_model = SMPL(
        model_path=_C.BMODEL.FLDR,
        gender=gender,
        batch_size=batch_size,
        create_transl=False).to(device)
    sys.stdout = sys.__stdout__
    return body_model


def build_smplx_body_model(device, batch_size=1, args=None, gender='neutral', **kwargs):
    sys.stdout = open(os.devnull, 'w')
    if args.use_acr:
        #False的时候会加上mean pose
        flat_hand_mean = False
    else:
        flat_hand_mean = True
    body_model = SMPLX(
        model_path = _C.BMODEL.SMPLX_model_path,
        num_betas             = 10,
        num_expression_coeffs = 50,
        use_hands             = True,
        use_face_contour      = True,
        use_face              = True,
        use_pca               = False,
        # use_pca               = True,
        num_pca_comps = 6,
        flat_hand_mean        = flat_hand_mean,
        batch_size=batch_size,
        create_transl=False
    ).to(device)
    sys.stdout = sys.__stdout__
    # import ipdb;ipdb.set_trace()
    attach_smplx_pca_func(body_model)
    return body_model


def build_network(cfg, smpl):
    from .wham import Network
    
    with open(cfg.MODEL_CONFIG, 'r') as f:
        model_config = yaml.safe_load(f)
    model_config.update({'d_feat': _C.IMG_FEAT_DIM[cfg.MODEL.BACKBONE]})
    
    network = Network(smpl, **model_config).to(cfg.DEVICE)
    setattr(network, 'cfg', cfg)
    
    # Load Checkpoint
    if os.path.isfile(cfg.TRAIN.CHECKPOINT):
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT)
        ignore_keys = ['smpl.body_pose', 'smpl.betas', 'smpl.global_orient', 'smpl.J_regressor_extra', 'smpl.J_regressor_eval']
        model_state_dict = {k: v for k, v in checkpoint['model'].items() if k not in ignore_keys}
        network.load_state_dict(model_state_dict, strict=False)
        logger.info(f"=> loaded checkpoint '{cfg.TRAIN.CHECKPOINT}' ")
    else:
        logger.info(f"=> Warning! no checkpoint found at '{cfg.TRAIN.CHECKPOINT}'.")
        
    return network