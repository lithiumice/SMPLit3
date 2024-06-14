from infer import *
from easydict import EasyDict
from tempfile import TemporaryDirectory
import shutil


print=logger.info

__cur__=os.path.join(os.path.dirname(__file__))

def init_model_org():
    args = dict(
        calib=None, estimate_local_only=True, visualize=True, save_pkl=True, not_full_body=True, anno_pkl_path=None, use_smplx=True, 
        use_hamer=False, use_deca=False, use_smirk=False, use_acr=True, debug=True, save_split_vid=False, save_normal_map=False, 
        save_depth_map=False, save_prefix='', vitpose_model='body_only', vis_kpts=False, save_ex=False, use_smplerx=False, 
        only_one_person=True, use_opt_pose_reg=False, skip_save_vis_if_exists=False, save_opt_tmp=False, opt_lr=1.0, runing_fps=-1, 
        save_wham_npz=False, run_smplify=False, replace_lower=False, use_hmr2=False, inp_body_pose=None, opts=[])
    args = EasyDict(args)
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    cfg.merge_from_list(args.opts)


    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()

    smplx = build_smplx_body_model(cfg.DEVICE, smpl_batch_size, args)
    setattr(network, 'smplx', smplx)
    
    detector = DetectionModel(cfg.DEVICE.lower(), args)
    
    return (
        cfg, 
        args,
        network,
        detector
    )

def mocap_org(
    models,
    input_video_path,
    output_smplx_path,
):
    # must use abs path
    # import ipdb;ipdb.set_trace()
    output_smplx_path = os.path.abspath(output_smplx_path)
    
    (
        cfg, 
        args,
        network,
        detector
    ) = models
    with TemporaryDirectory() as temp_dir:
        with torch.no_grad():
            run(cfg, 
                input_video_path, 
                temp_dir, 
                network, 
                detector=detector,
                calib=None, 
                run_global=False, 
                save_pkl=True,
                visualize=False,
                args=args,
                )
        src_npz = glob(f'{temp_dir}/*.npz')[0]
        os.makedirs(os.path.dirname(output_smplx_path), exist_ok=True)
        shutil.copy(
            src_npz,
            output_smplx_path,
        )
        print(f'Save result to {output_smplx_path=}')
       
def init_model():
    with path_enter(__cur__):
        ret = init_model_org()     
    return ret

def mocap(*args, **kwargs):
    with path_enter(__cur__):
        ret = mocap_org(*args, **kwargs)
    return ret

