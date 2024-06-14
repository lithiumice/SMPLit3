import os
import json
import sys
sys.path.insert(0,'/apdcephfs/private_wallyliang/PLANT/difftraj')
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from utils.model_util import create_gaussian_diffusion
from data_loaders.get_data import get_model_args,collate
import torch

args = train_args()
args.dataset = 'AMASS_GLAMR_taming_style'
fixseed(args.seed)
train_platform_type = eval(args.train_platform_type)
train_platform = train_platform_type(args.save_dir)
train_platform.report_args(args, name='Args')
os.makedirs(args.save_dir,exist_ok=True)
args_path = os.path.join(args.save_dir, 'args.json')
with open(args_path, 'w') as fw: json.dump(vars(args), fw, indent=4, sort_keys=True)
dist_util.setup_dist(args.device)
print("creating data loader...")
data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,args=args)
print("creating model and diffusion...")
from model.mdm_taming_style import MDM
model = MDM(**get_model_args(args, data),inp_len=data.dataset.t2m_dataset.inp_len)
diffusion = create_gaussian_diffusion(args)
model.to(dist_util.dev())

if os.path.exists(args.base_model):
    checkpoints = torch.load(args.base_model)
    miss_key, unexpect_key = model.load_state_dict(checkpoints, strict=False)
    
if hasattr(model,'rot2xyz'): model.rot2xyz.smpl_model.eval()
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
TrainLoop(args, train_platform, model, diffusion, data).run_loop()
train_platform.close()


# import os
# import json
# import sys
# sys.path.insert(0,'/apdcephfs/private_wallyliang/PLANT/difftraj')
# from utils.fixseed import fixseed
# from utils.parser_util import train_args
# from utils import dist_util
# from train.training_loop import TrainLoop
# from data_loaders.get_data import get_dataset_loader
# from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
# from utils.model_util import create_gaussian_diffusion
# from dataloader.get_datas import get_model_args

# args = train_args()
# fixseed(args.seed)
# train_platform_type = eval(args.train_platform_type)
# train_platform = train_platform_type(args.save_dir)
# train_platform.report_args(args, name='Args')
# os.makedirs(args.save_dir,exist_ok=True)
# args_path = os.path.join(args.save_dir, 'args.json')
# with open(args_path, 'w') as fw: json.dump(vars(args), fw, indent=4, sort_keys=True)
# dist_util.setup_dist(args.device)
# print("creating data loader...")
# data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,args=args)
# print("creating model and diffusion...")
# from model.mdm_taming import MDM
# model = MDM(**get_model_args(args, data))
# diffusion = create_gaussian_diffusion(args)
# model.to(dist_util.dev())
# if hasattr(model,'rot2xyz'): model.rot2xyz.smpl_model.eval()
# print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
# TrainLoop(args, train_platform, model, diffusion, data).run_loop()
# train_platform.close()
