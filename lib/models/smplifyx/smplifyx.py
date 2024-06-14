import os
import torch
from tqdm import tqdm
import numpy as np
from lib.models import build_body_model
from lib.loco.trajdiff import *
   
from lib.models.smplifyx.proxy_utils import BodyProxy
from lib.models.smplifyx.hands_proxy import HandsProxy
from lib.models.smplifyx.PCA_hand_proxy import PCAHandsProxy


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def compute_jitter(x):
    """
    Compute jitter for the input tensor
    """
    return torch.linalg.norm(x[:, 2:] + x[:, :-2] - 2 * x[:, 1:-1], dim=-1)

def l2smooth(x): return ((x[1:] - x[:-1])**2).mean()

lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,67, 28, 29, 30, 68, 34, 35, 36, 69,31, 32, 33, 70], dtype=np.int32)
rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,43, 44, 45, 73, 49, 50, 51, 74, 46,47, 48, 75], dtype=np.int32)#21
# face_mapping = np.arange(76, 127 + 17,dtype=np.int32)#68
face_mapping = np.concatenate([np.arange(127, 127+17,dtype=np.int32), np.arange(76, 127,dtype=np.int32)], axis=0)

# 开启的loss
keys_to_reg = [
    
]
keys_to_jitter = [
    'cam',
    'hand_pose',
]
in_keys_to_reg = [
    'exp',
    'betas',
]
in_keys_to_jitter = [
    'orient'
]
k_cmpl_v_coco = {
    8:16,11:16,
    7:15,10:15,
    5:14,4:13,
    2:12,1:11,
    14:6,17:6,
    13:5,16:5,
    19:8,18:7,
    21:10,20:9,
}
# loss weights
losses_w = {
    'F_kpts_proj': 100.*10,
    'B_kpts_proj': 100.,
    'Lh_kpts_proj': 100.,
    'Rh_kpts_proj': 100.,
    
    'orient_jitter': 20.0*30,
    'cam_jitter': 20.0,
    'hand_pose_jitter': 5.0,
    
    'in_z_l_reg': 60.0,
    'in_orient_jitter': 20.0,
    'B_kpts_jitter': 10.0,
    'F_kpts_jitter': 5.0,
    
    's_prior': 4,
    's_consit': 10.0,
    
    'lower_reg': 1000.0,
    
    'H_pca_jitter': 100.0,
    'exp_jitter': 1.0,
    
    'pose_reg': 20.0,
    'in_exp_reg': 10.0,
    'in_betas_reg': 10.0,
}

opt_type = 'lbfgs'
# opt_type = 'adam'

def params_dec(params_in, proxy_dict):
    # decode from body proxy
    if 'pose' in proxy_dict:
        pose, consit = proxy_dict['pose'].dec(
                {
                'cvt_vars': {
                    'orient': params_in['orient'],
                    'z_l': params_in['z_l'],
                }
            })
    else: pose = params_in['pose']
    
    # decode from two hands proxy
    if 'hand_pose' in proxy_dict:
        hand_pose = proxy_dict['hand_pose'].dec(
                {
                'cvt_vars': {
                    'z_hL': params_in['z_hL'],
                    'z_hR': params_in['z_hR'],
                }
            })
    else: hand_pose = params_in['hand_pose']
    
    # import ipdb;ipdb.set_trace()
    T = pose.shape[1]
    betas = params_in['betas'].unsqueeze(1).repeat(1,T,1)
    params = {
        'pose': pose,
        'betas': betas,
        'cam': params_in['cam'],
        'hand_pose': hand_pose,
        'jaw': params_in['jaw'],
        'leye': params_in['leye'],
        'reye': params_in['reye'],
        'exp': params_in['exp'],
    }
    
    misc_info = {}
    consit = torch.tensor(0)
    misc_info['consit'] = consit
    return params, misc_info

def params_enc(init_param_th, proxy_dict):
    init_param_th_proxy = {}
    for var_name, var in init_param_th.items():
        if var_name in proxy_dict.keys():
            init_param_th_proxy |= proxy_dict[var_name].enc(var)['cvt_vars']
        else:
            init_param_th_proxy[var_name] = var
    init_param_th_proxy['betas'] = init_param_th_proxy['betas'][:,0]
    return init_param_th_proxy

    

class SMPLifyLoss(torch.nn.Module):
    def __init__(self, 
                 res,
                 cam_intrinsics,
                 init_param_th_org,
                 init_param_th, 
                 device,
                 **kwargs
                 ):
        
        super().__init__()
        
        self.res = res
        self.cam_intrinsics = cam_intrinsics
        self.cam_intrinsics = cam_intrinsics
        def to_type(x): return x.float().to(device).detach().clone()
        self.init_param = {k: to_type(v) for k, v in init_param_th.items()}
        self.init_param_th_org = {k: to_type(v) for k, v in init_param_th_org.items()}
        
    def forward(self, smplx, params_in, parse_dst_data, bbox, proxy_dict, args,):
        sigma=100
        # <============ get predict data
        # proxy dec here
        params, misc_info = params_dec(params_in, proxy_dict)
        output = smplx(
            betas=params['betas'],  #T,10
            Pose=params['pose'], # T,24,6
            Lh=params['hand_pose'][:,:,0,:,:], #T,15,6
            Rh=params['hand_pose'][:,:,1,:,:],
            Exp=params['exp'], #T,50
            Jaw=params['jaw'], #T,3
            Leye=params['leye'], #T,3
            Reye=params['reye'],
            cam=params['cam'],
            cam_intrinsics=self.cam_intrinsics, 
            bbox=bbox, res=self.res
        )
        pred_data = {
            'B_kpts': output.full_joints2d[..., :17, :],
            'Lh_kpts': output.smpl_output_joints2d[..., lhand_mapping, :],
            'Rh_kpts': output.smpl_output_joints2d[..., rhand_mapping, :],
            'F_kpts': output.smpl_output_joints2d[..., face_mapping, :],
        }        
        # <============
        
                
        
        losses = {}
        # import ipdb;ipdb.set_trace()
        scale = bbox[..., 2:].unsqueeze(-1) * 200.
        
        if args.replace_lower:
            lower_reg = 0
            pose = params['pose'].reshape(-1,24,6)
            norm_pose = pose - torch.tensor([1.0,0,0,0,1,0],device=pose.device)
            lower_reg += ((norm_pose[:,np.array([0,1,3,4,6,7,9,10])+1])**2).mean()
            # lower_reg += ((norm_pose[:,np.array([2,5,8,])+1])**2).mean()
            losses['lower_reg'] = lower_reg
            
            # parse_dst_data['B_kpts']['conf'][:,[5,6]] = 0
        
        # if 0:    
        #     import ipdb;ipdb.set_trace()
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     # data = pred_data['B_kpts'].detach().cpu().numpy()[0][0]
        #     # data = parse_dst_data['B_kpts']['data'].detach().cpu().numpy()[0][0]
            
        #     # data = pred_data['Rh_kpts'].detach().cpu().numpy()[0][100]
        #     # data = parse_dst_data['Rh_kpts']['data'].detach().cpu().numpy()[0][100]
            
        #     data = pred_data['F_kpts'].detach().cpu().numpy()[0][100]
        #     # data = parse_dst_data['F_kpts']['data'].detach().cpu().numpy()[0][100]
            
        #     index_text = list(range(len(data)))
        #     plt.scatter(data[:,0],data[:,1],color='b',label='')
        #     for idx in index_text:
        #         plt.text(data[idx,0],data[idx,1],str(idx),color='r')
        #     plt.legend()
        #     plt.axis('equal')
        #     # inverse axis
        #     ax = plt.gca()
        #     ax.invert_yaxis()
        #     plt.savefig('t1.png')   
            
                    
        # Loss 1. Data term
        for k, v in pred_data.items():
            losses[f'{k}_proj']= ((gmof(pred_data[k] - parse_dst_data[k]['data'], sigma) * parse_dst_data[k]['conf'] * (parse_dst_data[k]['conf']>0.5)) / scale).mean()
            
        # Loss 2. Regularization term
        for k in keys_to_reg: losses[f'{k}_reg'] = torch.linalg.norm(params[k] - self.init_param[k], dim=-1).mean()

        # import ipdb;ipdb.set_trace()
        for k in in_keys_to_reg: losses[f'in_{k}_reg'] = torch.linalg.norm(params_in[k] - self.init_param[k], dim=-1).mean()

        # Loss 4. Smooth loss
        for k in keys_to_jitter: losses[f'{k}_jitter'] = compute_jitter(params[k]).mean()
                    
        for k in in_keys_to_jitter: losses[f'in_{k}_jitter'] = l2smooth(params_in[k])
        
        losses['B_kpts_jitter'] = compute_jitter(pred_data['B_kpts'] / scale).mean()
        losses['F_kpts_jitter'] = compute_jitter(pred_data['F_kpts'] / scale).mean()
        losses['exp_jitter'] = compute_jitter(params_in['exp']).mean()
        
        if 'z_hL' in params_in:
            losses['H_pca_jitter'] = l2smooth(params_in['z_hL']) + l2smooth(params_in['z_hR'])
        
        if args.use_opt_pose_reg:
            kpts_conf=(parse_dst_data['B_kpts']['conf']>0.5).float()
            pose_conf=torch.zeros([*(params['pose'].shape[:2]),24,1]).type_as(params['pose'])
            pose_conf[:,:,list(k_cmpl_v_coco.keys()),:]=kpts_conf[:,:,list(k_cmpl_v_coco.values()),:]
            losses['pose_reg'] = torch.linalg.norm((params['pose'] - self.init_param_th_org['pose']).reshape(-1,24,6)*(pose_conf), dim=-1).mean()
                        
            # losses['pose_reg'] = torch.linalg.norm((params['pose'] - self.init_param_th_org['pose']).reshape(-1,24,6), dim=-1).mean()

        # Loss 3. Shape prior and consistency error
        shape = params['betas']
        # losses['s_consit'] = shape.std(dim=1).mean()
        
        losses['s_prior'] = torch.linalg.norm(shape, dim=-1).mean()
        
        # losses['consit'] = misc_info['consit']
        
        # Sum up losses
        losses_w_weight = {}
        for k in losses.keys():
            losses_w_weight[k] = losses[k] * losses_w[k]
            
        if args.debug:
            log_str = ', '.join([f'loss/{k}: {v.item():.4f}' for k, v in losses.items()])
            print(log_str)
            
        # import ipdb;ipdb.set_trace()
        return losses_w_weight
        
    def create_closure(
        self,
        optimizer,
        smplx, 
        params_in,
        bbox,
        parse_dst_data,
        proxy_dict,
        args,
    ):
        
        def closure():
            optimizer.zero_grad()
            loss_dict = self.forward(smplx, params_in, parse_dst_data, bbox, proxy_dict, args,)
            loss = sum(loss_dict.values())
            loss.backward()
            return loss
        
        return closure
    
from copy import deepcopy
class TemporalSMPLifyX():
    
    def __init__(self, 
                 smplx=None,
                 img_w=None,
                 img_h=None,
                 device=None
                 ):
        
        self.smplx = smplx
        self.img_w = img_w
        self.img_h = img_h
        self.device = device
        
    def fit(
        self, 
        init_param_th, 
        dst_data,
        args, 
        bbox, 
        **kwargs
    ):
        # import ipdb;ipdb.set_trace()

        proxy_dict = {
            'pose': BodyProxy(),
            # 'hand_pose': HandsProxy(),
            'hand_pose': PCAHandsProxy(self.smplx),
        }
        init_param_th_org = deepcopy((init_param_th))
        init_param_th = params_enc(init_param_th, proxy_dict)
        
        # <===========================
        # 读取dst数据，torch
        def to_type(x): return torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        B_kpts = to_type(dst_data['keypoints'])
        Lh_kpts = to_type(dst_data['left_hand_keyp'])
        Rh_kpts = to_type(dst_data['right_hand_keyp'])
        F_kpts = to_type(dst_data['face_keyp'])
        parse_dst_data = {
            'B_kpts': {
                'data': B_kpts[..., :17, :2],
                'conf': B_kpts[..., :17, -1:],
            },
            'Lh_kpts': {
                'data': Lh_kpts[..., :2],
                'conf': Lh_kpts[..., -1:],
            },
            'Rh_kpts': {
                'data': Rh_kpts[..., :2],
                'conf': Rh_kpts[..., -1:],
            },
            'F_kpts': {
                'data': F_kpts[..., :2],
                'conf': F_kpts[..., -1:],
            },            
        }
        
        
        
        # <===========================
        def to_params(x): return x.to(self.device).requires_grad_(True)
        init_param_th = {k: to_params(v) for k, v in init_param_th.items()}
        all_params_keys = list(init_param_th.keys())
        print(f'{all_params_keys=}')
        BN = B_kpts.shape[1]
        
        stage_dict = {
            '1': {
                'lr': args.opt_lr,
                'num_iters': 30,
                'num_steps': 5,
                'opt_param_names': ['cam'],
            },
            '2': {
                'lr': args.opt_lr,
                'num_iters': 30,
                'num_steps': 10,
                'opt_param_names': all_params_keys,
                # 'opt_param_names': [ii for ii in all_params_keys if ii not in ['exp','jaw']],
            }
        }
        
        
        for stage_name, stage_info in stage_dict.items():
            # break
        
            print(f'[INFO] optimize {stage_name}')
            optim_params = [init_param_th[k] for k in stage_info['opt_param_names']]
            
            # import ipdb;ipdb.set_trace()
            if opt_type=='lbfgs':
                optimizer = torch.optim.LBFGS(
                    optim_params, 
                    lr=stage_info['lr'], 
                    max_iter=stage_info['num_iters'], 
                    line_search_fn='strong_wolfe')
            # else:
            #     self.num_steps = 100
            #     optimizer = torch.optim.Adam(
            #         optim_params, 
            #         lr=0.001 * BN, 
            #     )
                            
            loss_fn = SMPLifyLoss(init_param_th_org=init_param_th_org, init_param_th=init_param_th, device=self.device, **kwargs)
            
            closure = loss_fn.create_closure(
                optimizer,
                self.smplx, 
                init_param_th,
                bbox,
                parse_dst_data,
                proxy_dict, 
                args,
            )
            
            for j in (j_bar := tqdm(range(stage_info['num_steps']), leave=False)):
                optimizer.zero_grad()
                loss = optimizer.step(closure)
                msg = f'Loss: {loss.item():.1f}'
                j_bar.set_postfix_str(msg)

        # <===========================
        # 返回最后的参数
        init_param_th, _ = params_dec(init_param_th, proxy_dict)
        init_param_th_detach = {}
        for k in init_param_th: init_param_th_detach[k] = init_param_th[k].detach()
        return init_param_th_detach
