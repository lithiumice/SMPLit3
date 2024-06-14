from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys

import torch
import numpy as np
from lib.utils import transforms

from smplx import SMPLX as _SMPLX
from smplx.utils import SMPLXOutput as ModelOutput
from smplx.lbs import vertices2joints

from configs import constants as _C
from lib.loco.trajdiff import *

    
    
class SMPLX(_SMPLX):
    """ Extension of the official SMPLX implementation to support more joints """

    def __init__(self, *args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        super(SMPLX, self).__init__(*args, **kwargs)
        sys.stdout = sys.__stdout__
        
        J_regressor_wham = np.load(_C.BMODEL.JOINTS_REGRESSOR_WHAM)
        J_regressor_eval = np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
        self.register_buffer('J_regressor_wham', torch.tensor(
            J_regressor_wham, dtype=torch.float32))
        self.register_buffer('J_regressor_eval', torch.tensor(
            J_regressor_eval, dtype=torch.float32))
        self.register_buffer('J_regressor_feet', torch.from_numpy(
            np.load(_C.BMODEL.JOINTS_REGRESSOR_FEET)
        ).float())
        

        smplx_to_smpl = 'model_files/uhc_data/wham_data/smplx2smpl.pkl'
        with open(smplx_to_smpl,'rb') as f: import pickle;smplx_to_smpl = pickle.load(f, encoding='latin1')
        smplx2smpl = torch.tensor(smplx_to_smpl['matrix']).float().to(self.J_regressor_wham.device)
        self.J_regressor_wham_smplx = torch.matmul(self.J_regressor_wham, smplx2smpl).cuda()
        self.J_regressor_feet_smplx = torch.matmul(self.J_regressor_feet, smplx2smpl).cuda()
                
        
    def get_local_pose_from_reduced_global_pose(self, reduced_pose):
        full_pose = torch.eye(
            3, device=reduced_pose.device
        )[(None, ) * 2].repeat(reduced_pose.shape[0], 24, 1, 1)
        full_pose[:, _C.BMODEL.MAIN_JOINTS] = reduced_pose
        return full_pose

    def forward(self, 
                betas,  #1,T,10
                Pose, # 1,T,24,6
                Lh, #1,T,15,6
                Rh,
                Exp, #1,T,50
                Jaw, #1,T,3
                Leye, #1,T,3
                Reye,
                cam=None, 
                cam_intrinsics=None, 
                bbox=None, 
                res=None,
                return_full_pose=False):
        """
        all pose in rot 6D format
        """
        def cvt_vars(x): return s2a(x.reshape(*x.shape[:2], -1, 6))[0]
        Pose = cvt_vars(Pose)
        Lh = cvt_vars(Lh)
        Rh = cvt_vars(Rh)
        Jaw = cvt_vars(Jaw)
        Leye = cvt_vars(Leye)
        Reye = cvt_vars(Reye)
        Exp = Exp[0]

        output = self.get_output(
            betas=betas.view(-1, 10),
            global_orient=Pose[:, :1],
            body_pose=Pose[:, 1:22],
            left_hand_pose=Lh,
            right_hand_pose=Rh,
            expression=Exp,
            jaw_pose=Jaw,
            leye_pose=Leye,
            reye_pose=Reye,
            pose2rot=False,
            return_full_pose=return_full_pose)

        # import ipdb;ipdb.set_trace()
        if cam is not None:
            joints3d = output.joints.reshape(*cam.shape[:2], -1, 3)
            
            # Weak perspective projection (for InstaVariety)
            weak_cam = convert_weak_perspective_to_perspective(cam)
            
            weak_joints2d = weak_perspective_projection(
                joints3d,
                rotation=torch.eye(3, device=cam.device).unsqueeze(0).unsqueeze(0).expand(*cam.shape[:2], -1, -1),
                translation=weak_cam,
                focal_length=5000.,
                camera_center=torch.zeros(*cam.shape[:2], 2, device=cam.device)
            )
            output.weak_joints2d = weak_joints2d
            
            # Full perspective projection
            full_cam = convert_pare_to_full_img_cam(
                cam, 
                bbox[:, :, 2] * 200., 
                bbox[:, :, :2], 
                res[:, 0].unsqueeze(-1), 
                res[:, 1].unsqueeze(-1), 
                focal_length=cam_intrinsics[:, :, 0, 0]
            )
            
            full_joints2d = full_perspective_projection(
                joints3d,
                translation=full_cam,
                cam_intrinsics=cam_intrinsics,
            )
            output.full_joints2d = full_joints2d
            output.full_cam = full_cam.reshape(-1, 3)
            
            smpl_output_joints = output.smpl_output_joints
            output.smpl_output_joints2d = full_perspective_projection(
                smpl_output_joints,
                translation=full_cam,
                cam_intrinsics=cam_intrinsics,
            )
            
            
            # import ipdb;ipdb.set_trace()
            # from Thirdparty.PIXIE_fork.pixielib.utils.util import estimate_translation_np
            # # import ipdb;ipdb.set_trace()
            # S1=joints3d.detach().cpu().numpy()[0][:,:17]
            # keypoints=full_joints2d.detach().cpu().numpy()[0][:,:17]
            # j2d = keypoints[:,:,:2]
            # # j2d[:,:,0]=-j2d[:,:,0]+width
            # # j2d[:,:,1]=-j2d[:,:,1]+height   
                
            # conf1=np.ones_like(keypoints[:,:,0])
            # T=S1.shape[0]
            # est_T = np.zeros((T,3))
            # for idx in range(T):
            #     est_t = estimate_translation_np(S1[idx], j2d[idx], conf1[idx], tonp(cam_intrinsics[0][0]))
            #     est_T[idx]=est_t
            # pred['cam'] = t2t(est_T).unsqueeze(0)
        
            # pare_cam = convert_full_img_cam_to_pare(
            #     full_cam, 
            #     bbox_height=bbox[:, :, 2] * 200., 
            #     bbox_center=bbox[:, :, :2],
            #     img_w=res[:, 0].unsqueeze(-1), 
            #     img_h=res[:, 1].unsqueeze(-1), 
            #     focal_length=cam_intrinsics[:, :, 0, 0], 
            #     crop_res=224
            # )
            
        return output
    
    def forward_nd(self, 
                pred_rot6d, 
                root,
                betas, 
                return_full_pose=False):
        
        rotmat = transforms.rotation_6d_to_matrix(pred_rot6d.reshape(*pred_rot6d.shape[:2], -1, 6)
        ).reshape(-1, 24, 3, 3)

        output = self.get_output(body_pose=rotmat[:, 1:],
                                 global_orient=root.reshape(-1, 1, 3, 3),
                                 betas=betas.view(-1, 10),
                                 pose2rot=False,
                                 return_full_pose=return_full_pose)

        return output

    def get_output(self, *args, **kwargs):
        kwargs['get_skin'] = True
        kwargs['pose2rot'] = True
        # for k,v in kwargs.items(): print(f'{k}: {v.shape}')
        smpl_output = super(SMPLX, self).forward(*args, **kwargs)
        
        smplx_vert = smpl_output.vertices
        joints = vertices2joints(self.J_regressor_wham_smplx, smplx_vert)
        feet = vertices2joints(self.J_regressor_feet_smplx, smplx_vert)  
              
        # smpl_vert = torch.bmm(smplx2smpl[None].expand(smplx_vert.shape[0], -1, -1), smplx_vert)
        # joints2 = vertices2joints(self.J_regressor_wham, smpl_vert)
        # feet2 = vertices2joints(self.J_regressor_feet, smpl_vert)
        
        
        offset = joints[..., [11, 12], :].mean(-2)
        if 'transl' in kwargs:
            offset = offset - kwargs['transl']
            
        smpl_output_joints = smpl_output.joints - offset.unsqueeze(-2)
        vertices = smplx_vert - offset.unsqueeze(-2)
        joints = joints - offset.unsqueeze(-2)
        feet = feet - offset.unsqueeze(-2)

        output = ModelOutput(vertices=vertices,#smplx vert
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        output.feet = feet
        output.offset = offset
        output.smpl_output_joints = smpl_output_joints
        return output
    
    def get_offset(self, *args, **kwargs):
        kwargs['get_skin'] = True
        kwargs['pose2rot'] = True
        smpl_output = super(SMPLX, self).forward(*args, **kwargs)
        joints = vertices2joints(self.J_regressor_wham_smplx, smplx_vert)
        offset = joints[..., [11, 12], :].mean(-2)
        return offset
    

def convert_weak_perspective_to_perspective(
        weak_perspective_camera,
        focal_length=5000.,
        img_res=224,
):
    
    perspective_camera = torch.stack(
        [
            weak_perspective_camera[..., 1],
            weak_perspective_camera[..., 2],
            2 * focal_length / (img_res * weak_perspective_camera[..., 0] + 1e-9)
        ],
        dim=-1
    )
    return perspective_camera    


def weak_perspective_projection(
        points, 
        rotation, 
        translation,
        focal_length, 
        camera_center, 
        img_res=224,
        normalize_joints2d=True,
):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (b, f, N, 3): 3D points
        rotation (b, f, 3, 3): Camera rotation
        translation (b, f, 3): Camera translation
        focal_length (b, f,) or scalar: Focal length
        camera_center (b, f, 2): Camera center
    """

    K = torch.zeros([*points.shape[:2], 3, 3], device=points.device)
    K[:,:,0,0] = focal_length
    K[:,:,1,1] = focal_length
    K[:,:,2,2] = 1.
    K[:,:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bfij,bfkj->bfki', rotation, points)
    points = points + translation.unsqueeze(-2)

    # Apply perspective distortion
    projected_points = points / points[...,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bfij,bfkj->bfki', K, projected_points)
    
    if normalize_joints2d:
        projected_points = projected_points / (img_res / 2.) 

    return projected_points[..., :-1]

    
def full_perspective_projection(
        points, 
        cam_intrinsics, 
        rotation=None,
        translation=None,
):

    K = cam_intrinsics

    if rotation is not None:
        points = (rotation @ points.transpose(-1, -2)).transpose(-1, -2)
    if translation is not None:
        points = points + translation.unsqueeze(-2)
    projected_points = points / (points[..., -1].unsqueeze(-1)+1e-8)
    projected_points = (K @ projected_points.transpose(-1, -2)).transpose(-1, -2)
    return projected_points[..., :-1]


def convert_full_img_cam_to_pare(
        full_cam, 
        bbox_height, 
        bbox_center,
        img_w, 
        img_h, 
        focal_length, 
        crop_res=224
):

    tx_cx, ty_cy, tz = full_cam[..., 0], full_cam[..., 1], full_cam[..., 2]

    res = crop_res
    r = bbox_height / res
    s = 2 * focal_length / (r * res * tz)

    cx = 2 * (bbox_center[..., 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[..., 1] - (img_h / 2.)) / (s * bbox_height)

    tx = tx_cx - cx
    ty = ty_cy - cy

    pare_cam = torch.stack([s, tx, ty], dim=-1)
    return pare_cam


def convert_pare_to_full_img_cam(
        pare_cam, 
        bbox_height, 
        bbox_center,
        img_w, 
        img_h, 
        focal_length, 
        crop_res=224
):

    s, tx, ty = pare_cam[..., 0], pare_cam[..., 1], pare_cam[..., 2]
    res = crop_res
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[..., 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[..., 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t


def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam