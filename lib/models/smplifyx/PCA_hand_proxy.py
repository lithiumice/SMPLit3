
import os;
import sys;
import torch

torch.set_default_dtype(torch.float32)
from lib.loco.trajdiff import *
    


class PCAHandsProxy():
    def __init__(self, smplx):
        self.smplx = smplx

    def forward(self):
        pass
    
    def enc(self, pose):
        """
        pose: [1, T, 2, 15, 6])
        """

        pose = s2a(pose.squeeze().reshape(-1,2,15,6)) #[T, 2, 15, 3]
        org_T = pose.shape[0]
        
        Lh_pose  = pose[:,0,:,:]#T,15,3
        Rh_pose  = pose[:,1,:,:]
        
        # import ipdb;ipdb.set_trace()
        Lh_pca, Rh_pca = self.smplx.hand_axis_to_pca(self.smplx, Lh_pose, Rh_pose)
        
        proxy_var = self._proxy_var = {
            'org_T': org_T,
            'cvt_vars': {
                'z_hL': Lh_pca,
                'z_hR': Rh_pca,
            }
        }
        return self._proxy_var
    
    def dec(self, proxy_var):
        """
        B, 512
        """
        org_T = self._proxy_var['org_T']
        # import ipdb;ipdb.set_trace()
        Lh_pca = proxy_var['cvt_vars']['z_hL']
        Rh_pca = proxy_var['cvt_vars']['z_hR']
        Lh_rec_pose, Rh_rec_pose = self.smplx.hand_pca_to_axis(self.smplx, Lh_pca, Rh_pca)
        rec_pose = torch.stack([Lh_rec_pose.reshape(-1,15,3), Rh_rec_pose.reshape(-1,15,3)],dim=1).unsqueeze(0)
        # rec_pose: [1, T, 2, 15, 3]
        rec_pose = a2s(rec_pose)
        # rec_pose: [1, T, 2, 15, 6]
        return rec_pose
    
    def get_proxy(self):
        return self._proxy_var
    