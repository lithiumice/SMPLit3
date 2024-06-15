import torch
import numpy as np
from lib.loco.trajdiff import *

# (PIXIE_init_hand-hand_mean)@inv_hand_comp=hand_pca_delta
# hand_pca_delta@hand_comp+hand_mean=PIXIE_init_hand
# hand_pca_full@hand_comp=PIXIE_init_hand


def attach_smplx_pca_func(model):
    """
    PCA -> axis
    """

    def hand_pca_to_axis(self, lhand_pca, rhand_pca):
        """
        input:
            lhand_pca: T,pca_num
            rhand_pca: T,pca_num

        return:
            lhand_axis: T,45
            rhand_axis: T,45
        """

        def xhand_cvt(H_pca, H_mean, H_comp):
            H_pca = toth(H_pca)
            H_aa = torch.einsum("bi,ij->bj", [H_pca, H_comp])
            if not self.flat_hand_mean:
                H_aa = H_aa + H_mean.type_as(H_aa)
            return H_aa

        lhand_axis = xhand_cvt(
            lhand_pca, self.left_hand_mean, self.left_hand_components
        )
        rhand_axis = xhand_cvt(
            rhand_pca, self.right_hand_mean, self.right_hand_components
        )
        return lhand_axis, rhand_axis

    """
    axis -> PCA
    """

    def hand_axis_to_pca(self, lhand_axis, rhand_axis):
        """
        lhand_axis: T,45 or T,15,3
        """

        def xhand_cvt(H_aa, H_mean, H_inv_comp):
            H_aa = toth(H_aa).reshape(-1, 45)
            if not self.flat_hand_mean:
                H_aa = H_aa - H_mean.type_as(H_aa)
            H_pca = torch.einsum("bi,ij->bj", [H_aa, H_inv_comp])
            return H_pca

        lhand_pca = xhand_cvt(lhand_axis, self.left_hand_mean, self.Lh_inv_comp)
        rhand_pca = xhand_cvt(rhand_axis, self.right_hand_mean, self.Rh_inv_comp)
        return lhand_pca, rhand_pca

    """
    setup vars
    """
    # setattr(model, 'left_hand_components', toth(model.np_left_hand_components))
    # setattr(model, 'right_hand_components', toth(model.np_right_hand_components))

    if not hasattr(model, "hand_axis_to_pca"):
        setattr(model, "hand_axis_to_pca", hand_axis_to_pca)
    if not hasattr(model, "hand_pca_to_axis"):
        setattr(model, "hand_pca_to_axis", hand_pca_to_axis)
    Lh_inv_comp = torch.linalg.pinv(model.left_hand_components)
    Rh_inv_comp = torch.linalg.pinv(model.right_hand_components)
    setattr(model, "Lh_inv_comp", Lh_inv_comp)
    setattr(model, "Rh_inv_comp", Rh_inv_comp)
