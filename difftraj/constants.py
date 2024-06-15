import numpy as np
import os

main_code_path = os.path.join(os.path.dirname(__file__), "..")
model_files_path = os.path.join(os.path.dirname(__file__), "../model_files")
third_party_path = os.path.join(os.path.dirname(__file__), "../third-party")


# path for smplx models
male_bm_path = f"{model_files_path}/uhc_data/smpl/SMPLH_MALE.npz"
male_dmpl_path = f"{model_files_path}/uhc_data/dmpls/male/model.npz"
female_bm_path = f"{model_files_path}/uhc_data/smpl/SMPLH_FEMALE.npz"
female_dmpl_path = f"{model_files_path}/uhc_data/dmpls/female/model.npz"

# for joints convention
l_idx1, l_idx2 = 5, 8
r_hip, l_hip, sdr_r, sdr_l = face_joint_indx = [2, 1, 17, 16]
smplh52_to_smpl24 = np.array(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        37,
    ]
)
