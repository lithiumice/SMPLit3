import os

main_code_path = os.path.join(os.path.dirname(__file__), "../..")

SMPL_KINTREE_PATH = os.path.join(main_code_path, "model_files/kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(main_code_path, "model_files/uhc_data/smpl/SMPL_NEUTRAL.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(main_code_path, "model_files/J_regressor_extra.npy")

ROT_CONVENTION_TO_ROT_NUMBER = {
    "legacy": 23,
    "no_hands": 21,
    "full_hands": 51,
    "mitten_hands": 33,
}

GENDERS = ["neutral", "male", "female"]
NUM_BETAS = 10
