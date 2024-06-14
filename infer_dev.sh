

source ~/miniconda3/bin/activate
conda activate smplit

cd /apdcephfs/private_wallyliang/SMPLit/

export PYOPENGL_PLATFORM=osmesa
export https_proxy=
export http_proxy=

OUT_DIR=/apdcephfs/private_wallyliang/origin_videos/boy_slickback_3s
VID_PATH=/apdcephfs/private_wallyliang/mocap_test_case/boy_slickback_3s.mp4


pip install ffmpeg-python



take_taiji python infer.py \
--visualize \
--save_pkl \
--not_full_body \
--use_smplx \
--use_acr \
--output_pth /apdcephfs/private_wallyliang/videos/jntm+wham_acr3 \
--video /apdcephfs/private_wallyliang/jntm.mp4 \
--vitpose_model ViTPose+-G \
--estimate_local_only \
--debug --vis_kpts


take_taiji python infer.py \
--visualize \
--save_pkl \
--not_full_body \
--use_smplx \
--use_acr \
--output_pth /apdcephfs/private_wallyliang/videos/taijiquan_female_13s+wham_acr3 \
--video /apdcephfs/private_wallyliang/videos/mocap_test_case_30fps/taijiquan_female_13s.mp4 \
--vitpose_model body_only \
--estimate_local_only \
--debug 



# 1. WHAM+ACR
python infer.py \
--visualize \
--save_pkl \
--not_full_body \
--use_smplx \
--use_acr \
--output_pth /apdcephfs/private_wallyliang/videos/boy_slickback_3s+wham_acr \
--video /apdcephfs/private_wallyliang/videos/mocap_test_case_30fps/boy_slickback_3s.mp4 \
--vitpose_model body_only \
--estimate_local_only \
--debug 

python infer.py \
--visualize \
--save_pkl \
--not_full_body \
--use_smplx \
--use_acr \
--output_pth /apdcephfs/private_wallyliang/videos/boy_slickback_3s+wham_acr_stage1 \
--video /apdcephfs/private_wallyliang/videos/mocap_test_case_30fps/boy_slickback_3s.mp4 \
--vitpose_model body_only \
--estimate_local_only \
--debug 

# 2.0. smpler-x+slam+difftraj
python infer.py \
--visualize \
--save_pkl \
--not_full_body \
--use_smplx \
--only_one_person \
--output_pth /apdcephfs/private_wallyliang/origin_videos/boy_slickback_3s+smpler-x+slam+difftraj \
--video $VID_PATH \
--vitpose_model ViTPose+-G \
--inp_body_pose /apdcephfs/private_wallyliang/MotionCaption/output/boy_slickback_3s.npz \
--debug 


# 2. smpler-x+opt
python infer.py \
--visualize \
--run_smplify \
--use_opt_pose_reg \
--save_pkl \
--not_full_body \
--use_smplx \
--only_one_person \
--output_pth /apdcephfs/private_wallyliang/origin_videos/boy_slickback_3s+smplerx_opt \
--video $VID_PATH \
--vitpose_model ViTPose+-G \
--estimate_local_only \
--inp_body_pose /apdcephfs/private_wallyliang/MotionCaption/output/boy_slickback_3s.npz \
--debug 


# 3. wham+acr+opt
python infer.py \
--visualize \
--run_smplify \
--use_opt_pose_reg \
--save_pkl \
--not_full_body \
--use_smplx \
--use_acr \
--output_pth /apdcephfs/private_wallyliang/origin_videos/boy_slickback_3s+wham_acr_opt \
--video $VID_PATH \
--vitpose_model ViTPose+-G \
--estimate_local_only \
--debug 



# 4. hmr2+acr+opt
python infer.py \
--visualize \
--run_smplify \
--save_pkl \
--not_full_body \
--use_smplx \
--use_hmr2 \
--use_smirk \
--use_acr \
--output_pth /apdcephfs/private_wallyliang/origin_videos/boy_slickback_3s_hmr2+acr+opt \
--video $VID_PATH \
--vitpose_model ViTPose+-G \
--debug 


# 4. hmr2+acr+opt
python infer.py \
--visualize \
--save_pkl \
--not_full_body \
--use_smplx \
--use_smplerx \
--output_pth /apdcephfs/private_wallyliang/videos/mocap_test_case_30fps/boy_slickback_3s_hmr2+acr+opt \
--video /apdcephfs/private_wallyliang/videos/mocap_test_case_30fps/boy_slickback_3s.mp4 \
--vitpose_model ViTPose+-G \
--debug 

# body only, no visualize

cd /apdcephfs/private_wallyliang/SMPLit/
python testbench.py

