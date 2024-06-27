conda activate smplit
export PYOPENGL_PLATFORM=osmesa
cd /is/cluster/fast/hyi/workspace/VidGen/SMPLit3/

python infer.py \
--visualize \
--save_pkl \
--use_smplx \
--use_smirk \
--use_acr \
--vitpose_model "body_only" \
--output_pth output/boy_slickback_3s4 \
--video assets/boy_slickback_3s.mp4


python infer.py \
--visualize \
--save_pkl \
--use_smplx \
--use_acr \
--vitpose_model "body_only" \
--output_pth output/test2 \
--video assets/A1E+sxSfPQS.mp4



python infer.py \
--run_smplify \
--visualize \
--save_pkl \
--use_smplx \
--use_hmr2 \
--use_smirk \
--use_acr \
--vis_kpts \
--vitpose_model "mmpose" \
--output_pth output/test1 \
--video assets/A1E+sxSfPQS.mp4


python infer.py \
--run_smplify \
--visualize \
--save_pkl \
--use_smplx \
--use_hmr2 \
--use_smirk \
--use_acr \
--vis_kpts \
--vitpose_model "mmpose" \
--output_pth output/boy_slickback_3s2 \
--video assets/boy_slickback_3s.mp4


python infer.py \
--run_smplify \
--visualize \
--save_pkl \
--use_smplx \
--use_hmr2 \
--use_smirk \
--use_acr \
--vis_kpts \
--output_pth output/boy_slickback_3s2 \
--video assets/boy_slickback_3s.mp4


python infer.py \
--run_smplify \
--visualize \
--save_pkl \
--use_hmr2 \
--use_smplx \
--use_smirk \
--use_acr \
--vitpose_model ViTPose+-B \
--output_pth output/boy_slickback_3s \
--video assets/boy_slickback_3s.mp4



