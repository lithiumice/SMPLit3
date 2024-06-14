conda activate isaac
cd /home/lithiumice/PHC/third-party/
# wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
# tar -xzf mujoco210-linux-x86_64.tar.gz
# mkdir ~/.mujoco
# mv mujoco210 ~/.mujoco/


nm -D /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0|grep tls
sudo ln -sf /usr/lib/x86_64-linux-gnu/libGLX_mesa.so.0 /usr/lib/x86_64-linux-gnu/libOpenGL.so.0


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lithiumice/miniconda3/envs/isaac/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lithiumice/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

NPZ_MOTOIN_FILE=/home/lithiumice/fit_npzs/test_staticCam_incTraj_diffusion_0_273_ID1.npz
SAVE_MODEL_PATH=./results/slickback

NPZ_MOTOIN_FILE=/home/lithiumice/fit_npzs/male_WuDangQuan_test_dynCam_incTraj_noJointsAlign_diffusion_0_1069_ID1.npz
MODEL_CKPT=/home/lithiumice/PHC/third-party/results/im_finetune/models/Djokovic_latest.pth
NPZ_SAVE_PATH=./wudangquan_imitation_result.npz

NPZ_MOTOIN_FILE=/home/lithiumice/fit_npzs/test_staticCam_incTraj_diffusion_0_273_ID1.npz
MODEL_CKPT=/home/lithiumice/PHC/third-party/results/slickback/im_finetune/models/Djokovic_epoch00500.pth
NPZ_SAVE_PATH=./slickback_imitation_result.npz

MODEL_CKPT=/home/lithiumice/PHC/third-party/results/amass_im/models0/Humanoid_latest.pth
NPZ_MOTOIN_FILE=/media/lithiumice/MISC2/putin_ID0_origin.npz
SAVE_MODEL_PATH=./results/putin_ID0_origin
MODEL_CKPT=./results/putin_ID0_origin/models/human_latest.pth
NPZ_SAVE_PATH=/home/lithiumice/PHC/third-party/results/putin_ID0_origin/putin_walk_finetune_rollout.npz

NPZ_MOTOIN_FILE=/media/lithiumice/MISC2/wudang_ID0_origin.npz
SAVE_MODEL_PATH=./results/wudang_ID0_origin
MODEL_CKPT=/home/lithiumice/PHC/third-party/./results/wudang_ID0_origin/models/Djokovic_latest.pth

NPZ_MOTOIN_FILE=/media/lithiumice/MISC1/eval_emdb/eval_emdb_split1_idx1_difftraj.npz 
SAVE_MODEL_PATH=./results/eval_emdb_split1_idx1_difftraj
MODEL_CKPT=/home/lithiumice/PHC/third-party/results/amass_im/models0/Humanoid_latest.pth


# finetune
python embodied_pose/run_im.py --cfg im_finetune --rl_device cuda:0 \
--motion_file $NPZ_MOTOIN_FILE --results_dir $SAVE_MODEL_PATH --headless

# run visualize
python embodied_pose/run_im.py --cfg im_finetune --rl_device cuda:0 \
--motion_file $NPZ_MOTOIN_FILE --checkpoint $MODEL_CKPT --test --num_envs 1 --show_mujoco_viewer

# record npz only
python embodied_pose/run_im.py --cfg im_finetune --rl_device cuda:0 \
--motion_file $NPZ_MOTOIN_FILE --checkpoint $MODEL_CKPT --test --num_envs 1 --record_npz --record_npz_path $NPZ_SAVE_PATH

python vid2player/run.py --cfg federer_train_stage_1 --rl_device cuda:0 --headless


#################################
# preprocess npz motion data
python embodied_pose/npzs_to_motion_lib_pt.py

# train low-level policy
python embodied_pose/run_im.py --cfg amass_im \
--motion_file /home/lithiumice/PHC/third-party/data/motion_lib/eval_emdb \
--results_dir results/eval_emdb_train \
--rl_device cuda:0 \
--headless

# cd /home/lithiumice/PHC/third-party/vid2player
# python motion_vae/train.py


# NPZ_MOTOIN_FILE=/home/lithiumice/fit_npzs/male_WuDangQuan_test_dynCam_incTraj_noJointsAlign_diffusion_0_1069_ID1.npz
# NPZ_MOTOIN_FILE=/home/lithiumice/fit_npzs/trim_and_cvt_video_test_inc_diffusion_0_501_ID1.npz
# NPZ_MOTOIN_FILE=/home/lithiumice/fit_npzs/test_staticCam_incTraj_diffusion_0_273_ID1.npz
