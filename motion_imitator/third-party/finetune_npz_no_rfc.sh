cd /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party

# setting enriroment variables
# ln -s /apdcephfs /root/apdcephfsc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin


NPZ_MOTOIN_FILE=$1
SAVE_MODEL_PATH=$2
MODEL_CKPT="$SAVE_MODEL_PATH/models/human_latest.pth"

if [ ! -f $MODEL_CKPT ]; then
    # <=======================finetune to remove RFC
    python -m embodied_pose.run_im --cfg im_finetune --rl_device cuda:0 \
    --motion_file $NPZ_MOTOIN_FILE --results_dir $SAVE_MODEL_PATH --headless --rm_rfc_during_finetune --num_envs 40960
fi

file_name=$(basename $NPZ_MOTOIN_FILE)
npz_save_path="$SAVE_MODEL_PATH/$file_name"

# record npz only
python -m embodied_pose.run_im --cfg im_finetune --rl_device cuda:0 \
--motion_file $NPZ_MOTOIN_FILE --checkpoint $MODEL_CKPT \
--test --num_envs 1 --record_npz --record_npz_path $npz_save_path \
--episode_length 3000000000 --rfc_scale 0.001

