
# test, no finetune, use RFC, convert SMPL axis to AMASS axis
cd /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
MODEL_CKPT=/apdcephfs/share_1330077/wallyliang/vid2player_data_amass_low_policy_aug/models/Humanoid_epoch10000.pth
# NPZ_MOTOIN_FILE=/root/apdcephfs/private_wallyliang/debug_beat/t2_wavlm_60W_beats_1speaker_didffges_seed1.npz
for NPZ_MOTOIN_FILE in /apdcephfs/private_wallyliang/debug_beat_pose2Traj/*.npz; do
npz_save_path=${NPZ_MOTOIN_FILE/.npz/_imitation.npz}
/apdcephfs/private_wallyliang/PLANT/take_taiji.sh python -m embodied_pose.run_im --cfg im_finetune --rl_device cuda:0 \
--motion_file $NPZ_MOTOIN_FILE --checkpoint $MODEL_CKPT \
--test --num_envs 1 --record_npz --record_npz_path $npz_save_path \
--episode_length 3000000000 
# --cvt_npz_to_amass
done
