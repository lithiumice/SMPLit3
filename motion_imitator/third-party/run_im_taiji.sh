# <=======================retrain low-level policy with full motion lib
bash /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party/setup_taiji_isaac_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
cd /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party
/apdcephfs/private_wallyliang/PLANT/take_taiji.sh python -m embodied_pose.run_im --cfg im_pretrain_amass \
--motion_file /apdcephfs/share_1330077/wallyliang/vid2player_data/motion_lib/amass_aug \
--results_dir /apdcephfs/share_1330077/wallyliang/vid2player_data_amass_low_policy_aug \
--rl_device cuda:0 \
--num_envs 4096 \
--headless

# <=======================retrain low-level policy with full motion lib, BIG network
python /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party/embodied_pose/amass_to_motion_lib.py
cd /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party
/apdcephfs/private_wallyliang/PLANT/take_taiji.sh python -m embodied_pose.run_im --cfg im_pretrain_amass_big \
--motion_file /apdcephfs/share_1330077/wallyliang/vid2player_data/motion_lib/amass_aug \
--results_dir /apdcephfs/share_1330077/wallyliang/vid2player_data_amass_low_policy_aug_big_re_re_re \
--rl_device cuda:0 \
--num_envs 4096 \
--headless


# # <=================== for demo
/apdcephfs/private_wallyliang/PLANT/take_taiji.sh bash /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party/finetune_rollout_imitation_wo_rfc.sh \
"/apdcephfs/private_wallyliang/eval_emdb_diftraj_for_imitation_iter100" \
"/apdcephfs/private_wallyliang/eval_emdb_diftraj_for_imitation_iter100_process"

# # <=================== for p-GT data
# temporly use RFC to save time, >_<
/apdcephfs/private_wallyliang/PLANT/take_taiji.sh bash /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party/finetune_rollout_imitation.sh \
"/apdcephfs/share_1330077/wallyliang/walk_wham_0227" \
"/apdcephfs/share_1330077/wallyliang/walk_wham_0227_process"


# # <=================== for teaser
/apdcephfs/private_wallyliang/PLANT/take_taiji.sh bash /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party/finetune_rollout_imitation_wo_rfc.sh \
"/apdcephfs/private_wallyliang/origin_videos/100_ways_walk_youtube_split_wham_npzs" \
"/apdcephfs/private_wallyliang/origin_videos/100_ways_walk_youtube_split_wham_npzs_process"

bash /root/apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party/finetune_npz_w_rfc.sh \
/root/apdcephfs/private_wallyliang/origin_videos/taijiquan_female/ID0_difTraj_raw.npz \
/root/apdcephfs/private_wallyliang/origin_videos/taijiquan_female_process2

# finetune on individual npz
for npz_file in /root/apdcephfs/private_wallyliang/eval_emdb_diftraj_for_imitation_iter100/*.npz; do
baseName=$(basename $npz_file)
baseName=${baseName/.npz/}
echo "baseName:"$baseName
/apdcephfs/private_wallyliang/PLANT/take_taiji.sh bash /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party/finetune_npz_no_rfc.sh \
"$npz_file" \
"/apdcephfs/private_wallyliang/origin_videos/eval_emdb_imitation/${baseName}_process"
done




# utils
cp_dir=/root/apdcephfs/private_wallyliang/origin_videos/100_ways_walk_youtube_split_wham_npzs
mkdir -p $cp_dir
for npz_path in $(ls -d /root/apdcephfs/private_wallyliang/origin_videos/100_ways_walk_youtube_split_wham/*/*.npz); do
    dirname=$(basename $(dirname $npz_path))
    base_name=$(basename $npz_path)
    cp $npz_path $cp_dir/${dirname}_${base_name}
done
