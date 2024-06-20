
# sqlite
cd /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party
sqlite_db_path=/apdcephfs/private_wallyliang/PLANT_dbs/npzs_to_imitate.db
save_npz_root_path=/apdcephfs/share_1330077/wallyliang/walk_wham_0227_process/imitated
model_ckpt=/apdcephfs/share_1330077/wallyliang/walk_wham_0227_process/finetune_policy/models/human_latest.pth
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
while true; do
    max_attempts=15
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if [ -n "$3" ]; then
            line=$(sqlite3 $sqlite_db_path "SELECT clip_path FROM redis_table WHERE status='raw' AND clip_path LIKE '%$3%' ORDER BY RANDOM() LIMIT 1")
        else
            line=$(sqlite3 $sqlite_db_path "SELECT clip_path FROM redis_table WHERE status='raw' ORDER BY RANDOM() LIMIT 1")
        fi    
        if [ $? -eq 0 ]; then
            break
        fi
        attempt=$((attempt + 1))
        echo "attempt:"$attempt
        sleep 1
    done

    if [ -z "$line" ]; then
        echo "No raw content found, exiting"
        break
    fi

    npz_motion_file=$line
    echo "NPZ_MOTION_FILE:" $npz_motion_file
    file_name=$(basename $npz_motion_file)
    npz_save_path="$save_npz_root_path/$file_name"
    if [ ! -f $npz_save_path ]; then
        echo "[INFO] start eval with fintune model"
        /apdcephfs/private_wallyliang/PLANT/take_taiji.sh python -m embodied_pose.run_im --cfg im_finetune --rl_device cuda:0 \
        --motion_file $npz_motion_file --checkpoint $model_ckpt \
        --test --num_envs 1 --record_npz --record_npz_path $npz_save_path --episode_length 3000000000
    fi
    
    # break
    sqlite3 $sqlite_db_path "UPDATE redis_table SET status='done' WHERE clip_path='$line'"
done

# bash /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party/batch_run_im_taiji.sh