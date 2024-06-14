#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

function process_npz() {
    npzs_root_dir=$1
    process_dir=$2

    out_dir="$process_dir/motion_pt"
    results_dir="$process_dir/finetune_policy"
    save_npz_root_path="$process_dir/imitated"
    model_ckpt="$results_dir/models/human_latest.pth"

    mkdir -p $out_dir
    mkdir -p $results_dir
    mkdir -p $save_npz_root_path

    cd /apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party

        # # preprocess npz motion data
    if [ $(find $out_dir -type f | wc -l) -eq 0 ]; then
        echo "<=================================PROCESS NPZS to MOTION PT..."
        python embodied_pose/npzs_to_motion_lib_pt.py \
        --npzs_root_dir $npzs_root_dir \
        --out_dir $out_dir \
        --num_motion_libs 1
    fi

    if [ ! -f $model_ckpt ]; then
        echo "<=================================RUNNING IMITATION FINETUNE..."
        python -m embodied_pose.run_im --cfg im_finetune \
        --motion_file $out_dir \
        --results_dir $results_dir \
        --rl_device cuda:0 \
        --headless --rm_rfc_during_finetune
        # --max_iterations 250
    fi

    npz_list=$(find $npzs_root_dir -name "*.npz" -printf "%s %p\n" | sort -n -r | awk '{print $2}')
    for npz_motion_file in $npz_list; do
        echo "<=================================rolling:" $npz_motion_file
        file_name=$(basename $npz_motion_file)
        npz_save_path="$save_npz_root_path/$file_name"
        echo "[INFO] start eval with fintune model"
        python -m embodied_pose.run_im --cfg im_finetune --rl_device cuda:0 \
        --motion_file $npz_motion_file --checkpoint $model_ckpt \
        --test --num_envs 1 --record_npz --record_npz_path $npz_save_path --episode_length 3000000000 --rfc_scale 0.001
    done
}


# echo $3
# if [ -f $3 ]; then
#     echo "use pretrained model..."
#     model_ckpt=$3
# else
#     echo "use finetune model..."
#     model_ckpt="$results_dir/models/human_latest.pth"
#     # preprocess npz motion data
#     if [ $(find $out_dir -type f | wc -l) -eq 0 ]; then
#         echo "PROCESS NPZS to MOTION PT..."
#         python embodied_pose/npzs_to_motion_lib_pt.py \
#         --npzs_root_dir $npzs_root_dir \
#         --out_dir $out_dir \
#         --num_motion_libs 1
#     fi

#     if [ ! -f $model_ckpt ]; then
#         echo "RUNNING IMITATION FINETUNE..."
#         python -m embodied_pose.run_im --cfg im_finetune \
#         --motion_file $out_dir \
#         --results_dir $results_dir \
#         --rl_device cuda:0 \
#         --headless --max_iterations 250
#     fi
# fi

# npz_list=$(find $npzs_root_dir -name "*.npz" -printf "%s %p\n" | sort -n -r | awk '{print $2}')
# for npz_motion_file in $npz_list; do
#     echo "NPZ_MOTION_FILE:" $npz_motion_file
#     file_name=$(basename $npz_motion_file)
#     npz_save_path="$save_npz_root_path/$file_name"
#     if [ ! -f $npz_save_path ]; then
#         echo "[INFO] start eval with fintune model"
#         python -m embodied_pose.run_im --cfg im_finetune --rl_device cuda:0 \
#         --motion_file $npz_motion_file --checkpoint $model_ckpt \
#         --test --num_envs 1 --record_npz --record_npz_path $npz_save_path --episode_length 3000000000
#     fi
# done


process_npz $1 $2