# Motion Capture 2 / SMPLit 

Some important modification:
- useing abolute path, can run from anywhere.
- unified interface with `Motion Capture` repo.
- You must activate conda environment before using docker container.
- all large model files are stored in LFS, make sure you have pull all lfs files before running.

## Docker

### 如何制作docker

首先，参考[Installation](INSTALL.md)来安装python环境。

或者直接使用配置好的docker镜像

```bash
# pull from docker hub
docker pull mirrors.tencent.com/wallydocker/smplit_tlinux:v0.0.1
```

提交taiji任务参考的json配置：
```json
{
    "Token":"xxx",  
    "business_flag": "AILab_DHD",   
    "image_full_name": "mirrors.tencent.com/wallydocker/smplerx_tlinux:v0.0.2",       
    "host_num": 1,             
    "host_gpu_num": 1,         
    "is_resource_waiting": true,  
    "model_local_file_path": "/apdcephfs/private_wallyliang/SMPLit",
    "exec_start_in_all_mpi_pods": true,
    "start_cmd": "python3 occupy_gpu/occupy_gpu.py", 
    "GPUName": "V100",
    "cuda_version": "11.0",
    "readable_name": "SMPLit_all_tlinux",
    "task_flag": "SMPLit_all_tlinux"
}
```
1. 请将Token替换为你自己的taiji Token。
2. 将`/apdcephfs/private_wallyliang/SMPLit`替换为你自己的private盘绝对路径
3. 注意`cuda_version`必须为“11.0”，因为安装的dpvo编译时依赖指定的cuda版本。

## models wieghts

该git下面直接使用LFS储存了model权重等大文件。或者可以从`https://www.dropbox.com/scl/fi/uu87dq9g3wj7v0vejbu8m/models.tar.gz?rlkey=a6me2oodq9v9z7vq3oqoaxir6&st=knu4fgr7&dl=0`下载，并使用
`ln -s path_to_model model_files`解压到SMPLit文件夹下面的model_files文件夹。

taiji容器无法联网，导致下载huggingface的model的时候报错。将文件夹复制到你能访问的位置然后软连接到~/.cache
`ln -s /apdcephfs_cq10/share_301996436/share_data/.cache ~/.cache`

## Inference 

在使用docker前必须使用conda激活指定的环境。
Important! You have to activate conda environment before runing inference:


```bash
source ~/miniconda3/bin/activate
conda activate smplit

# for rendering on server
export PYOPENGL_PLATFORM=osmesa
# if you met hmr home error
# export HOME=/root 

# use this to enable network on devcloud
export https_proxy=
export http_proxy=
```

Then run:

```bash
# Please refer to testbench.py
# or run:
# python3 testbench_v100.py
# python3 testbench.py

from main.mocap import init_model, mocap

input_video_path = 'data/test.mp4'
output_smplx_path = 'output/test.npz'

model = init_model()
mocap(
    model,
    input_video_path,
    output_smplx_path,
)
```

# Video MoCap
这部分的代码是基于WHAM的，并整合了ACR、HMR2、等body和face的初始化方法，修改成了whole-body版本（使用SMPL-X）。
同时添加了DiffTraj（可见下节的介绍）来得到Global Trajectory。
运行这部分有几个setting，在不同的需求和情况下使用：

1. 直接使用WHAM的body pose、ACR的hands、smrik的face作为初始化，并跳过优化fitting，以节省耗时。
·$VID_PATH· 是输入的视频文件路径。
·$OUT_DIR· 指输出保存文件的目标文件夹。

```bash
python infer.py \
--visualize \
--save_pkl \
--est_hands \
--not_full_body \
--use_smplx \
--use_smirk \ # smrik的face
--use_acr \ # ACR的hands
--output_pth $OUT_DIR \
--video $VID_PATH \
FLIP_EVAL True
```

2. 对于半身的case，WHAM会有比较多的问题，可以使用HMR2的body pose来替换WHAM，同时运行smplifyx的optimization。
```bash
python infer.py \
--run_smplify \ # 运行smplifyx的optimization
--visualize \
--save_pkl \
--est_hands \
--not_full_body \
--use_hmr2 \ # 使用HMR2的body pose
--use_smplx \
--use_smirk \
--use_acr \
--output_pth $OUT_DIR \
--video $VID_PATH \
FLIP_EVAL True
```

the final result are store in a single npz for one person.
you can import it in blender with smplx-blender-addon.

# DiffTraj
## Train 

主要是DiffTraj的训练。DiffTraj是一个Motion Prior，用于从local body pose得到全局的运动轨迹。
以下操作都在SMPLit/difftraj文件夹下面进行。

```bash
cd /apdcephfs/private_wallyliang/SMPLit/difftraj
```

### 处理数据

AMASS数据位置: /apdcephfs_cq10/share_301996436/share_data/amass_raw

DiffTraj的训练只需要AMASS。（之前有尝试过加入100STYLES的数据进去训练，但是并没有得到提升。）

#### AMASS

先将所有的amass的npz的路径缓存到txt文件：

```bash
mkdir -p data
AMASS_PATH=/apdcephfs_cq10/share_301996436/share_data/amass_raw
find $AMASS_PATH -type f -name "*.npz" > data/amass_npzs_list.txt
# `cat data/amass_npzs_list.txt | wc -l` 可见有15805个npz。

# 分割train和test：
SOURCE=data/amass_npzs_list.txt
total_lines=$(wc -l $SOURCE | awk '{print $1}') # 计算文件的总行数
train_lines=$((total_lines * 8 / 10)) # 计算训练集的行数（80%）
head -n $train_lines $SOURCE > data/amass_train.txt # 使用head和tail命令分割文件
tail -n +$(($train_lines + 1)) $SOURCE > data/amass_test.txt

# `data/amass_train.txt`有1.2w个npz。

# 处理train：
# for train
python process_dataset.py \
--save_train_mean_std data/amass_mean_std_train.npz \
--save_train_jpkl data/amass_train.jpkl \
--in_npz_list data/amass_train.txt \
--in_data_type amass \
--save_data_fps 30 \
--win_size 35 # 35 是为了兼容后面的diffgen，不需要重新处理一遍

# for test
python process_dataset.py \
--save_train_mean_std data/amass_mean_std_test.npz \
--save_train_jpkl data/amass_test.jpkl \
--in_npz_list data/amass_test.txt \
--in_data_type amass \
--save_data_fps 30 \
--win_size 35 \
--max_process_npz_num 30
```

`data/amass_train.txt`有效的clips有8988个（win_size=35）。
经过dataloder过滤win_size>=200后剩下5406个。

处理完成会得到data目录下的amassDate_train.jpkl，这个文件会被用于训练的dataloader中加载。
`amass_mean_std.npz`是用于数据的归一化，注意这个文件在训练和推理都用到，并需要保持一致，如果不一致会导致归一化出问题，出现比如滑步之类的问题。

#### 100styles
```bash
mkdir -p data
# DATA_PATH=/apdcephfs/share_1290939/shaolihuang/wallyliang_backup_files/PLANT_data/100styles_merge
DATA_PATH=/apdcephfs_cq10/share_301996436/share_data/100styles_merge
find $DATA_PATH -type f -name "*.npz" > data/100styles_npzs_list.txt

SOURCE=data/100styles_npzs_list.txt
total_lines=$(wc -l $SOURCE | awk '{print $1}') 
train_lines=$((total_lines * 8 / 10)) 
head -n $train_lines $SOURCE > data/100styles_train.txt 
tail -n +$(($train_lines + 1)) $SOURCE > data/100styles_test.txt

# for train
python process_dataset.py \
--save_train_mean_std data/100styles_mean_std_train.npz \
--save_train_jpkl data/100styles_train.jpkl \
--in_npz_list data/100styles_train.txt \
--in_data_type 100styles \
--save_data_fps 30 \
--win_size 35 
# we get 648 items for train.

# for test
python process_dataset.py \
--save_train_mean_std data/100styles_mean_std_test.npz \
--save_train_jpkl data/100styles_test.jpkl \
--in_npz_list data/100styles_test.txt \
--in_data_type 100styles \
--save_data_fps 30 \
--win_size 35 \
--max_process_npz_num 30
```


### 开始训练
使用论文中的setting，目前只用到了单卡。
```bash
python -m train.train_main \
--save_dir exps/difftraj_amass \
--train_data_path data/amass_train.jpkl \
--load_mean_path data/amass_mean_std_train.npz \
--dataset difftraj \
--arch trans_enc \
--batch_size 218 \
--log_interval 5 \
--save_interval 5000 \
--diffusion_steps 1000 \ # paper setting
--seq_len 200 \
--add_emb
```
在A100上训大概只占用10GB左右的显存。
一般训到10w steps就有可以用的效果了。

```bash
python plot_eval_curve.py
```
使用以上命令可视化训练evaluation曲线。
这个脚本仅供参考，你得自己改里面的路径。

### 推理
将model_path替换为训好的pt文件路径。
```bash
python -m sample.generate_traj \
--model_path exps/difftraj_amass/model000605280.pt \
--train_data_path data/amass_test.jpkl \
--load_mean_path data/amass_mean_std_train.npz \
--dataset difftraj \
--vis_save_dir output
```

这里提供了一个包装好的使用difftraj的接口，可以直接输入任意从 其他HPS方法得到的body pose，然后difftraj输出带有全局轨迹的motion。
```bash
python3 difftraj_entry.py \
--inp assets/diftraj_demo/beat2_our_rot.npz \
--flip_x --rot_deg -90

# 批量运行所有的npz
find assets/diftraj_demo -name *.npz -print -exec sh -c \
'python difftraj_entry.py --inp "$1" --flip_y' _ {} \;
```

# LOCO
## 训练
这部分的motion  generation使用和DiffTraj一套代码，本质上都是diffusion，只有condition和output是不一样而已。

在100 styles data上训练Motion generation
```bash
# onehot style control
python -m train.train_main \
--save_dir exps/loco_generation_exp_100styles \
--train_data_path data/100styles_train.jpkl \
--load_mean_path data/100styles_mean_std_train.npz \
--dataset diffgen \
--batch_size 128 \
--log_interval 10 \
--save_interval 5000 \
--diffusion_steps 8 \
--p_len 10 \
--f_len 30 \
--train_fps 30 \
--use_onehot_style \
--use_past_motion
# --debug # add this if you want to debug
```

在scrape data上训练Motion generation
```bash
python -m train.train_main \
--save_dir exps/loco_generation_fit_data \
--eval_during_training \
--batch_size 512 \
--log_interval 5 \
--save_interval 25000 \
--diffusion_steps 8 \
--train_cmdm_base
```

如何finetune[not test]
```bash
# --base_model: 继承的base model weight
python -m train.train_main \
--base_model exps/loco_generation_fit_data/model000250000.pt \ 
--save_dir exps/loco_generation_exp3 \
--eval_during_training \
--batch_size 512 \
--log_interval 5 \
--save_interval 25000 \
--diffusion_steps 8 
```

Autoregresssive Motion generation推理
```bash
# visualize bast finetune model
# --difine_traj: 参考infer代码，指定多种预设的轨迹
python -m sample.generate_style \
--model_path exps/loco_generation_exp_100styles \
--user_transl \
--difine_traj circle \
--traj_rev_dir 0 \
--traj_face_type 0 \
--infer_step 3 \
--num_samples 10 \
--gen_time_length 8 \
--infer_walk_speed 0.8 \
--vis_save_dir output/loco_gen_infer

# inference finetuned model
python -m sample.generate_style \
--model_path exps/loco_generation_exp_100styles \
--user_transl \
--difine_traj circle \
--traj_rev_dir 0 \
--traj_face_type 0 \
--infer_step 3 \
--num_samples 10 \
--gen_time_length 8 \
--eval_cmdm_finetune \
--infer_walk_speed 0.8 \
--vis_save_dir output/loco_gen_infer
```

Evaluation:
```bash
python -m eval.eval_style100 \
--model_path exps/loco_generation_fit_data/model000602605.pt \
--dataset diffgen \
--batchsize 3200
```

# 训练WHAM

我验证过可以训练的代码放在了/apdcephfs/share_1290939/shaolihuang/wallyliang_backup_files/WHAM_train下面

按readme准备好data下面的所有文件，包括预处理的amass pt文件。

train stage-1
```bash
python train.py --cfg configs/yamls/stage1.yaml
```

train stage-2
```bash
take_taiji python train.py --cfg configs/yamls/stage2.yaml \
TRAIN.CHECKPOINT "/apdcephfs/private_wallyliang/WHAM_new/experiments/train_stage1/checkpoint.pth.tar"
/apdcephfs/private_wallyliang/WHAM_new/experiments/train_stage1/checkpoint.pth.tar
```

### Issues

- fix imageio ffmpeg error on CentOS like TencentOS:

```bash
pip install -U imageio-ffmpeg
```

- 在taiji上安装pip包：
```bash
pip install imgaug --index-url https://mirrors.tencent.com/pypi/simple/
```

- Issue: "ax.lines = []; AttributeError: can't set attribute"
Solution:
```bash
pip install matplotlib==3.1.3 --index-url https://mirrors.tencent.com/pypi/simple/
# 如果用以上的命令安装的时候没有预编译包而需要编译导致报错，则使用一下命令：
pip install --only-binary :all: "matplotlib<=3.4.3"
```

## Contact
Please contact fthualinliang@scut.edu.cn for any questions related to this work.
