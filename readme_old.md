
# Make SMPLX from anyone and any video

This repo contrain 4 component of my works: fitting method, trajectory predictor, motion imitation, locomotion generation.

# Prepare ENV
## Installation
Please see [Installation](INSTALL.md) for details.

## Download models

download models zip from `https://www.dropbox.com/scl/fi/uu87dq9g3wj7v0vejbu8m/models.tar.gz?rlkey=a6me2oodq9v9z7vq3oqoaxir6&st=knu4fgr7&dl=0` and unpack it. rename it to model_file. or link a symlink `ln -s path_to_model ./model_files`

```bash
pip install imgaug --index-url https://mirrors.tencent.com/pypi/simple/
```

```bash
conda activate smplit
```

# DiffTraj

A trajectory predictor.

## prepare data

```bash
cd difftraj
python process_dataset.py --in_type amassData
```

after process, we get amassDate_train.jpkl and amassData_mean_std.npz

## Train 

```bash
python -m train.train_difftraj_diffpose \
--save_dir ./exps/DiffTraj_models/difftraj_amass_100styles_seq128_step1000_addEmb_allAMASS \
--train_data_path amassDate_train.jpkl \
--load_mean_path amassData_mean_std.npz \
--dataset difftraj \
--arch trans_enc \
--batch_size 218 \
--log_interval 5 \
--save_interval 25000 \
--diffusion_steps 1000 \
--seq_len 200 --add_emb
```

## Inference

```bash
python -m sample.generate_traj \
--model_path ./exps/DiffTraj_models/difftraj_amass_100styles_seq128_step1000_addEmb_amassOnly/model000605280.pt \
--train_data_path amassDate_noOcclusion_train.jpkl \
--load_mean_path amassData_noOcclusion_mean_std.npz \
--dataset difftraj \
--vis_save_dir ./output/difftraj_vis
```

## visualize evaluation curve

```bash
python plot_eval_curve.py
```


# Video MoCap Pipeline

run smplx estimator along with difftraj

as for whole body case, we should use body pose form WHAM directly without optimization.

```bash
python infer.py \
--visualize \
--save_pkl \
--est_hands \
--not_full_body \
--use_smplx \
--use_smirk \
--use_acr \
--output_pth $OUT_DIR \
--video $VID_PATH \
FLIP_EVAL True
```

we use HMR2 in uppper body only case, and run smplifyx optimization:

```bash
python infer.py \
--run_smplify \
--visualize \
--save_pkl \
--est_hands \
--not_full_body \
--use_hmr2 \
--use_smplx \
--use_smirk \
--use_acr \
--output_pth $OUT_DIR \
--video $VID_PATH \
FLIP_EVAL True
```

the final result are store in a single npz for one person.

you can import it in blender with smplx-blender-addon.

# locomotion generation of LOCO

## Train

### Baseline

```bash
# step 8, onehot style control
python -m train.train_taming_style \
--save_dir ./exps/loco_generation/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_100Styles_30fps \
--in_type pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_100Styles_30fps \
--eval_during_training \
--batch_size 512 \
--log_interval 5 \
--save_interval 25000 \
--diffusion_steps 8 
```

### Pretrain

pretrain with fitting data.

```bash
python -m train.train_cmdm_style \
--save_dir ./exps/loco_generation/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_cmdm_whamFitData_30fps \
--in_type pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_cmdm_whamFitData_30fps \
--eval_during_training \
--batch_size 512 \
--log_interval 5 \
--save_interval 25000 \
--diffusion_steps 8 \
--train_cmdm_base
```

### Finetune

```bash
python -m train.train_cmdm_style \
--base_model ./exps/loco_generation/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_cmdm_whamFitData_30fps/model000250000.pt \
--save_dir ./exps/loco_generation/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_cmdm_100Styles_30fps_finetune2 \
--in_type pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_cmdm_100Styles_30fps_finetune2 \
--eval_during_training \
--batch_size 512 \
--log_interval 5 \
--save_interval 25000 \
--diffusion_steps 8 
```



## Inference

```bash
# visualize bast finetune model
python -m sample.generate_style \
--model_path ./exps/loco_generation/pose_joints_onehot_simpleCtrl_diffusionSteps8_pastMotion_pastMotionLen15_egoTraj_cmdm_100Styles_30fps_finetune2/model000602605.pt \
--user_transl \
--difine_traj circle \
--traj_rev_dir 0 \
--traj_face_type 0 \
--infer_step 3 \
--num_samples 1000 \
--gen_time_length 8 \
--eval_cmdm_finetune \
--infer_walk_speed 0.8 \
--vis_save_dir ./output/loco_gen_infer
```

## Acknowledgement
The video mocap pipeline is mainly build upon WHAM and many other works.

## Contact
Please contact fthualinliang@scut.edu.cn for any questions related to this work.
