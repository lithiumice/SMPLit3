cd /apdcephfs/private_wallyliang/SMPLit/difftraj
source ~/miniconda3/bin/activate
conda activate smplit

python -m train.train_difftraj_diffpose \
--save_dir exps/difftraj_amass_0615 \
--train_data_path data/amass_train.jpkl \
--load_mean_path data/amass_mean_std_train.npz \
--dataset difftraj \
--arch trans_enc \
--batch_size 128 \
--log_interval 10 \
--save_interval 5000 \
--diffusion_steps 1000 \
--seq_len 200 \
--add_emb

python -m sample.generate_traj \
--model_path exps/difftraj_amass_0615 \
--train_data_path data/amass_test.jpkl \
--load_mean_path data/amass_mean_std_train.npz \
--dataset difftraj \
--vis_save_dir output



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


--debug


