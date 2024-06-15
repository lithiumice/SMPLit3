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
--save_interval 25000 \
--diffusion_steps 1000 \
--seq_len 200 \
--add_emb

python -m sample.generate_traj \
--model_path exps/difftraj_amass_0615 \
--train_data_path data/amass_test.jpkl \
--load_mean_path data/amass_mean_std_train.npz \
--dataset difftraj \
--vis_save_dir output