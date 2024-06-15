
pip install matplotlib==3.1.3 --index-url https://mirrors.tencent.com/pypi/simple/

# # # FGD AE, window size 24
# cd /root/apdcephfs/private_wallyliang/PLANT/text-to-motion
# take_taiji python train_style_ae.py \
# --name Style_Decomp_SP001_SM001_H512_1 \
# --gpu_id 0 --window_size 24 \
# --batch_size 4096 --eval_every_e 10 --lr 1e-3 \
# --in_type style100 \
# --ae_type ae3

# # FID AE triplet loss, window size 24
# cd /root/apdcephfs/private_wallyliang/PLANT/text-to-motion
# take_taiji python train_style_fitData_vae.py \
# --name style100_triplet_win24_FID_AE \
# --gpu_id 0 --window_size 24 \
# --batch_size 1024 --eval_every_e 10 --lr 1e-3 \
# --in_type style100 \
# --ae_type ae3 \
# --ex_fps 20 --use_triplet_loss

# # # FID, VAE with style predict, window size 24
# cd /root/apdcephfs/private_wallyliang/PLANT/text-to-motion
# take_taiji python train_style_fitData_vae.py \
# --name Style_Decomp_SP001_SM001_H512_tripletLoss_stylePred_vae \
# --gpu_id 0 --window_size 24 \
# --batch_size 1024 --eval_every_e 10 --lr 1e-3 \
# --use_vae \
# --in_type style100 \
# --ae_type ae3

# # VAE with Fitting data and 100 style data, win 24
# cd /root/apdcephfs/private_wallyliang/PLANT/text-to-motion
# take_taiji python train_style_fitData_vae.py \
# --name FitData_Style100_VAE_windowSize24 \
# --gpu_id 0 --window_size 24 \
# --batch_size 1024 \
# --eval_every_e 10 --lr 1e-3 \
# --use_vae \
# --in_type style100_and_fitData \
# --ae_type aeLarge

# cd /root/apdcephfs/private_wallyliang/PLANT/text-to-motion
# take_taiji python train_style_fitData_vae.py \
# --name Style100_VAE_win60_aeLarge_fps30 \
# --gpu_id 0 --window_size 60 \
# --batch_size 4096 \
# --eval_every_e 1 --lr 1e-3 \
# --use_vae --in_type style100 \
# --ae_type aeLarge --ex_fps 30

# FID AE triplet loss, fps 30, 2s, win 60， 100 style data
cd /root/apdcephfs/private_wallyliang/PLANT/text-to-motion
take_taiji python train_style_fitData_vae.py \
--name style100_AE_FID_triplet_win60_fps30 \
--gpu_id 0 --window_size 60 \
--batch_size 1024 --eval_every_e 10 --lr 1e-3 \
--in_type 100Styles \
--ae_type ae3 \
--ex_fps 30 \
--use_triplet_loss

# FMD AE, fps 30, 2s, win 60， 100 style data
cd /root/apdcephfs/private_wallyliang/PLANT/text-to-motion
take_taiji python train_style_fitData_vae.py \
--name style100_AE_FMD_win60_fps30 \
--gpu_id 0 --window_size 60 \
--batch_size 1024 --eval_every_e 10 --lr 1e-3 \
--in_type 100Styles \
--ae_type ae3 \
--ex_fps 30 

# FMD AE, fps 30, 2s, win 60， 100 style data + fitting data
cd /root/apdcephfs/private_wallyliang/PLANT/text-to-motion
take_taiji python train_style_fitData_vae.py \
--name style100_fitData_AE_FMD_win60_fps30 \
--gpu_id 0 --window_size 60 \
--batch_size 1024 --eval_every_e 10 --lr 1e-3 \
--in_type whamFitData_100Styles \
--ae_type ae3 \
--ex_fps 30 



# <=================win 30, fps 30
# FID AE triplet loss, fps 30, 2s, win 32, 100 style data
cd /root/apdcephfs/private_wallyliang/PLANT/text-to-motion
take_taiji python train_style_fitData_vae.py \
--name style100_AE_FID_triplet_win32_fps30 \
--gpu_id 0 --window_size 32 \
--batch_size 1024 --eval_every_e 10 --lr 1e-3 \
--in_type 100Styles \
--ae_type ae3 \
--ex_fps 30 \
--use_triplet_loss


# train sytle inception classifier
cd /root/apdcephfs/private_wallyliang/PLANT/text-to-motion
python train_style_classifer.py \
--gpu_id 0 --window_size 32 \
--batch_size 10240 --lr 1e-3 \
--in_type 100Styles \
--ex_fps 30 \
--log_interval 30


# FID AE triplet loss, fps 32, 2s, win 24, 100 style data
cd /root/apdcephfs/private_wallyliang/PLANT/text-to-motion
take_taiji python train_style_fitData_vae.py \
--name style100_AE_FID_triplet_win24_fps30 \
--gpu_id 0 --window_size 24 \
--batch_size 1024 --eval_every_e 10 --lr 1e-3 \
--in_type 100Styles \
--ae_type ae3 \
--ex_fps 30 \
--use_triplet_loss
