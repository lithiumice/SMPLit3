# sudo guestmount --add /media/lithiumice/WIN10/Users/lithiumice/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu20.04LTS_79rhkp1fndgsc/LocalState/ext4.vhdx -i --rw /mnt/vhdxdrive
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# pip install numpy==1.23.0
# pip uninstall torch torchvision torchaudio
# python -c "import torch;torch.ones(1).cuda()"

conda activate isaac
cd /home/lithiumice/PHC
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/lithiumice/miniconda3/envs/isaac/lib

python -c "import torch; torch.cuda.empty_cache()"

####################
# in local
IN_NPZ_PATH=/home/lithiumice/male_walk_split_vids_clip0001_dynamic000_test_dynCam_incTraj_noJointsAlign_diffusion_0_288_ID1.npz
python scripts/data_process/convert_data_smpl.py $IN_NPZ_PATH /tmp/ref_motion_imitaion.pkl
python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_kp_mcp_iccv.yaml \
--cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file /tmp/ref_motion_imitaion.pkl \
--network_path output/phc_kp_mcp_iccv \
--test --num_envs 1 --epoch -1 --render_o3d --no_virtual_display

# data process
IN_NPZ_PATH=/home/lithiumice/male_WuDangQuan_test_dynCam_incTraj_noJointsAlign_diffusion_0_1069_ID1.npz
python scripts/data_process/convert_data_smpl.py $IN_NPZ_PATH sample_data/wudang2.pkl

# train prim
python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_prim_iccv.yaml \
--cfg_train phc/data/cfg/train/rlg/im.yaml --motion_file sample_data/wudang2.pkl \
--network_path output/phc_prim_iccv --no_log --debug \
--headless --has_eval

# test prim
python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_prim_iccv.yaml \
--cfg_train phc/data/cfg/train/rlg/im.yaml --motion_file sample_data/wudang2.pkl \
--network_path output/phc_prim_iccv \
--test --num_envs 1 --epoch -1 --render_o3d --no_virtual_display

# train composer
python phc/run.py --task HumanoidImMCPGetup \
--cfg_env phc/data/cfg/phc_kp_mcp_iccv.yaml \
--cfg_train phc/data/cfg/train/rlg/im_mcp.yaml \
--motion_file sample_data/wudang2.pkl \
--network_path output/phc_kp_mcp_iccv \
--has_eval --headless --no_log --debug

# test composer
python phc/run.py --task HumanoidImMCPGetup \
--cfg_env phc/data/cfg/phc_kp_mcp_iccv.yaml \
--cfg_train phc/data/cfg/train/rlg/im_mcp.yaml \
--motion_file sample_data/wudang2.pkl \
--network_path output/phc_kp_mcp_iccv \
--test --num_envs 1 --epoch -1 --no_virtual_display


####################
# in taiji
conda activate isaac
cd /root/apdcephfs/private_wallyliang/PHC
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/miniconda3/envs/isaac/lib
alias python="take_taiji python3"

python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_prim_iccv.yaml \
--cfg_train phc/data/cfg/train/rlg/im.yaml --motion_file /tmp/ref_motion_imitaion.pkl \
--network_path output/phc_prim_iccv --headless --has_eval --no_log
