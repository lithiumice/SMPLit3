# ssh hyi@10.15.1.6

# condor_submit_bid 40 -i -append request_cpus=8 -append request_gpus=1 -append request_memory=102400 -append 'requirements = (CUDADeviceName=="NVIDIA A100-SXM-80GB" || CUDADeviceName=="NVIDIA A100-SXM4-80GB") && UtsnameNodename =!= g149 && UtsnameNodename =!= g179'

conda activate smplit
export PYOPENGL_PLATFORM=osmesa
export HOME=/home/hyi

cd /is/cluster/fast/hyi/workspace/VidGen/SMPLit




OUT_DIR=/is/cluster/fast/hyi/workspace/VidGen/Open-EMO/tmp/smplit_result
VID_PATH=/is/cluster/fast/hyi/workspace/VidGen/Open-EMO/tmp/test_vid_det/syncdet_split/-2z6b5fOqHY_000_0000_000.mp4


python infer.py \
--run_smplify \
--visualize \
--save_pkl \
--not_full_body \
--use_smplx \
--use_hmr2 \
--use_smirk \
--use_acr \
--output_pth $OUT_DIR \
--video $VID_PATH \
FLIP_EVAL True



python infer.py \
--run_smplify \
--visualize \
--save_pkl \
--not_full_body \
--use_hmr2 \
--use_smplx \
--use_smirk \
--use_acr \
--output_pth /is/cluster/work/hyi/ExpressiveBody/estimator_processed/bilibili_30fps \
--video /is/cluster/scratch/hyi/ExpressiveBody/vids_splits/bilibili_30fps/BV1zT4y1m79B+S0000-E0360.mp4



