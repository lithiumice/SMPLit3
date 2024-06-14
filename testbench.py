
from pathlib import Path
import time
import subprocess
import os

import sys


def run_ffmpeg_subprocess(ffmpeg_cmd):
    if isinstance(ffmpeg_cmd, list):
        ffmpeg_cmd=' '.join(ffmpeg_cmd)
    print(ffmpeg_cmd)
    ret = subprocess.run(ffmpeg_cmd, shell=True)
    return ret
            
vids_list = list((Path('/apdcephfs/private_wallyliang/dance_1')).glob("*.mp4"))
# vids_list = list((Path('/apdcephfs/private_wallyliang/videos')).glob("*.mp4"))


def testbench():
    os.chdir('/apdcephfs/private_wallyliang/SMPLit')
    print(f'{vids_list=}')
    
    for vid_path in vids_list:
        vid_path = str(vid_path)

        # 0. body only
        # seq_out_dir = Path('output', Path(vid_path).stem)
        # cmds = f"""
        # python infer.py \
        # --visualize \
        # --save_pkl \
        # --not_full_body \
        # --use_smplx \
        # --output_pth {seq_out_dir} \
        # --video {vid_path} \
        # --only_one_person \
        # --vitpose_model body_only
        # """

        # 1. WHAM+ACR
        seq_out_dir = Path('output+wham_acr_difftraj', Path(vid_path).stem)
        cmds = f"""
        python infer.py \
        --save_pkl \
        --not_full_body \
        --use_smplx \
        --use_acr \
        --output_pth {seq_out_dir} \
        --video {vid_path} \
        --vitpose_model body_only \
        --estimate_local_only \
        --only_one_person \
        --debug 
        """


        # # 2.0. vis smpler-x
        # seq_out_dir = Path('output+smplerx_opt', Path(vid_path).stem)
        # cmds = f"""
        # python infer.py \
        # --visualize \
        # --save_pkl \
        # --not_full_body \
        # --use_smplx \
        # --output_pth {seq_out_dir} \
        # --video {vid_path} \
        # --vitpose_model ViTPose+-G \
        # --estimate_local_only \
        # --only_one_person \
        # --inp_body_pose /apdcephfs/private_wallyliang/MotionCaption/output/{Path(vid_path).stem}.npz \
        # --debug 
        # """


        # # 2. smpler-x+opt+DiffTraj+vis
        # seq_out_dir = Path('output+smplerx_opt', Path(vid_path).stem)
        # cmds = f"""
        # python infer.py \
        # --run_smplify \
        # --use_opt_pose_reg \
        # --visualize \
        # --save_pkl \
        # --not_full_body \
        # --use_smplx \
        # --output_pth {seq_out_dir} \
        # --video {vid_path} \
        # --vitpose_model ViTPose+-G \
        # --estimate_local_only \
        # --only_one_person \
        # --inp_body_pose /apdcephfs/private_wallyliang/MotionCaption/output/{Path(vid_path).stem}.npz \
        # --debug 
        # """

        # # 3. wham+acr+opt
        # seq_out_dir = Path('output+wham_acr_opt', Path(vid_path).stem)
        # cmds = f"""
        # python infer.py \
        # --visualize \
        # --run_smplify \
        # --use_opt_pose_reg \
        # --save_pkl \
        # --not_full_body \
        # --use_smplx \
        # --use_acr \
        # --output_pth {seq_out_dir} \
        # --video {vid_path} \
        # --vitpose_model ViTPose+-G \
        # --estimate_local_only \
        # --only_one_person \
        # --debug 
        # """


        run_ffmpeg_subprocess(cmds)
    

if __name__ == '__main__':
    testbench()
