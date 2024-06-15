import os
import platform
from dataclasses import dataclass
from pathlib import Path
import subprocess
from loguru import logger


def run_openpose(
    openpose_root_path,
    openpose_bin_path,
    img_dir,
    out_dir,
    video_out=None,
    img_out=None,
    low_res=False,
    limit_num=False,
):
    # logger.info(f'run_openpose: {img_dir} -> {out_dir} {video_out} {img_out} {low_res} {limit_num}')
    """
    Runs OpenPose for 2D joint detection on the images in img_dir.
    """

    def make_absolute(rel_paths):
        """Makes a list of relative paths absolute"""
        return [os.path.join(os.getcwd(), rel_path) for rel_path in rel_paths]

    SKELETON = "BODY_25"

    # make all paths absolute to call OP
    openpose_path = make_absolute([openpose_root_path])[0]
    img_dir = make_absolute([img_dir])[0]
    out_dir = make_absolute([out_dir])[0]
    if video_out is not None:
        video_out = make_absolute([video_out])[0]
    if img_out is not None:
        img_out = make_absolute([img_out])[0]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # run open pose
    # must change to openpose dir path to run properly
    og_cwd = os.getcwd()
    os.chdir(openpose_path)

    run_cmds = [
        openpose_bin_path,
        "--image_dir",
        img_dir,
        "--write_json",
        out_dir,
        "--display",
        "0",
        "--model_pose",
        SKELETON,
        "--face",
        "--hand",
    ]
    if limit_num:
        run_cmds += [
            "--number_people_max",
            "1",
        ]
        # run_cmds+=['--number_people_max', str(limit_num),]

    if low_res:
        run_cmds += ["--net_resolution", "320x176", "--face_net_resolution", "200x200"]

    if video_out is not None:
        run_cmds += ["--write_video", video_out, "--write_video_fps", "30"]

    if img_out is not None:
        run_cmds += ["--write_images", img_out]

    if not (video_out is not None or img_out is not None):
        run_cmds += ["--render_pose", "0"]

    logger.info(" ".join(run_cmds))
    subprocess.call(run_cmds, shell=False)
    os.chdir(og_cwd)  # change back to resume


"""
cd /apdcephfs/private_wallyliang/openpose/build_A100/examples/openpose
./build_A100/examples/openpose/openpose.bin --image_dir /root/apdcephfs/private_wallyliang/test_op --display 0 --model_pose BODY_25 --face --hand --write_images /root/apdcephfs/private_wallyliang/test_op
"""
