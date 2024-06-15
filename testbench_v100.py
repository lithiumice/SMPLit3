from mocap import init_model, mocap
from pathlib import Path
import time
import os


def testbench():
    vid_path = "/apdcephfs/private_wallyliang/videos/mocap_test_case_30fps/taijiquan_female_13s.mp4"

    # model init
    st = time.time()

    model = init_model()

    used_time = time.time() - st
    print(f">>>> init model{used_time=}")

    # run model
    st = time.time()

    input_video_path = vid_path
    output_smplx_path = "output/{}.npz".format(Path(vid_path).stem)

    mocap(
        model,
        input_video_path,
        output_smplx_path,
        # use_yolo_det=False
    )

    print(f"{output_smplx_path=}")
    used_time = time.time() - st
    print(f"{used_time=}")

    with open("output/{}.log".format(Path(vid_path).stem), "a+") as f:
        f.write(
            f"Video {input_video_path}, Output {output_smplx_path}, Time used {used_time}\n"
        )


if __name__ == "__main__":
    testbench()
