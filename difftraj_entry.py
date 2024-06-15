import os
import sys

__cur__ = os.path.join(os.path.dirname(__file__))
print(f"{__cur__=}")
os.chdir(__cur__)

sys.path.append(__cur__)
sys.path.append(__cur__ + "/third-party")
sys.path.append(__cur__ + "/difftraj")

import time
import click
from lib.loco.trajdiff import *


@click.command()
@click.option("--inp", type=str)
@click.option("--flip_x", is_flag=True)
@click.option("--flip_y", is_flag=True)
@click.option("--rot_deg", type=float, default=0.0)
def difftraj_entry(
    inp,
    flip_x,
    flip_y,
    rot_deg,
):
    traj_predictor = DiffTraj()

    def pose_to_traj(npz_file, output_pth):
        st = time.time()
        data = np.load(npz_file)
        pred_body_pose = tt(torch.from_numpy(data["poses"][:, 1 : 22 + 2, :]))
        pred_root_world = tt(torch.from_numpy(data["poses"][:, 0, :]))
        pred_trans_world = tt(torch.from_numpy(data["trans"]))

        nonlocal rot_deg
        rot_deg = (rot_deg / 180) * np.pi
        # import ipdb;ipdb.set_trace()
        if flip_x:
            cvt = (
                a2m(torch.tensor([[rot_deg, 0, 0]])).float().to(pred_root_world.device)
            )
            pred_root_world = m2a(cvt.mT @ a2m(pred_root_world))

        if flip_y:
            cvt = (
                a2m(torch.tensor([[0, rot_deg, 0]])).float().to(pred_root_world.device)
            )
            pred_root_world = m2a(cvt.mT @ a2m(pred_root_world))

        difftraj_root, difftraj_trans = traj_predictor.pose_to_traj(
            pred_root_world, pred_body_pose, pred_trans_world
        )
        results = {
            "0": {
                "pred_body_pose": tt(pred_body_pose),
                "difftraj_root": tt(difftraj_root),
                "difftraj_trans": tt(difftraj_trans),
                "betas": tt(torch.zeros(3, 10)),
            }
        }
        save_to_blender_smplx_addon_npz(results, "difTraj_raw", output_pth=output_pth)
        print(f"difftraj used time: {time.time()-st} seconds.")

    if inp.endswith(".npz"):
        pose_to_traj(inp, inp.replace(".npz", "_cvt.npz"))
    else:
        save_npz_root = inp + "_pose2Traj"
        os.makedirs(save_npz_root, exist_ok=True)
        for npz_file in glob(f"{inp}/*.npz"):
            print(f"npz_file: {npz_file}")
            output_pth = os.path.join(save_npz_root, os.path.basename(npz_file))
            pose_to_traj(npz_file, output_pth)


if __name__ == "__main__":
    difftraj_entry()
