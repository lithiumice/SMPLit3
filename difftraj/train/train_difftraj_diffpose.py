import os
import sys


main_code_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(main_code_path)

import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_gaussian_diffusion
from data_loaders.get_data import get_model_args, collate
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation


def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name="Args")

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        pass
        # raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        args=args,
    )

    print("creating model and diffusion...")
    if args.dataset == "difftraj":
        from model.mdm_traj import MDM

        model = MDM(**get_model_args(args, data))
    elif args.dataset == "diffpose":
        from model.model_diffpose import MODEL_DIFFPOSE

        model = MODEL_DIFFPOSE(**get_model_args(args, data))

    diffusion = create_gaussian_diffusion(args)

    model.to(dist_util.dev())
    if hasattr(model, "rot2xyz"):
        model.rot2xyz.smpl_model.eval()

    print(
        "Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0)
    )
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()