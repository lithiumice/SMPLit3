import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe, crop_face
from datasets.base_dataset import create_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="samples/mead_90.png",
        help="Path to the input image/video",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="assets/SMIRK_em1.pt",
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--crop", action="store_true", help="Crop the face using mediapipe"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="output",
        help="Path to save the output (will be created if not exists)",
    )
    parser.add_argument(
        "--use_smirk_generator",
        action="store_true",
        help="Use SMIRK neural image to image translator to reconstruct the image",
    )

    args = parser.parse_args()

    image_size = 224

    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint)
    checkpoint_encoder = {
        k.replace("smirk_encoder.", ""): v
        for k, v in checkpoint.items()
        if "smirk_encoder" in k
    }  # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    if args.use_smirk_generator:
        from src.smirk_generator import SmirkGenerator

        smirk_generator = SmirkGenerator(
            in_channels=6, out_channels=3, init_features=32, res_blocks=5
        ).to(args.device)

        checkpoint_generator = {
            k.replace("smirk_generator.", ""): v
            for k, v in checkpoint.items()
            if "smirk_generator" in k
        }  # checkpoint includes both smirk_encoder and smirk_generator
        smirk_generator.load_state_dict(checkpoint_generator)
        smirk_generator.eval()

    # ---- visualize the results ---- #

    flame = FLAME().to(args.device)
    renderer = Renderer().to(args.device)

    # check if input is an image or a video or webcam or directory

    image = cv2.imread(args.input_path)
    kpt_mediapipe = run_mediapipe(image)

    # crop face if needed
    if args.crop:
        if kpt_mediapipe is None:
            print(
                "Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting..."
            )
            exit()

        kpt_mediapipe = kpt_mediapipe[..., :2]

        tform = crop_face(image, kpt_mediapipe, scale=1.4, image_size=image_size)

        cropped_image = warp(
            image, tform.inverse, output_shape=(224, 224), preserve_range=True
        ).astype(np.uint8)

        cropped_kpt_mediapipe = np.dot(
            tform.params,
            np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T,
        ).T
        cropped_kpt_mediapipe = cropped_kpt_mediapipe[:, :2]
    else:
        cropped_image = image
        cropped_kpt_mediapipe = kpt_mediapipe

    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cropped_image = cv2.resize(cropped_image, (224, 224))
    cropped_image = (
        torch.tensor(cropped_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )
    cropped_image = cropped_image.to(args.device)

    outputs = smirk_encoder(cropped_image)

    flame_output = flame.forward(outputs)
    renderer_output = renderer.forward(
        flame_output["vertices"],
        outputs["cam"],
        landmarks_fan=flame_output["landmarks_fan"],
        landmarks_mp=flame_output["landmarks_mp"],
    )

    rendered_img = renderer_output["rendered_img"]

    grid = torch.cat([cropped_image, rendered_img], dim=3)

    # ---- create the neural renderer reconstructed img ---- #
    if args.use_smirk_generator:
        if kpt_mediapipe is None:
            print(
                "Could not find landmarks for the image using mediapipe and cannot create the hull mask for the smirk generator. Exiting..."
            )
            exit()

        mask_ratio_mul = 5
        mask_ratio = 0.01
        mask_dilation_radius = 10

        hull_mask = create_mask(cropped_kpt_mediapipe, (224, 224))

        face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()

        rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
        tmask_ratio = (
            mask_ratio * mask_ratio_mul
        )  # upper bound on the number of points to sample

        npoints, _ = masking_utils.mesh_based_mask_uniform_faces(
            renderer_output["transformed_vertices"],  # sample uniformly from the mesh
            flame_faces=flame.faces_tensor,
            face_probabilities=face_probabilities,
            mask_ratio=tmask_ratio,
        )

        pmask = torch.zeros_like(rendered_mask)
        rsing = torch.randint(0, 2, (npoints.size(0),)).to(npoints.device) * 2 - 1
        rscale = (
            torch.rand((npoints.size(0),)).to(npoints.device) * (mask_ratio_mul - 1) + 1
        )
        rbound = (npoints.size(1) * (1 / mask_ratio_mul) * (rscale**rsing)).long()

        for bi in range(npoints.size(0)):
            pmask[bi, :, npoints[bi, : rbound[bi], 1], npoints[bi, : rbound[bi], 0]] = 1

        hull_mask = (
            torch.from_numpy(hull_mask)
            .type(dtype=torch.float32)
            .unsqueeze(0)
            .to(args.device)
        )

        extra_points = cropped_image * pmask
        masked_img = masking_utils.masking(
            cropped_image,
            hull_mask,
            extra_points,
            mask_dilation_radius,
            rendered_mask=rendered_mask,
        )

        smirk_generator_input = torch.cat([rendered_img, masked_img], dim=1)

        reconstructed_img = smirk_generator(smirk_generator_input)

        grid = torch.cat([grid, reconstructed_img], dim=3)

    grid_numpy = grid.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
    grid_numpy = grid_numpy.astype(np.uint8)
    grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    image_name = args.input_path.split("/")[-1]

    cv2.imwrite(f"{args.out_path}/{image_name}", grid_numpy)
