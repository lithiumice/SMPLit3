from data_loaders.dataset import CommanDataloader
from torch.utils.data import DataLoader
import torch


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["inp"] for b in notnone_batches]
    if "lengths" in notnone_batches[0]:
        lenbatch = [b["lengths"] for b in notnone_batches]
    else:
        lenbatch = [len(b["inp"][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = (
        lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1])
        .unsqueeze(1)
        .unsqueeze(1)
    )  # unqueeze for broadcasting

    motion = databatchTensor
    cond = {"y": {"mask": maskbatchTensor, "lengths": lenbatchTensor}}

    if "text" in notnone_batches[0]:
        textbatch = [b["text"] for b in notnone_batches]
        cond["y"].update({"text": textbatch})

    if "tokens" in notnone_batches[0]:
        textbatch = [b["tokens"] for b in notnone_batches]
        cond["y"].update({"tokens": textbatch})

    for select_key in notnone_batches[0]:
        if select_key not in ["text", "tokens"]:
            actionbatch = [b[select_key] for b in notnone_batches]
            cond["y"].update({select_key: collate_tensors(actionbatch)})

    return motion, cond


def diffgen_collate(batch):
    adapted_batch = []
    for b in batch:
        (style_code, target_motion, past_motion, ctrl_traj) = b
        adapted_batch.append(
            {
                "inp": torch.tensor(target_motion.T)
                .float()
                .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
                "past_motion": torch.tensor(past_motion.T)
                .float()
                .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
                "example_motoin": torch.tensor(style_code.T)
                .float()
                .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
                "ctrl_traj": torch.tensor(ctrl_traj.T)
                .float()
                .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
            }
        )

    return collate(adapted_batch)


def difftraj_collate(batch):
    adapted_batch = []

    def cvt(ds):
        return torch.tensor(ds.T).float().unsqueeze(1)  # [seqlen, J] -> [J, 1, seqlen]

    for b in batch:
        # this is defined in difftraj/data_loaders/humanml/data/dataset.py 's dataset return
        style_code, condition, target, motoin_info_all = b
        adapted_batch.append(
            {
                "inp": cvt(target),
                "condition": cvt(condition),
                "example_motoin": cvt(style_code),
                "motoin_info_all": cvt(motoin_info_all),
            }
        )
    return collate(adapted_batch)


def traj_pred_ar_collate(batch):
    adapted_batch = []
    for b in batch:
        (style_code, p_motion, f_pose, f_motion) = b
        adapted_batch.append(
            {
                "inp": torch.tensor(f_motion.T)
                .float()
                .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
                "p_motion": torch.tensor(p_motion.T)
                .float()
                .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
                "f_pose": torch.tensor(f_pose.T)
                .float()
                .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
                "example_motoin": torch.tensor(style_code.T)
                .float()
                .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
            }
        )
    return collate(adapted_batch)


def diffpose_collate(batch):
    adapted_batch = []
    for b in batch:
        (norm_kp2d, cam_angvel, target, all_mask) = b
        adapted_batch.append(
            {
                "inp": torch.tensor(target.T)
                .float()
                .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
                "norm_kp2d": torch.tensor(norm_kp2d.T)
                .float()
                .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
                "cam_angvel": torch.tensor(cam_angvel.T)
                .float()
                .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
                "all_mask": torch.tensor(all_mask.T).int(),
            }
        )
    return collate(adapted_batch)


def get_dataset_loader(
    name, batch_size, num_frames, split="train", hml_mode="train", args=None
):
    if name == "difftraj":
        dataset_type = 'difftraj_dataset'
        collate_fn = traj_pred_ar_collate if args.use_ar else difftraj_collate
    elif name == "diffgen":
        dataset_type = 'diffgen_dataset'
        collate_fn = diffgen_collate
    elif name == "diffpose":
        dataset_type = 'diffpose_dataset'
        collate_fn = diffpose_collate

    if args.debug:
        num_workers=0
    else:
        num_workers = 8
    dataset = CommanDataloader(dataset_type=dataset_type, split=split, num_frames=num_frames, mode=hml_mode, args=args)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return loader


def get_model_args(args, data):
    # print(f'args.in_type: {args.in_type}')
    clip_version = "ViT-B/32"
    num_actions = 1
    if data is not None:
        if hasattr(data.dataset, "num_actions"):
            num_actions = data.dataset.num_actions
        if hasattr(data.dataset, "inp_len"):
            args.inp_len = data.dataset.t2m_dataset.inp_len
        if hasattr(data.dataset, "pastMotion_len"):
            args.pastMotion_len = data.dataset.t2m_dataset.pastMotion_len

    if args.dataset == "difftraj":
        if args.use_old_model:
            njoints = 11 + 4 - 2
        else:
            njoints = 11
    elif args.dataset == "diffgen":
        njoints = 203
    elif args.dataset == "diffpose":
        njoints = 147
        if args.diffpose_body_only:
            # njoints -= 11
            njoints = 17 * 3

    print(f"njoints: {njoints}")
    return {
        "modeltype": "",
        "njoints": njoints,
        "nfeats": 1,
        "num_actions": num_actions,
        "translation": True,
        "pose_rep": "rot6d",
        "glob": True,
        "glob_rot": True,
        "latent_dim": args.latent_dim,
        "ff_size": 1024,
        "num_layers": args.layers,
        "num_heads": 4,
        "dropout": 0.1,
        "activation": "gelu",
        "data_rep": "rot6d",
        "cond_mode": "",
        "cond_mask_prob": args.cond_mask_prob,
        "action_emb": "tensor",
        "arch": args.arch,
        "emb_trans_dec": args.emb_trans_dec,
        "clip_version": clip_version,
        "dataset": args.dataset,
        "args": args,
    }
