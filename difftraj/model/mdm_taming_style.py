import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
import model.PAE as PAE_model

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
import numpy as np


def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out", "fan_avg"]
    if mode not in valid_modes:
        raise ValueError(
            "Mode {} not supported, please use one of {}".format(mode, valid_modes)
        )

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(tensor, gain=1.0, mode="fan_in"):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    var = gain / max(1.0, fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode="fan_avg")


def dense(in_channels, out_channels, init_scale=1.0):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin


def conv2d(
    in_planes,
    out_planes,
    kernel_size=(3, 3),
    stride=1,
    dilation=1,
    padding=1,
    bias=True,
    padding_mode="zeros",
    init_scale=1.0,
):
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        padding_mode=padding_mode,
    )
    variance_scaling_init_(conv.weight, scale=init_scale)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


class GANDiscriminator(nn.Module):
    def __init__(self, init_scale=1.0):
        super(GANDiscriminator, self).__init__()

        self.dropout = 0
        self.latent_dim = 32
        self.input_feats = 203
        # import ipdb;ipdb.set_trace()
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        self.fc1 = nn.Linear(203, self.latent_dim)
        self.fc2 = nn.Linear(100, self.latent_dim)
        self.cat_dim = 15 + 1 + 1 + 15 * 2  # ctrl + t + style + xt and x_t+1

        if 0:
            self.seqTransEncoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=4,
                    dim_feedforward=1024,
                    dropout=self.dropout,
                    activation="gelu",
                ),
                num_layers=1,
            )
            self.final_fc = dense(self.cat_dim * self.latent_dim, 1)
        else:
            init_scale = 1.0
            self.layers = nn.ModuleList()
            layers_sizes = [self.latent_dim, 256, 128, 1]
            for i in range(len(layers_sizes) - 1):
                self.layers.append(
                    dense(layers_sizes[i], layers_sizes[i + 1], init_scale)
                )
                if i < len(layers_sizes) - 2:
                    self.layers.append(nn.LeakyReLU(0.2))
            self.final_fc = dense(self.cat_dim * 1, 10)

    def forward(self, x, t, x_t, y=None, **kwargs):

        x = x.squeeze().permute(2, 0, 1)
        x_t = x_t.squeeze().permute(2, 0, 1)
        x = self.poseEmbedding(x)
        x_t = self.poseEmbedding(x_t)
        # import ipdb;ipdb.set_trace()
        emb = self.embed_timestep(t)  # [1, bs, d]
        xseq = torch.cat((emb, x, x_t), axis=0)  # [xx, bs, 512]

        c1 = y["past_motion"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
        c2 = y["example_motoin"].to(x.device).permute(2, 0, 1)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        # xseq = xseq + c1 + c2
        xseq = torch.cat((c1, c2, xseq), axis=0)  # [xx, bs, 512]

        if 0:
            xseq = self.sequence_pos_encoder(xseq)
            output = self.seqTransEncoder(xseq)
            output = output.permute(1, 0, 2).reshape(-1, self.cat_dim * self.latent_dim)
        else:
            output = xseq
            for layer in self.layers:
                output = layer(output)
            output = output.permute(1, 0, 2).reshape(-1, self.cat_dim * 1)

        output = self.final_fc(output)

        return output


class GANDiscriminatorSIDDMS(nn.Module):
    def __init__(self, init_scale=1.0):
        super(GANDiscriminatorSIDDMS, self).__init__()

        self.dropout = 0
        self.latent_dim = 4
        self.input_feats = 203
        # import ipdb;ipdb.set_trace()
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        self.fc1 = nn.Linear(203, self.latent_dim)
        self.fc2 = nn.Linear(100, self.latent_dim)
        self.cat_dim = 15 + 1 + 1 + 15  # ctrl + t + style + xt

        if 0:
            self.seqTransEncoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=4,
                    dim_feedforward=1024,
                    dropout=self.dropout,
                    activation="gelu",
                ),
                num_layers=1,
            )
            self.final_fc = dense(self.cat_dim * self.latent_dim, 1)
        else:
            init_scale = 1.0
            self.layers = nn.ModuleList()
            layers_sizes = [self.latent_dim, 256, 128, 1]
            for i in range(len(layers_sizes) - 1):
                self.layers.append(
                    dense(layers_sizes[i], layers_sizes[i + 1], init_scale)
                )
                if i < len(layers_sizes) - 2:
                    self.layers.append(nn.LeakyReLU(0.2))
            self.final_fc = dense(self.cat_dim * 1, 1)

    def forward(self, x, t, y=None, **kwargs):

        # import ipdb;ipdb.set_trace()
        x = x.squeeze().permute(2, 0, 1)
        x = self.poseEmbedding(x)
        emb = self.embed_timestep(t)  # [1, bs, d]
        xseq = torch.cat((emb, x), axis=0)  # [xx, bs, 512]

        c1 = y["past_motion"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
        c2 = y["example_motoin"].to(x.device).permute(2, 0, 1)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        # xseq = xseq + c1 + c2
        xseq = torch.cat((c1, c2, xseq), axis=0)  # [xx, bs, 512]

        if 0:
            xseq = self.sequence_pos_encoder(xseq)
            output = self.seqTransEncoder(xseq)
            output = output.permute(1, 0, 2).reshape(-1, self.cat_dim * self.latent_dim)
        else:
            output = xseq
            for layer in self.layers:
                output = layer(output)
            output = output.permute(1, 0, 2).reshape(-1, self.cat_dim * 1)

        output = self.final_fc(output)

        return output


class MDM(nn.Module):
    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        num_actions,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
        legacy=False,
        data_rep="rot6d",
        dataset="amass",
        clip_dim=512,
        arch="trans_enc",
        emb_trans_dec=False,
        clip_version=None,
        inp_len=200,
        **kargs,
    ):
        super().__init__()
        njoints = 203
        self.args = kargs["args"]
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get("action_emb", None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get("normalize_encoder_output", False)
        self.inp_len = inp_len

        self.cond_mode = kargs.get("cond_mode", "no_cond")
        print(f"self.cond_mode: {self.cond_mode}")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        print(f"self.cond_mask_prob: {self.cond_mask_prob}")
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == "gru" else 0

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        if self.arch == "trans_enc":
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
            )

            self.seqTransEncoder = nn.TransformerEncoder(
                seqTransEncoderLayer, num_layers=self.num_layers
            )
        elif self.arch == "trans_dec":
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=activation,
            )
            self.seqTransDecoder = nn.TransformerDecoder(
                seqTransDecoderLayer, num_layers=self.num_layers
            )
        elif self.arch == "gru":
            print("GRU init")
            self.gru = nn.GRU(
                self.latent_dim,
                self.latent_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(
                "Please choose correct architecture [trans_enc, trans_dec, gru]"
            )

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        #############
        # style condition
        self.use_pae_enc = "paeStyle" in self.args.in_type
        if self.use_pae_enc:
            fps = 20
            window = 2.0
            frames = int(window * fps) + 1
            joints = 22
            input_channels = 3 * joints
            embedding_channels = 5
            self.pae_latent_dim = embedding_channels * 41
            self.pae2latent_fc = nn.Linear(self.pae_latent_dim, self.latent_dim)
            self.pae_network = PAE_model.Model(
                input_channels=input_channels,
                embedding_channels=embedding_channels,
                time_range=frames,
                window=window,
            )

        self.use_gru_enc = "gruStyle" in self.args.in_type
        if self.use_gru_enc:
            self.pae_network = nn.GRU(
                input_channels,
                input_channels,
                num_layers=self.num_layers,
                batch_first=True,
            )
            self.pae2latent_fc = nn.Linear(input_channels, self.latent_dim)

        self.use_onehot_style = "onehot" in self.args.in_type
        if self.use_onehot_style:
            self.style_onehot_fc = nn.Linear(100, self.latent_dim)

        self.use_motion_clip = "motionclip" in self.args.in_type
        if self.use_motion_clip:
            if not "preCompute" in self.args.in_type:
                from data_loaders.humanml.data.dataset import get_motionclip_model

                self.motoin_clip_model = get_motionclip_model()
            self.motion_clip_emb_fc = nn.Linear(self.latent_dim, self.latent_dim)

        self.ar_type = 1
        self.use_past_motion = "pastMotion" in self.args.in_type
        if self.use_past_motion:
            self.past_motion_encoder = nn.Linear(
                203, self.latent_dim
            )  # traj（11） + pose6d（21*3）
            self.ctrl_traj_encoder = nn.Linear(4, self.latent_dim)
        else:
            self.ctrl_traj_encoder = nn.Linear(5, self.latent_dim)

        # import ipdb;ipdb.set_trace()
        # if self.args.use_gan:

        self.down_rate = 1
        if "simpleDownsampleCtrl" in self.args.in_type:
            self.down_rate = 2  # -》10fps
            if "downRate4" in self.args.in_type:
                self.down_rate = 4  # -》5fps
        #############

        self.output_process = OutputProcess(
            self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats
        )
        self.input_process = InputProcess(
            self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim
        )
        self.rot2xyz = Rotation2xyz(device="cpu", dataset=self.dataset)

    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device="cpu", jit=False
        )  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        # 获取cond的批次大小
        bs = cond.shape[0]
        # af = cond.shape[1:]
        # af = [1 for _ in af]
        if force_mask:
            import ipdb

            ipdb.set_trace()
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            # import ipdb;ipdb.set_trace()
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            )  # 1-> use null_cond, 0-> use real cond
            mask = mask.unsqueeze(1).unsqueeze(1)
            return cond * (1.0 - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = (
            20 if self.dataset in ["humanml", "kit"] else None
        )  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(
                raw_text, context_length=context_length, truncate=True
            ).to(
                device
            )  # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length],
                dtype=texts.dtype,
                device=texts.device,
            )
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(
                device
            )  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None, **kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        force_mask = y.get("uncond", False)

        if self.arch == "gru":
            x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)  # [#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)  # [bs, d, #frames]
            emb_gru = emb_gru.reshape(
                bs, self.latent_dim, 1, nframes
            )  # [bs, d, 1, #frames]
            x = torch.cat(
                (x_reshaped, emb_gru), axis=1
            )  # [bs, d+joints*feat, 1, #frames]

        # import ipdb;ipdb.set_trace()
        x = self.input_process(x)  # T,B,512

        if self.arch == "trans_enc":
            if self.use_past_motion:
                tmp_cond = y["past_motion"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
                if "useCFG" in self.args.in_type:
                    tmp_cond = self.mask_cond(tmp_cond, force_mask=force_mask)
                past_motion_emb = self.past_motion_encoder(tmp_cond)

            # add style condition
            # y['example_motoin']: [bs,x,1,T]
            if self.use_motion_clip:
                # 60 frame, fps 30, 2 seconds
                if "preCompute" in self.args.in_type:
                    example_motoin_embedding = (
                        y["example_motoin"].to(x.device).squeeze(2).unsqueeze(0)
                    )
                else:
                    example_motoin = (
                        y["example_motoin"]
                        .to(x.device)[:, :, 0, :, :]
                        .permute(0, 2, 1, 3)
                    )
                    batch = {
                        "x": example_motoin,
                        "y": torch.zeros(bs).long().to(x.device),
                        "mask": torch.ones(bs, 60).bool().to(x.device),
                        "lengths": torch.ones(bs, 60)
                        .bool()
                        .to(x.device)
                        .data.fill_(60),
                    }
                    batch = self.motoin_clip_model(batch)
                    example_motoin_embedding = batch["z"].unsqueeze(0)  # 1, bs, 512
                example_motoin_embedding = self.motion_clip_emb_fc(
                    example_motoin_embedding
                )

            if self.use_pae_enc:
                # latent # bs, 5, 61
                example_motoin = y["example_motoin"].to(x.device)[
                    :, :, 0, :
                ]  # bs, joints*3, example_len,
                rec_y, latent, signal, params = self.pae_network(example_motoin)
                example_motoin_embedding = self.pae2latent_fc(
                    signal.reshape(-1, self.pae_latent_dim)
                ).unsqueeze(0)

            if self.use_gru_enc:
                example_motoin = (
                    y["example_motoin"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
                )  # [T,bs,x]
                aaa, _ = self.pae_network(example_motoin)
                example_motoin_embedding = self.pae2latent_fc(aaa)

            if self.use_onehot_style:
                example_motoin = y["example_motoin"].to(x.device).permute(2, 0, 1)
                example_motoin_embedding = self.style_onehot_fc(
                    example_motoin
                )  # 1,bs,512

            # ctrl_traj: bs,T,1,10 -> T,bs,10
            ctrl_traj = y["ctrl_traj"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]

            if self.use_past_motion:
                ctrl_traj_emb = self.ctrl_traj_encoder(ctrl_traj)  # T,bs,512
            else:
                ctrl_traj_emb = self.ctrl_traj_encoder(ctrl_traj)  # T,bs,512

            xseq = torch.cat((ctrl_traj_emb, emb, x), axis=0)  # [xx, bs, 512]

            if not "noStyle" in self.args.in_type:
                xseq = torch.cat((example_motoin_embedding, xseq), axis=0)

            if self.use_past_motion:
                xseq = torch.cat((past_motion_emb, xseq), axis=0)  # [xx, bs, 512]

            # import ipdb;ipdb.set_trace()
            ctrl_emb_t = self.inp_len // self.down_rate
            if self.use_past_motion:
                ctrl_emb_t += self.args.pastMotion_len

            if not "noStyle" in self.args.in_type:
                style_dim = 1
            else:
                style_dim = 0

            # output + time + style + past + ctrlTraj
            if not xseq.shape[0] == (self.inp_len + 1 + style_dim + ctrl_emb_t):
                import ipdb

                ipdb.set_trace()

            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[
                -self.inp_len :
            ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            # torch.isnan(output.sum())

        # elif self.arch == 'trans_dec':
        #     if self.emb_trans_dec:
        #         xseq = torch.cat((emb, x), axis=0)
        #     else:
        #         xseq = x
        #     xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        #     if self.emb_trans_dec:
        #         output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
        #     else:
        #         output = self.seqTransDecoder(tgt=xseq, memory=emb)
        # elif self.arch == 'gru':
        #     xseq = x
        #     xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
        #     output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "rot2xyz"):
            self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if hasattr(self, "rot2xyz"):
            self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == "rot_vel":
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        if self.data_rep in ["rot6d", "xyz", "hml_vec"]:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == "rot_vel":
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == "rot_vel":
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ["rot6d", "xyz", "hml_vec"]:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == "rot_vel":
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
