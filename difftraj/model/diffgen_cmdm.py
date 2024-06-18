import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from .transformer import *

from model.mdm import (
    PositionalEncoding,
    TimestepEmbedder,
    InputProcess,
    EmbedAction
)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class CMDM(nn.Module):
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

        self.args = args = kargs["args"]
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

        self.emb_trans_dec = emb_trans_dec

        # <==================MDM
        # seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
        #                                                     nhead=self.num_heads,
        #                                                     dim_feedforward=self.ff_size,
        #                                                     dropout=self.dropout,
        #                                                     activation=self.activation)

        # self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
        #                                                 num_layers=self.num_layers)
        self.input_process = InputProcess(
            self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim
        )

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )

        self.seqTransEncoder = TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers
        )

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        self.past_motion_encoder = nn.Linear(
            203, self.latent_dim
        )  # traj（11） + pose6d（21*3）
        self.ctrl_traj_encoder = nn.Linear(4, self.latent_dim)
        # ==================>

        # <==================CMDM
        # n_joints = 22 if njoints == 263 else 21
        # self.input_hint_block = HintBlock(self.data_rep, n_joints * 3, self.latent_dim)
        # import ipdb;ipdb.set_trace()
        if not self.args.train_cmdm_base:
            print("[INFO] setting cmdm here")
            self.style_onehot_fc = nn.Linear(100, self.latent_dim)

            self.c_input_process = InputProcess(
                self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim
            )

            self.c_sequence_pos_encoder = PositionalEncoding(
                self.latent_dim, self.dropout
            )

            seqTransEncoderLayer = TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
            )
            self.c_seqTransEncoder = TransformerEncoder(
                seqTransEncoderLayer,
                num_layers=self.num_layers,
                return_intermediate=True,
            )

            self.zero_convs = zero_module(
                nn.ModuleList(
                    [
                        nn.Linear(self.latent_dim, self.latent_dim)
                        for _ in range(self.num_layers)
                    ]
                )
            )

            self.c_embed_timestep = TimestepEmbedder(
                self.latent_dim, self.sequence_pos_encoder
            )

            self.c_past_motion_encoder = nn.Linear(
                203, self.latent_dim
            )  # traj（11） + pose6d（21*3）
            self.c_ctrl_traj_encoder = nn.Linear(4, self.latent_dim)
            # ==================>

        self.output_process = OutputProcess(
            self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats
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

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # if 'example_motoin' in y.keys():
        if not self.args.train_cmdm_base:
            control = self.cmdm_forward(x, timesteps, y)
            # print('cmdm here')
        else:
            control = None
        output = self.mdm_forward(x, timesteps, y, control)
        return output

    # def forward(self, x, timesteps, y=None, **kwargs):
    def mdm_forward(self, x, timesteps, y=None, control=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        x = self.input_process(x)  # T,B,512

        tmp_cond = y["past_motion"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
        past_motion_emb = self.past_motion_encoder(tmp_cond)
        ctrl_traj = (
            y["ctrl_traj"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
        )  # ctrl_traj: bs,T,1,10 -> T,bs,10
        ctrl_traj_emb = self.ctrl_traj_encoder(ctrl_traj)  # T,bs,512
        xseq = torch.cat(
            (past_motion_emb, ctrl_traj_emb, emb, x), axis=0
        )  # [xx, bs, 512]
        # assert xseq.shape[0] == (20 + 1 + 15)
        # import ipdb;ipdb.set_trace()

        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq, control=control)[
            -self.inp_len :
        ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def cmdm_forward(self, x, timesteps, y=None, weight=1.0):
        """
        Realism Guidance
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        emb = self.c_embed_timestep(timesteps)  # [1, bs, d]
        example_motoin = y["example_motoin"].to(x.device).permute(2, 0, 1)
        example_motoin_embedding = self.style_onehot_fc(example_motoin)  # 1,bs,512
        x = self.c_input_process(x)
        x += example_motoin_embedding
        # import ipdb;ipdb.set_trace()

        tmp_cond = y["past_motion"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
        past_motion_emb = self.c_past_motion_encoder(tmp_cond)
        ctrl_traj = (
            y["ctrl_traj"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
        )  # ctrl_traj: bs,T,1,10 -> T,bs,10
        ctrl_traj_emb = self.c_ctrl_traj_encoder(ctrl_traj)  # T,bs,512
        xseq = torch.cat(
            (past_motion_emb, ctrl_traj_emb, emb, x), axis=0
        )  # [xx, bs, 512]
        # assert xseq.shape[0] == (20 + 1 + 15)

        xseq = self.c_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.c_seqTransEncoder(xseq)  # [seqlen+1, bs, d]

        control = []
        for i, module in enumerate(self.zero_convs):
            control.append(module(output[i]))
        control = torch.stack(control)
        control = control * weight
        return control

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                bs, 1
            )  # 1-> use null_cond, 0-> use real cond
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
    def cmdm_forward_text_mdm(self, x, timesteps, y=None, weight=1.0):
        """
        Realism Guidance
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        emb = self.c_embed_timestep(timesteps)  # [1, bs, d]

        seq_mask = y["hint"].sum(-1) != 0

        guided_hint = self.input_hint_block(y["hint"].float())  # [bs, d]

        force_mask = y.get("uncond", False)
        if "text" in self.cond_mode:
            enc_text = self.encode_text(y["text"])
            emb += self.c_embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        x = self.c_input_process(x)

        x += guided_hint * seq_mask.permute(1, 0).unsqueeze(-1)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.c_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.c_seqTransEncoder(xseq)  # [seqlen+1, bs, d]

        control = []
        for i, module in enumerate(self.zero_convs):
            control.append(module(output[i]))
        control = torch.stack(control)

        control = control * weight
        return control



    def mdm_forward_text_mdm(self, x, timesteps, y=None, control=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get("uncond", False)
        if "text" in self.cond_mode:
            enc_text = self.encode_text(y["text"])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        x = self.input_process(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq, control=control)[
            1:
        ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def forward_text_mdm(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        if "hint" in y.keys():
            control = self.cmdm_forward_text_mdm(x, timesteps, y)
        else:
            control = None
        output = self.mdm_forward_text_mdm(x, timesteps, y, control)
        return output

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "rot2xyz"):
            self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if hasattr(self, "rot2xyz"):
            self.rot2xyz.smpl_model.train(*args, **kwargs)

class HintBlock(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.ModuleList(
            [
                nn.Linear(self.input_feats, self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim),
                zero_module(nn.Linear(self.latent_dim, self.latent_dim)),
            ]
        )

    def forward(self, x):
        x = x.permute((1, 0, 2))

        for module in self.poseEmbedding:
            x = module(x)  # [seqlen, bs, d]
        return x
