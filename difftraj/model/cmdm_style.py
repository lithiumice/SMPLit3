import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from .transformer import *


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


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
