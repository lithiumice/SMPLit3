import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MODEL_DIFFPOSE(nn.Module):
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
        **kargs
    ):
        super().__init__()

        self.args = kargs["args"]
        self.seq_len = self.args.seq_len
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
        self.cond_mode = kargs.get("cond_mode", "no_cond")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.arch = arch

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

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

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        self.input_process = InputProcess(self.data_rep, self.njoints, self.latent_dim)

        # import pdb;pdb.set_trace()
        self.kpt2d_enc = nn.Linear(17 * 2, self.latent_dim)
        if not self.args.diffpose_body_only:
            self.cam_angvel_enc = nn.Linear(6, self.latent_dim)
        self.output_process = OutputProcess(
            self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats
        )

    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

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

    def mask_cond2(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                bs, 1, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        force_mask = y.get("uncond", False)
        x = self.input_process(x)  # T,B,512

        cond = y["norm_kp2d"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]

        if self.args.add_mask:
            import pdb

            pdb.set_trace()

        emb1 = self.kpt2d_enc(cond)

        if self.args.diffpose_body_only:
            # import pdb;pdb.set_trace()
            x = x + emb1
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            assert xseq.shape[0] == (self.seq_len + 1)

        else:
            cond2 = y["cam_angvel"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
            emb2 = self.cam_angvel_enc(cond2)

            # x = x + emb1
            # xseq = torch.cat((emb2, emb, x), axis=0)  # [seqlen+1, bs, d]
            # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            # assert xseq.shape[0]==(self.seq_len*2+1)

            # import pdb;pdb.set_trace()
            xseq = torch.cat((emb1, emb2, emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            assert xseq.shape[0] == (self.seq_len * 3 + 1)

        output = self.seqTransEncoder(xseq)[
            -self.seq_len :
        ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        # import pdb;pdb.set_trace()
        return output

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "rot2xyz"):
            self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if hasattr(self, "rot2xyz"):
            self.rot2xyz.smpl_model.train(*args, **kwargs)


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
        **kargs
    ):
        super().__init__()

        self.args = kargs["args"]
        self.seq_len = self.args.seq_len
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
        self.cond_mode = kargs.get("cond_mode", "no_cond")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.arch = arch

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

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

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        self.input_process = InputProcess(self.data_rep, self.njoints, self.latent_dim)

        # import pdb;pdb.set_trace()
        if self.args.use_old_model:
            self.local_pose_encoder = nn.Linear(22 * 3 + 2, self.latent_dim)
            self.output_process = OutputProcess(
                self.data_rep,
                self.input_feats,
                self.latent_dim,
                self.njoints,
                self.nfeats,
            )
        else:
            if self.args.use_ar:
                self.p_motion_enc = nn.Linear(203, self.latent_dim)
                self.f_pose_enc = nn.Linear(22 * 3 + 21 * 6 + 2, self.latent_dim)
            else:
                self.condition_encoder = nn.Linear(22 * 3 + 2, self.latent_dim)
            self.output_process = OutputProcess(
                self.data_rep,
                self.input_feats,
                self.latent_dim,
                self.njoints,
                self.nfeats,
            )

    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

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

    def mask_cond2(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                bs, 1, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        force_mask = y.get("uncond", False)
        x = self.input_process(x)  # T,B,512

        if self.args.use_old_model:
            tmp_cond = y["local_pose"].to(x.device)
            cond_bs, cond_njoints, cond_nfeats, cond_nframes = tmp_cond.shape
            tmp_cond = tmp_cond.permute((3, 0, 1, 2)).reshape(
                cond_nframes, cond_bs, cond_njoints * cond_nfeats
            )
            local_pose_emb = self.local_pose_encoder(tmp_cond)
            x += local_pose_emb

            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[
                1:
            ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        else:
            # import pdb;pdb.set_trace()
            if self.args.use_ar:
                cond = y["p_motion"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
                p_motion_emb = self.p_motion_enc(cond)

                cond = y["f_pose"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
                f_pose_emb = self.f_pose_enc(cond)

                xseq = torch.cat(
                    (p_motion_emb, f_pose_emb, emb, x), axis=0
                )  # [seqlen+1, bs, d]
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                # import pdb;pdb.set_trace()
                assert xseq.shape[0] == (self.args.p_len + self.args.f_len * 2 + 1)
                output = self.seqTransEncoder(xseq)[
                    -self.args.f_len :
                ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
                output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
            else:
                # import pdb;pdb.set_trace()
                if "add_emb" in self.args and self.args.add_emb:
                    cond = y["condition"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
                    condition_emb = self.condition_encoder(cond)
                    x = x + condition_emb
                    xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
                    xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                    assert xseq.shape[0] == (self.seq_len + 1)
                    output = self.seqTransEncoder(xseq)[
                        -self.seq_len :
                    ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
                    output = self.output_process(
                        output
                    )  # [bs, njoints, nfeats, nframes]
                else:
                    cond = y["condition"].to(x.device).permute(3, 2, 0, 1)[:, 0, :, :]
                    condition_emb = self.condition_encoder(cond)
                    xseq = torch.cat(
                        (condition_emb, emb, x), axis=0
                    )  # [seqlen+1, bs, d]
                    xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                    # import pdb;pdb.set_trace()
                    assert xseq.shape[0] == (self.seq_len * 2 + 1)
                    output = self.seqTransEncoder(xseq)[
                        -self.seq_len :
                    ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
                    output = self.output_process(
                        output
                    )  # [bs, njoints, nfeats, nframes]
                return output

        # import pdb;pdb.set_trace()
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
