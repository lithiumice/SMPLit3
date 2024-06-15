import envs
import os, time
import importlib
from collections import namedtuple
import torch
import numpy as np
from typing import Optional
import random
from torch.utils.tensorboard import SummaryWriter
import argparse
import copy

TRAINING_PARAMS = dict(
    horizon=8,
    num_envs=512,
    batch_size=256,
    opt_epochs=5,
    actor_lr=5e-6,
    critic_lr=1e-4,
    gamma=0.95,
    lambda_=0.95,
    disc_lr=1e-5,
    max_epochs=10000,
    save_interval=None,
    terminate_reward=-1,
    control_mode="position",
)


class RunningMeanStd(torch.nn.Module):
    def __init__(self, dim: int, clamp: float = 0):
        super().__init__()
        self.epsilon = 1e-5
        self.clamp = clamp
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, x, unnorm=False):
        mean = self.mean.to(torch.float32)
        var = self.var.to(torch.float32) + self.epsilon
        if unnorm:
            if self.clamp:
                x = torch.clamp(x, min=-self.clamp, max=self.clamp)
            return mean + torch.sqrt(var) * x
        x = (x - mean) * torch.rsqrt(var)
        if self.clamp:
            return torch.clamp(x, min=-self.clamp, max=self.clamp)
        return x

    @torch.no_grad()
    def update(self, x):
        x = x.view(-1, x.size(-1))
        var, mean = torch.var_mean(x, dim=0, unbiased=True)
        count = x.size(0)
        count_ = count + self.count
        delta = mean - self.mean
        m = self.var * self.count + var * count + delta**2 * self.count * count / count_
        self.mean.copy_(self.mean + delta * count / count_)
        self.var.copy_(m / count_)
        self.count.copy_(count_)

    def reset_counter(self):
        self.count.fill_(1)


class DiagonalPopArt(torch.nn.Module):
    def __init__(
        self, dim: int, weight: torch.Tensor, bias: torch.Tensor, momentum: float = 0.1
    ):
        super().__init__()
        self.epsilon = 1e-5

        self.momentum = momentum
        self.register_buffer("m", torch.zeros((dim,), dtype=torch.float64))
        self.register_buffer("v", torch.full((dim,), self.epsilon, dtype=torch.float64))
        self.register_buffer("debias", torch.zeros(1, dtype=torch.float64))

        self.weight = weight
        self.bias = bias

    def forward(self, x, unnorm=False):
        debias = self.debias.clip(min=self.epsilon)
        mean = self.m / debias
        var = (self.v - self.m.square()).div_(debias)
        if unnorm:
            std = torch.sqrt(var)
            return (mean + std * x).to(x.dtype)
        x = ((x - mean) * torch.rsqrt(var)).to(x.dtype)
        return x

    @torch.no_grad()
    def update(self, x):
        x = x.view(-1, x.size(-1))
        running_m = torch.mean(x, dim=0)
        running_v = torch.mean(x.square(), dim=0)
        new_m = self.m.mul(1 - self.momentum).add_(running_m, alpha=self.momentum)
        new_v = self.v.mul(1 - self.momentum).add_(running_v, alpha=self.momentum)
        std = (self.v - self.m.square()).sqrt_()
        new_std_inv = (new_v - new_m.square()).rsqrt_()

        scale = std.mul_(new_std_inv)
        shift = (self.m - new_m).mul_(new_std_inv)

        self.bias.data.mul_(scale).add_(shift)
        self.weight.data.mul_(scale.unsqueeze_(-1))

        self.debias.data.mul_(1 - self.momentum).add_(1.0 * self.momentum)
        self.m.data.copy_(new_m)
        self.v.data.copy_(new_v)


class Discriminator(torch.nn.Module):
    def __init__(self, disc_dim, latent_dim=256):
        super().__init__()
        self.rnn = torch.nn.GRU(disc_dim, latent_dim, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
        )
        if self.rnn is not None:
            i = 0
            for n, p in self.mlp.named_parameters():
                if "bias" in n:
                    torch.nn.init.constant_(p, 0.0)
                elif "weight" in n:
                    gain = 1 if i == 2 else 2**0.5
                    torch.nn.init.orthogonal_(p, gain=gain)
                    i += 1
        self.ob_normalizer = RunningMeanStd(disc_dim)
        self.all_inst = torch.arange(0)

    def forward(self, s, seq_end_frame, normalize=True):
        if normalize:
            s = self.ob_normalizer(s)
        if self.rnn is None:
            s = s.view(s.size(0), -1)
        else:
            n_inst = s.size(0)
            if n_inst > self.all_inst.size(0):
                self.all_inst = torch.arange(
                    n_inst, dtype=seq_end_frame.dtype, device=seq_end_frame.device
                )
            s, _ = self.rnn(s)
            s = s[
                (self.all_inst[:n_inst], torch.clip(seq_end_frame, max=s.size(1) - 1))
            ]
        return self.mlp(s)


class ACModel(torch.nn.Module):

    class Critic(torch.nn.Module):
        def __init__(self, state_dim, goal_dim, value_dim=1, latent_dim=256):
            super().__init__()
            self.rnn = torch.nn.GRU(state_dim, latent_dim, batch_first=True)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(latent_dim + goal_dim, 1024),
                torch.nn.ReLU6(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU6(),
                torch.nn.Linear(512, value_dim),
            )
            i = 0
            for n, p in self.mlp.named_parameters():
                if "bias" in n:
                    torch.nn.init.constant_(p, 0.0)
                elif "weight" in n:
                    torch.nn.init.uniform_(p, -0.0001, 0.0001)
                    i += 1
            self.all_inst = torch.arange(0)

        def forward(self, s, seq_end_frame, g=None):
            if self.rnn is None:
                s = s.view(s.size(0), -1)
            else:
                n_inst = s.size(0)
                if n_inst > self.all_inst.size(0):
                    self.all_inst = torch.arange(
                        n_inst, dtype=seq_end_frame.dtype, device=seq_end_frame.device
                    )
                s, _ = self.rnn(s)
                s = s[
                    (
                        self.all_inst[:n_inst],
                        torch.clip(seq_end_frame, max=s.size(1) - 1),
                    )
                ]
            if g is not None:
                s = torch.cat((s, g), -1)
            return self.mlp(s)

    class Actor(torch.nn.Module):
        def __init__(
            self,
            state_dim,
            act_dim,
            goal_dim,
            latent_dim=256,
            init_mu=None,
            init_sigma=None,
        ):
            super().__init__()
            self.rnn = torch.nn.GRU(state_dim, latent_dim, batch_first=True)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(latent_dim + goal_dim, 1024),
                torch.nn.ReLU6(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU6(),
            )
            self.mu = torch.nn.Linear(512, act_dim)
            self.log_sigma = torch.nn.Linear(512, act_dim)
            with torch.no_grad():
                if init_mu is not None:
                    if torch.is_tensor(init_mu):
                        mu = torch.ones_like(self.mu.bias) * init_mu
                    else:
                        mu = np.ones(self.mu.bias.shape, dtype=np.float32) * init_mu
                        mu = torch.from_numpy(mu)
                    self.mu.bias.data.copy_(mu)
                    torch.nn.init.uniform_(self.mu.weight, -0.00001, 0.00001)
                if init_sigma is None:
                    torch.nn.init.constant_(self.log_sigma.bias, -3)
                    torch.nn.init.uniform_(self.log_sigma.weight, -0.0001, 0.0001)
                else:
                    if torch.is_tensor(init_sigma):
                        log_sigma = (
                            torch.ones_like(self.log_sigma.bias) * init_sigma
                        ).log_()
                    else:
                        log_sigma = np.log(
                            np.ones(self.log_sigma.bias.shape, dtype=np.float32)
                            * init_sigma
                        )
                        log_sigma = torch.from_numpy(log_sigma)
                    self.log_sigma.bias.data.copy_(log_sigma)
                    torch.nn.init.uniform_(self.log_sigma.weight, -0.00001, 0.00001)
                self.all_inst = torch.arange(0)

        def forward(self, s, seq_end_frame, g=None):
            if self.rnn is None:
                s = s.view(s.size(0), -1)
            else:
                n_inst = s.size(0)
                if n_inst > self.all_inst.size(0):
                    self.all_inst = torch.arange(
                        n_inst, dtype=seq_end_frame.dtype, device=seq_end_frame.device
                    )
                s, _ = self.rnn(s)
                s = s[
                    (
                        self.all_inst[:n_inst],
                        torch.clip(seq_end_frame, max=s.size(1) - 1),
                    )
                ]
            if g is not None:
                s = torch.cat((s, g), -1)
            latent = self.mlp(s)
            mu = self.mu(latent)
            sigma = torch.exp(self.log_sigma(latent)) + 1e-8
            return torch.distributions.Normal(mu, sigma)

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        goal_dim: int = 0,
        value_dim: int = 1,
        normalize_value: bool = True,
        init_mu: Optional[torch.Tensor or float] = None,
        init_sigma: Optional[torch.Tensor or float] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.actor = self.Actor(
            state_dim, act_dim, self.goal_dim, init_mu=init_mu, init_sigma=init_sigma
        )
        self.critic = self.Critic(state_dim, goal_dim, value_dim)
        self.ob_normalizer = RunningMeanStd(state_dim, clamp=5.0)
        if normalize_value:
            self.value_normalizer = DiagonalPopArt(
                value_dim, self.critic.mlp[-1].weight, self.critic.mlp[-1].bias
            )
        else:
            self.value_normalizer = None

    def observe(self, obs, norm=True):
        if self.goal_dim > 0:
            s = obs[:, : -self.goal_dim]
            g = obs[:, -self.goal_dim :]
        else:
            s = obs
            g = None
        s = s.view(*s.shape[:-1], -1, self.state_dim)
        return self.ob_normalizer(s) if norm else s, g

    def eval_(self, s, seq_end_frame, g, unnorm):
        v = self.critic(s, seq_end_frame, g)
        if unnorm and self.value_normalizer is not None:
            v = self.value_normalizer(v, unnorm=True)
        return v

    def act(self, obs, seq_end_frame, stochastic=None, unnorm=False):
        if stochastic is None:
            stochastic = self.training
        s, g = self.observe(obs)
        pi = self.actor(s, seq_end_frame, g)
        if stochastic:
            a = pi.sample()
            lp = pi.log_prob(a)
            if g is not None:
                g = g[..., : self.goal_dim]
            return a, self.eval_(s, seq_end_frame, g, unnorm), lp
        else:
            return pi.mean

    def evaluate(self, obs, seq_end_frame, unnorm=False):
        s, g = self.observe(obs)
        if g is not None:
            g = g[..., : self.goal_dim]
        return self.eval_(s, seq_end_frame, g, unnorm)

    def forward(self, obs, seq_end_frame, unnorm=False):
        s, g = self.observe(obs)
        pi = self.actor(s, seq_end_frame, g)
        if g is not None:
            g = g[..., : self.goal_dim]
        return pi, self.eval_(s, seq_end_frame, g, unnorm)


class ACModel2(torch.nn.Module):

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        goal_dim: int = 0,
        value_dim: int = 1,
        normalize_value: bool = True,
        init_mu: Optional[torch.Tensor or float] = None,
        init_sigma: Optional[torch.Tensor or float] = None,
        meta_goal_dim: int = 0,
    ):
        super().__init__()
        self.state_dim = state_dim

        self.goal_dim_actor = goal_dim
        self.goal_dim_critic = goal_dim + meta_goal_dim

        self.actor = ACModel.Actor(
            state_dim,
            act_dim,
            self.goal_dim_actor,
            init_mu=init_mu,
            init_sigma=init_sigma,
        )
        self.critic = ACModel.Critic(state_dim, self.goal_dim_critic, value_dim)
        self.actor_ob_normalizer = RunningMeanStd(state_dim, clamp=5.0)
        self.critic_ob_normalizer = self.actor_ob_normalizer
        self.ob_normalizer = [self.actor_ob_normalizer]
        if isinstance(self.critic_ob_normalizer, torch.nn.ModuleList):
            self.ob_normalizer.extend(self.critic_ob_normalizer)
        if normalize_value:
            self.value_normalizer = DiagonalPopArt(
                value_dim, self.critic.mlp[-1].weight, self.critic.mlp[-1].bias
            )
        else:
            self.value_normalizer = None

    def observe(self, obs, norm=True):
        if self.goal_dim_critic > 0:
            s = obs[:, : -self.goal_dim_critic]
            g = obs[:, -self.goal_dim_critic :]
        else:
            s = obs
            g = None
        s = s.view(*s.shape[:-1], -1, self.state_dim)
        return [normalizer(s) for normalizer in self.ob_normalizer] if norm else s, g

    def eval_(self, s, seq_end_frame, g, unnorm):
        v = self.critic(s[-1], seq_end_frame, g)
        if unnorm and self.value_normalizer is not None:
            v = self.value_normalizer(v, unnorm=True)
        return v

    def act(self, obs, seq_end_frame, stochastic=None, unnorm=False):
        if stochastic is None:
            stochastic = self.training
        s, g = self.observe(obs)
        pi = self.actor(
            s, seq_end_frame, None if g is None else g[:, : self.goal_dim_actor]
        )
        if stochastic:
            a = pi.sample()
            lp = pi.log_prob(a)
            # if g is not None:
            #     g = g[...,:self.goal_dim_critic]
            return a, self.eval_(s, seq_end_frame, g, unnorm), lp
        else:
            return pi.mean

    def evaluate(self, obs, seq_end_frame, unnorm=False):
        # no meta_goal passed with obs
        # self.goal_dim, self.goal_dim_critic = self.goal_dim_critic, self.goal_dim
        s, g = self.observe(obs)
        # self.goal_dim, self.goal_dim_critic = self.goal_dim_critic, self.goal_dim
        # if g is not None:
        # g = g[...,:self.goal_dim_critic]
        return self.eval_(s, seq_end_frame, g, unnorm)

    def forward(self, obs, seq_end_frame, unnorm=False):
        s, g = self.observe(obs)
        pi = self.actor(
            s, seq_end_frame, None if g is None else g[:, : self.goal_dim_actor]
        )
        return pi, self.eval_(s, seq_end_frame, g, unnorm)


class MapCNN(torch.nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 5, 5),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(128, 256, 3),
            torch.nn.Flatten(),
        )

    def forward(self, m):
        return self.cnn(m)


class AdaptNet(torch.nn.Module):
    def __init__(self, meta_model, g_dim=0):
        super().__init__()
        actor_ob_normalizer = copy.deepcopy(meta_model.actor_ob_normalizer)
        for normalizer in meta_model.ob_normalizer:
            normalizer.reset_counter()
        meta_model.ob_normalizer.insert(0, actor_ob_normalizer)

        meta_policy = meta_model.actor
        for n, p in meta_policy.named_parameters():
            p.requires_grad = False

        input_size = meta_policy.rnn.input_size
        hidden_size = meta_policy.rnn.hidden_size
        num_layers = meta_policy.rnn.num_layers
        batch_first = meta_policy.rnn.batch_first

        self.meta_policy = meta_policy
        self.rnn = meta_policy.rnn.__class__(
            input_size, hidden_size, num_layers, batch_first=batch_first
        )

        self.embed = torch.nn.Linear(
            meta_policy.mlp[0].in_features + g_dim, meta_policy.mlp[0].in_features
        )

        ia_layer = lambda in_dim, out_dim: torch.nn.Linear(in_dim, out_dim)
        self.ia = torch.nn.ModuleList(
            [
                (
                    ia_layer(op.in_features, op.out_features)
                    if isinstance(op, torch.nn.Linear)
                    else torch.nn.Identity()
                )
                for op in meta_policy.mlp
            ]
        )
        for e in self.ia:
            if isinstance(e, torch.nn.Identity):
                continue
            if isinstance(e, torch.nn.Sequential):
                e = e[-1]
            for p in e.parameters():
                torch.nn.init.zeros_(p)

        for p, p_ in zip(self.rnn.parameters(), meta_policy.rnn.parameters()):
            p.data.copy_(p_.data)
        for p in self.embed.parameters():
            torch.nn.init.zeros_(p)

        self.g = None

    def forward(self, s, seq_end_frame, g=None):
        s, s_ = s[0], s[1]

        n_inst = s.size(0)
        if n_inst > self.meta_policy.all_inst.size(0):
            self.meta_policy.all_inst = torch.arange(
                n_inst, dtype=seq_end_frame.dtype, device=seq_end_frame.device
            )
        ind = (
            self.meta_policy.all_inst[:n_inst],
            torch.clip(seq_end_frame, max=s.size(1) - 1),
        )
        s_, _ = self.rnn(s_)
        s_ = s_[ind]
        s, _ = self.meta_policy.rnn(s)
        s = s[ind]

        if g is not None:
            s = torch.cat((s, g), -1)
            if self.g is None:
                s_ = torch.cat((s_, g), -1)
            else:
                s_ = torch.cat((s_, g, self.g), -1)
        elif self.g is not None:
            s_ = torch.cat((s_, self.g), -1)

        if isinstance(self.embed, torch.nn.ModuleList):
            s_ = [embed(s_) for embed in self.embed]
        else:
            s_ = self.embed(s_)
        s = s + s_
        for j, op in enumerate(self.meta_policy.mlp):
            embed = self.ia[j]
            if isinstance(embed, torch.nn.Identity):
                s = op(s)
            else:
                s = op(s) + embed(s)
        mu = self.meta_policy.mu(s)
        sigma = torch.exp(self.meta_policy.log_sigma(s)) + 1e-8
        return torch.distributions.Normal(mu, sigma)


"""
1"pelvis": 
2"torso": 
3"head": 
4"right_upper_arm": 
5"right_lower_arm":
6"right_hand":
7"left_upper_arm":
8"left_lower_arm": 
9"left_hand": 
10"right_thigh":
11"right_shin": 
12"right_foot": 
13"left_thigh": 
14"left_shin":
15"left_foot":
"""


def main_adaptnet():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Configure file used for training. Please refer to files in `config` folder.",
    )
    parser.add_argument(
        "--meta", type=str, default=None, help="Pretrained meta checkpoint file."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint directory or file for training or evaluation.",
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Run visual evaluation."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the target GPU device for model running.",
    )
    settings = parser.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if hasattr(config, "training_params"):
        TRAINING_PARAMS.update(config.training_params)
    if not TRAINING_PARAMS["save_interval"]:
        TRAINING_PARAMS["save_interval"] = TRAINING_PARAMS["max_epochs"]
    training_params = namedtuple("x", TRAINING_PARAMS.keys())(*TRAINING_PARAMS.values())

    if hasattr(config, "discriminators"):
        discriminators = {
            name: envs.DiscriminatorConfig(**prop)
            for name, prop in config.discriminators.items()
        }
    else:
        discriminators = {"_/full": envs.DiscriminatorConfig()}

    if not hasattr(config, "env_params"):
        setattr(config, "env_params", {})

    if settings.test:
        num_envs = 1
    else:
        num_envs = training_params.num_envs

    env = getattr(envs, config.env_cls)(
        num_envs,
        discriminators=discriminators,
        compute_device=settings.device,
        **config.env_params
    )
    if settings.test:
        env.episode_length = 500000

    map_dim = 256 if hasattr(env, "info") and "map" in env.info else 0
    value_dim = len(env.discriminators) + env.rew_dim
    model = ACModel2(
        env.state_dim, env.act_dim, env.goal_dim, value_dim, meta_goal_dim=map_dim
    )
    discriminators = torch.nn.ModuleDict(
        {name: Discriminator(dim) for name, dim in env.disc_dim.items()}
    )
    device = torch.device(settings.device)
    model.to(device)
    discriminators.to(device)

    if settings.meta is not None and os.path.exists(settings.meta):
        if os.path.isdir(settings.meta):
            ckpt = os.path.join(settings.meta, "ckpt")
        else:
            ckpt = settings.meta
            settings.meta = os.path.dirname(ckpt)
        if os.path.exists(ckpt):
            print("Load meta-model from {}".format(ckpt))
            state_dict = torch.load(ckpt, map_location=device)
            pretrained = dict()
            for k, p in state_dict["model"].items():
                if "actor" in k or "actor_ob_normalizer" in k:
                    pretrained[k] = p
            model.load_state_dict(pretrained, strict=False)

    model.discriminators = discriminators
    model.actor = AdaptNet(model, g_dim=map_dim)
    if map_dim:
        model.critic.map = MapCNN()
        model.actor.map = MapCNN()
    model.to(device)

    if settings.test:
        if settings.ckpt is not None and os.path.exists(settings.ckpt):
            if os.path.isdir(settings.ckpt):
                ckpt = os.path.join(settings.ckpt, "ckpt")
            else:
                ckpt = settings.ckpt
                settings.ckpt = os.path.dirname(ckpt)
            if os.path.exists(ckpt):
                print("Load model from {}".format(ckpt))
                state_dict = torch.load(
                    ckpt, map_location=torch.device(settings.device)
                )
                model.load_state_dict(state_dict["model"])
        env.render()
        model.eval()
        env.reset()
        while not env.request_quit:
            obs, info = env.reset_done()
            seq_len = info["ob_seq_lens"]
            if "map" in info:
                m = info["map"]
                obs = torch.cat((obs, model.critic.map(m)), -1)
                model.actor.g = model.actor.map(m)
            actions = model.act(obs, seq_len - 1)
            env.step(actions)

    else:
        print(TRAINING_PARAMS)
        ckpt_dir = settings.ckpt
        if ckpt_dir is not None:
            logger = SummaryWriter(ckpt_dir)
        else:
            logger = None

        optimizer = torch.optim.Adam(
            [
                {"params": model.actor.parameters(), "lr": training_params.actor_lr},
                {"params": model.critic.parameters(), "lr": training_params.critic_lr},
            ]
        )
        ac_parameters = list(model.actor.parameters()) + list(model.critic.parameters())
        disc_optimizer = {
            name: torch.optim.Adam(disc.parameters(), training_params.disc_lr)
            for name, disc in model.discriminators.items()
        }

        buffer = dict(
            s=[], a=[], v=[], lp=[], v_=[], not_done=[], terminate=[], ob_seq_len=[]
        )
        multi_critics = env.reward_weights is not None
        if multi_critics:
            buffer["reward_weights"] = []
        has_goal_reward = env.rew_dim > 0
        if has_goal_reward:
            buffer["r"] = []

        buffer_disc = {
            name: dict(fake=[], real=[], seq_len=[])
            for name in env.discriminators.keys()
        }
        real_losses, fake_losses = {n: [] for n in buffer_disc.keys()}, {
            n: [] for n in buffer_disc.keys()
        }

        BATCH_SIZE = training_params.batch_size
        HORIZON = training_params.horizon
        GAMMA = training_params.gamma
        GAMMA_LAMBDA = training_params.gamma * training_params.lambda_
        OPT_EPOCHS = training_params.opt_epochs

        epoch = 0
        model.eval()
        env.reset()
        tic = time.time()
        while not env.request_quit:
            with torch.no_grad():
                obs, info = env.reset_done()
                seq_len = info["ob_seq_lens"]
                reward_weights = info["reward_weights"]
                actions, values, log_probs = model.act(
                    obs, seq_len - 1, stochastic=True
                )
                obs_, rews, dones, info = env.step(actions)
                log_probs = log_probs.sum(-1, keepdim=True)
                not_done = (~dones).unsqueeze_(-1)
                terminate = info["terminate"]

                fakes = info["disc_obs"]
                reals = info["disc_obs_expert"]
                disc_seq_len = info["disc_seq_len"]

                values_ = model.evaluate(obs_, seq_len)

            buffer["s"].append(obs)
            buffer["a"].append(actions)
            buffer["v"].append(values)
            buffer["lp"].append(log_probs)
            buffer["v_"].append(values_)
            buffer["not_done"].append(not_done)
            buffer["terminate"].append(terminate)
            buffer["ob_seq_len"].append(seq_len)
            if has_goal_reward:
                buffer["r"].append(rews)
            if multi_critics:
                buffer["reward_weights"].append(reward_weights)
            for name, fake in fakes.items():
                buffer_disc[name]["fake"].append(fake)
                buffer_disc[name]["real"].append(reals[name])
                buffer_disc[name]["seq_len"].append(disc_seq_len[name])

            if len(buffer["s"]) == HORIZON:
                with torch.no_grad():
                    disc_data_training = []
                    disc_data_raw = []
                    for name, data in buffer_disc.items():
                        disc = model.discriminators[name]
                        fake = torch.cat(data["fake"])
                        real = torch.cat(data["real"])
                        seq_len = torch.cat(data["seq_len"])
                        end_frame = seq_len - 1
                        disc_data_raw.append((name, disc, fake, end_frame))

                        length = torch.arange(
                            fake.size(1), dtype=end_frame.dtype, device=end_frame.device
                        )
                        mask = length.unsqueeze_(0) <= end_frame.unsqueeze(1)
                        disc.ob_normalizer.update(fake[mask])
                        disc.ob_normalizer.update(real[mask])

                        ob = disc.ob_normalizer(fake)
                        ref = disc.ob_normalizer(real)
                        disc_data_training.append((name, disc, ref, ob, end_frame))

                model.train()
                n_samples = 0
                for name, disc, ref, ob, seq_end_frame_ in disc_data_training:
                    real_loss = real_losses[name]
                    fake_loss = fake_losses[name]
                    opt = disc_optimizer[name]
                    if len(ref) != n_samples:
                        n_samples = len(ref)
                        idx = torch.randperm(n_samples)
                    for batch in range(n_samples // BATCH_SIZE):
                        sample = idx[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
                        r = ref[sample]
                        f = ob[sample]
                        seq_end_frame = seq_end_frame_[sample]

                        score_r = disc(r, seq_end_frame, normalize=False)
                        score_f = disc(f, seq_end_frame, normalize=False)

                        loss_r = torch.nn.functional.relu(1 - score_r).mean()
                        loss_f = torch.nn.functional.relu(1 + score_f).mean()

                        with torch.no_grad():
                            alpha = torch.rand(
                                r.size(0), dtype=r.dtype, device=r.device
                            )
                            alpha = alpha.view(-1, *([1] * (r.ndim - 1)))
                            interp = alpha * r + (1 - alpha) * f
                        interp.requires_grad = True
                        with torch.backends.cudnn.flags(enabled=False):
                            score_interp = disc(interp, seq_end_frame, normalize=False)
                        grad = torch.autograd.grad(
                            score_interp,
                            interp,
                            torch.ones_like(score_interp),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                        gp = (
                            grad.reshape(grad.size(0), -1)
                            .norm(2, dim=1)
                            .sub(1)
                            .square()
                            .mean()
                        )
                        l = loss_f + loss_r + 10 * gp
                        l.backward()
                        opt.step()
                        opt.zero_grad()

                        real_loss.append(score_r.mean().item())
                        fake_loss.append(score_f.mean().item())

                model.eval()
                with torch.no_grad():
                    terminate = torch.cat(buffer["terminate"])
                    if multi_critics:
                        reward_weights = torch.cat(buffer["reward_weights"])
                        rewards = torch.zeros_like(reward_weights)
                    else:
                        reward_weights = None
                        rewards = None
                    for name, disc, ob, seq_end_frame in disc_data_raw:
                        r = disc(ob, seq_end_frame).clamp_(-1, 1).mean(-1, keepdim=True)
                        if rewards is None:
                            rewards = r
                        else:
                            rewards[:, env.discriminators[name].id] = r.squeeze_(-1)
                    if has_goal_reward:
                        rewards_task = torch.cat(buffer["r"])
                        if rewards is None:
                            rewards = rewards_task
                        else:
                            rewards[:, -rewards_task.size(-1) :] = rewards_task
                    else:
                        rewards_task = None
                    rewards[terminate] = training_params.terminate_reward

                    values = torch.cat(buffer["v"])
                    values_ = torch.cat(buffer["v_"])
                    if model.value_normalizer is not None:
                        values = model.value_normalizer(values, unnorm=True)
                        values_ = model.value_normalizer(values_, unnorm=True)
                    values_[terminate] = 0
                    rewards = rewards.view(HORIZON, -1, rewards.size(-1))
                    values = values.view(HORIZON, -1, values.size(-1))
                    values_ = values_.view(HORIZON, -1, values_.size(-1))

                    not_done = buffer["not_done"]
                    advantages = (rewards - values).add_(values_, alpha=GAMMA)
                    for t in reversed(range(HORIZON - 1)):
                        advantages[t].add_(
                            advantages[t + 1] * not_done[t], alpha=GAMMA_LAMBDA
                        )

                    advantages = advantages.view(-1, advantages.size(-1))
                    returns = advantages + values.view(-1, advantages.size(-1))

                    log_probs = torch.cat(buffer["lp"])
                    actions = torch.cat(buffer["a"])
                    states = torch.cat(buffer["s"])
                    ob_seq_lens = torch.cat(buffer["ob_seq_len"])
                    ob_seq_end_frames = ob_seq_lens - 1

                    sigma, mu = torch.std_mean(advantages, dim=0, unbiased=True)
                    advantages = (advantages - mu) / (
                        sigma + 1e-8
                    )  # (HORIZON x N_ENVS) x N_DISC

                    length = torch.arange(
                        env.ob_horizon,
                        dtype=ob_seq_lens.dtype,
                        device=ob_seq_lens.device,
                    )
                    mask = length.unsqueeze_(0) < ob_seq_lens.unsqueeze(1)
                    states_raw = model.observe(states, norm=False)[0]
                    # here
                    # import ipdb;ipdb.set_trace()
                    for i in range(len(model.ob_normalizer)):
                        model.ob_normalizer[i].update(states_raw[mask])
                    if model.value_normalizer is not None:
                        model.value_normalizer.update(returns)
                        returns = model.value_normalizer(returns)
                    if multi_critics:
                        advantages = advantages.mul_(reward_weights)

                # import ipdb;ipdb.set_trace()
                n_samples = advantages.size(0)
                epoch += 1
                model.train()
                policy_loss, value_loss = [], []
                for _ in range(OPT_EPOCHS):
                    idx = torch.randperm(n_samples)
                    for batch in range(n_samples // BATCH_SIZE):
                        sample = idx[BATCH_SIZE * batch : BATCH_SIZE * (batch + 1)]
                        s = states[sample]
                        a = actions[sample]
                        lp = log_probs[sample]
                        adv = advantages[sample]
                        v_t = returns[sample]
                        end_frame = ob_seq_end_frames[sample]

                        pi_, v_ = model(s, end_frame)
                        lp_ = pi_.log_prob(a).sum(-1, keepdim=True)

                        ratio = torch.exp(lp_ - lp)
                        clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                        pg_loss = (
                            -torch.min(adv * ratio, adv * clipped_ratio).sum(-1).mean()
                        )
                        vf_loss = (v_ - v_t).square().mean()
                        loss = pg_loss + 0.5 * vf_loss

                        # import ipdb;ipdb.set_trace()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(ac_parameters, 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        policy_loss.append(pg_loss.item())
                        value_loss.append(vf_loss.item())
                model.eval()
                for v in buffer.values():
                    v.clear()
                for buf in buffer_disc.values():
                    for v in buf.values():
                        v.clear()

                lifetime = env.lifetime.to(torch.float32).mean().item()
                policy_loss, value_loss = np.mean(policy_loss), np.mean(value_loss)
                if multi_critics:
                    rewards = rewards.view(*reward_weights.shape)
                    r = rewards.mean(0).cpu().tolist()
                    # reward_tot = (rewards * reward_weights).sum(-1, keepdims=True).mean(0).item()
                else:
                    r = rewards.view(-1, rewards.size(-1)).mean(0).cpu().tolist()
                if rewards_task is not None:
                    rewards_task = rewards_task.mean(0).cpu().tolist()
                # import ipdb;ipdb.set_trace()
                print(
                    "Epoch: {}, Loss: {:.4f}/{:.4f}, Reward: {}, Lifetime: {:.4f} -- train comsume time: {:.4f}s".format(
                        epoch,
                        policy_loss,
                        value_loss,
                        "/".join(list(map("{:.4f}".format, r))),
                        lifetime,
                        time.time() - tic,
                    )
                )
                if logger is not None:
                    logger.add_scalar("train/lifetime", lifetime, epoch)
                    logger.add_scalar("train/reward", np.mean(r), epoch)
                    logger.add_scalar("train/loss_policy", policy_loss, epoch)
                    logger.add_scalar("train/loss_value", value_loss, epoch)
                    for name, r_loss in real_losses.items():
                        if r_loss:
                            logger.add_scalar(
                                "score_real/{}".format(name),
                                sum(r_loss) / len(r_loss),
                                epoch,
                            )
                    for name, f_loss in fake_losses.items():
                        if f_loss:
                            logger.add_scalar(
                                "score_fake/{}".format(name),
                                sum(f_loss) / len(f_loss),
                                epoch,
                            )
                    if rewards_task is not None:
                        for i in range(len(rewards_task)):
                            logger.add_scalar(
                                "train/task_reward_{}".format(i), rewards_task[i], epoch
                            )
                for v in real_losses.values():
                    v.clear()
                for v in fake_losses.values():
                    v.clear()

                if ckpt_dir is not None:
                    state = None
                    if epoch % 50 == 0:
                        state = dict(model=model.state_dict())
                        torch.save(state, os.path.join(ckpt_dir, "ckpt"))
                    if epoch % training_params.save_interval == 0:
                        if state is None:
                            state = dict(model=model.state_dict())
                        torch.save(
                            state, os.path.join(ckpt_dir, "ckpt-{}".format(epoch))
                        )
                    if epoch >= training_params.max_epochs:
                        exit()
                tic = time.time()


def main_composite():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Configure file used for training. Please refer to files in `config` folder.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint directory or file for training or evaluation.",
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Run visual evaluation."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the target GPU device for model running.",
    )
    settings = parser.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if hasattr(config, "training_params"):
        TRAINING_PARAMS.update(config.training_params)
    if not TRAINING_PARAMS["save_interval"]:
        TRAINING_PARAMS["save_interval"] = TRAINING_PARAMS["max_epochs"]
    training_params = namedtuple("x", TRAINING_PARAMS.keys())(*TRAINING_PARAMS.values())

    if hasattr(config, "discriminators"):
        discriminators = {
            name: env.DiscriminatorConfig(**prop)
            for name, prop in config.discriminators.items()
        }
    else:
        discriminators = {"_/full": env.DiscriminatorConfig()}

    if hasattr(config, "env_cls"):
        env_cls = getattr(env, config.env_cls)
    else:
        env_cls = env.ICCGANHumanoid
    print(env_cls, config.env_params)

    if settings.test:
        num_envs = 1
    else:
        num_envs = training_params.num_envs
        if settings.ckpt:
            if os.path.isfile(settings.ckpt) or os.path.exists(
                os.path.join(settings.ckpt, "ckpt")
            ):
                raise ValueError(
                    "Checkpoint folder {} exists. Add `--test` option to run test with an existing checkpoint file".format(
                        settings.ckpt
                    )
                )
            import shutil, sys

            os.makedirs(settings.ckpt, exist_ok=True)
            shutil.copy(settings.config, settings.ckpt)
            with open(
                os.path.join(settings.ckpt, "command_{}.txt".format(time.time())), "w"
            ) as f:
                f.write(" ".join(sys.argv))

    env = env_cls(
        num_envs,
        discriminators=discriminators,
        compute_device=settings.device,
        **config.env_params
    )
    if settings.test:
        env.episode_length = 500000

    value_dim = len(env.discriminators) + env.rew_dim
    model = ACModel(env.state_dim, env.act_dim, env.goal_dim, value_dim)
    discriminators = torch.nn.ModuleDict(
        {name: Discriminator(dim) for name, dim in env.disc_dim.items()}
    )
    device = torch.device(settings.device)
    model.to(device)
    discriminators.to(device)
    model.discriminators = discriminators

    if settings.test:
        if settings.ckpt is not None and os.path.exists(settings.ckpt):
            if os.path.isdir(settings.ckpt):
                ckpt = os.path.join(settings.ckpt, "ckpt")
            else:
                ckpt = settings.ckpt
                settings.ckpt = os.path.dirname(ckpt)
            if os.path.exists(ckpt):
                print("Load model from {}".format(ckpt))
                state_dict = torch.load(
                    ckpt, map_location=torch.device(settings.device)
                )
                model.load_state_dict(state_dict["model"])
        env.render()
        model.eval()
        env.reset()
        while not env.request_quit:
            obs, info = env.reset_done()
            seq_len = info["ob_seq_lens"]
            actions = model.act(obs, seq_len - 1)
            env.step(actions)

    else:
        print(TRAINING_PARAMS)
        ckpt_dir = settings.ckpt
        if ckpt_dir is not None:
            logger = SummaryWriter(ckpt_dir)
        else:
            logger = None

        optimizer = torch.optim.Adam(
            [
                {"params": model.actor.parameters(), "lr": training_params.actor_lr},
                {"params": model.critic.parameters(), "lr": training_params.critic_lr},
            ]
        )
        ac_parameters = list(model.actor.parameters()) + list(model.critic.parameters())
        disc_optimizer = {
            name: torch.optim.Adam(disc.parameters(), training_params.disc_lr)
            for name, disc in model.discriminators.items()
        }

        buffer = dict(
            s=[], a=[], v=[], lp=[], v_=[], not_done=[], terminate=[], ob_seq_len=[]
        )
        multi_critics = env.reward_weights is not None
        if multi_critics:
            buffer["reward_weights"] = []
        has_goal_reward = env.rew_dim > 0
        if has_goal_reward:
            buffer["r"] = []

        buffer_disc = {
            name: dict(fake=[], real=[], seq_len=[])
            for name in env.discriminators.keys()
        }
        real_losses, fake_losses = {n: [] for n in buffer_disc.keys()}, {
            n: [] for n in buffer_disc.keys()
        }

        BATCH_SIZE = training_params.batch_size
        HORIZON = training_params.horizon
        GAMMA = training_params.gamma
        GAMMA_LAMBDA = training_params.gamma * training_params.lambda_
        OPT_EPOCHS = training_params.opt_epochs

        epoch = 0
        model.eval()
        env.reset()
        tic = time.time()
        while not env.request_quit:
            with torch.no_grad():
                obs, info = env.reset_done()
                seq_len = info["ob_seq_lens"]
                reward_weights = info["reward_weights"]
                actions, values, log_probs = model.act(
                    obs, seq_len - 1, stochastic=True
                )
                obs_, rews, dones, info = env.step(actions)
                log_probs = log_probs.sum(-1, keepdim=True)
                not_done = (~dones).unsqueeze_(-1)
                terminate = info["terminate"]

                fakes = info["disc_obs"]
                reals = info["disc_obs_expert"]
                disc_seq_len = info["disc_seq_len"]

                values_ = model.evaluate(obs_, seq_len)

            buffer["s"].append(obs)
            buffer["a"].append(actions)
            buffer["v"].append(values)
            buffer["lp"].append(log_probs)
            buffer["v_"].append(values_)
            buffer["not_done"].append(not_done)
            buffer["terminate"].append(terminate)
            buffer["ob_seq_len"].append(seq_len)
            if has_goal_reward:
                buffer["r"].append(rews)
            if multi_critics:
                buffer["reward_weights"].append(reward_weights)
            for name, fake in fakes.items():
                buffer_disc[name]["fake"].append(fake)
                buffer_disc[name]["real"].append(reals[name])
                buffer_disc[name]["seq_len"].append(disc_seq_len[name])

            if len(buffer["s"]) == HORIZON:
                with torch.no_grad():
                    disc_data_training = []
                    disc_data_raw = []
                    for name, data in buffer_disc.items():
                        disc = model.discriminators[name]
                        fake = torch.cat(data["fake"])
                        real = torch.cat(data["real"])
                        seq_len = torch.cat(data["seq_len"])
                        end_frame = seq_len - 1
                        disc_data_raw.append((name, disc, fake, end_frame))

                        length = torch.arange(
                            fake.size(1), dtype=end_frame.dtype, device=end_frame.device
                        )
                        mask = length.unsqueeze_(0) <= end_frame.unsqueeze(1)
                        disc.ob_normalizer.update(fake[mask])
                        disc.ob_normalizer.update(real[mask])

                        ob = disc.ob_normalizer(fake)
                        ref = disc.ob_normalizer(real)
                        disc_data_training.append((name, disc, ref, ob, end_frame))

                model.train()
                n_samples = 0
                for name, disc, ref, ob, seq_end_frame_ in disc_data_training:
                    real_loss = real_losses[name]
                    fake_loss = fake_losses[name]
                    opt = disc_optimizer[name]
                    if len(ref) != n_samples:
                        n_samples = len(ref)
                        idx = torch.randperm(n_samples)
                    for batch in range(n_samples // BATCH_SIZE):
                        sample = idx[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
                        r = ref[sample]
                        f = ob[sample]
                        seq_end_frame = seq_end_frame_[sample]

                        score_r = disc(r, seq_end_frame, normalize=False)
                        score_f = disc(f, seq_end_frame, normalize=False)

                        loss_r = torch.nn.functional.relu(1 - score_r).mean()
                        loss_f = torch.nn.functional.relu(1 + score_f).mean()

                        with torch.no_grad():
                            alpha = torch.rand(
                                r.size(0), dtype=r.dtype, device=r.device
                            )
                            alpha = alpha.view(-1, *([1] * (r.ndim - 1)))
                            interp = alpha * r + (1 - alpha) * f
                        interp.requires_grad = True
                        with torch.backends.cudnn.flags(enabled=False):
                            score_interp = disc(interp, seq_end_frame, normalize=False)
                        grad = torch.autograd.grad(
                            score_interp,
                            interp,
                            torch.ones_like(score_interp),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                        gp = (
                            grad.reshape(grad.size(0), -1)
                            .norm(2, dim=1)
                            .sub(1)
                            .square()
                            .mean()
                        )
                        l = loss_f + loss_r + 10 * gp
                        l.backward()
                        opt.step()
                        opt.zero_grad()

                        real_loss.append(score_r.mean().item())
                        fake_loss.append(score_f.mean().item())

                model.eval()
                with torch.no_grad():
                    terminate = torch.cat(buffer["terminate"])
                    if multi_critics:
                        reward_weights = torch.cat(buffer["reward_weights"])
                        rewards = torch.zeros_like(reward_weights)
                    else:
                        reward_weights = None
                        rewards = None
                    for name, disc, ob, seq_end_frame in disc_data_raw:
                        r = disc(ob, seq_end_frame).clamp_(-1, 1).mean(-1, keepdim=True)
                        if rewards is None:
                            rewards = r
                        else:
                            rewards[:, env.discriminators[name].id] = r.squeeze_(-1)
                    if has_goal_reward:
                        rewards_task = torch.cat(buffer["r"])
                        if rewards is None:
                            rewards = rewards_task
                        else:
                            rewards[:, -rewards_task.size(-1) :] = rewards_task
                    else:
                        rewards_task = None
                    rewards[terminate] = training_params.terminate_reward

                    values = torch.cat(buffer["v"])
                    values_ = torch.cat(buffer["v_"])
                    if model.value_normalizer is not None:
                        values = model.value_normalizer(values, unnorm=True)
                        values_ = model.value_normalizer(values_, unnorm=True)
                    values_[terminate] = 0
                    rewards = rewards.view(HORIZON, -1, rewards.size(-1))
                    values = values.view(HORIZON, -1, values.size(-1))
                    values_ = values_.view(HORIZON, -1, values_.size(-1))

                    not_done = buffer["not_done"]
                    advantages = (rewards - values).add_(values_, alpha=GAMMA)
                    for t in reversed(range(HORIZON - 1)):
                        advantages[t].add_(
                            advantages[t + 1] * not_done[t], alpha=GAMMA_LAMBDA
                        )

                    advantages = advantages.view(-1, advantages.size(-1))
                    returns = advantages + values.view(-1, advantages.size(-1))

                    log_probs = torch.cat(buffer["lp"])
                    actions = torch.cat(buffer["a"])
                    states = torch.cat(buffer["s"])
                    ob_seq_lens = torch.cat(buffer["ob_seq_len"])
                    ob_seq_end_frames = ob_seq_lens - 1

                    sigma, mu = torch.std_mean(advantages, dim=0, unbiased=True)
                    advantages = (advantages - mu) / (
                        sigma + 1e-8
                    )  # (HORIZON x N_ENVS) x N_DISC

                    length = torch.arange(
                        env.ob_horizon,
                        dtype=ob_seq_lens.dtype,
                        device=ob_seq_lens.device,
                    )
                    mask = length.unsqueeze_(0) < ob_seq_lens.unsqueeze(1)
                    states_raw = model.observe(states, norm=False)[0]
                    model.ob_normalizer.update(states_raw[mask])
                    if model.value_normalizer is not None:
                        model.value_normalizer.update(returns)
                        returns = model.value_normalizer(returns)
                    if multi_critics:
                        advantages = advantages.mul_(reward_weights)

                n_samples = advantages.size(0)
                epoch += 1
                model.train()
                policy_loss, value_loss = [], []
                for _ in range(OPT_EPOCHS):
                    idx = torch.randperm(n_samples)
                    for batch in range(n_samples // BATCH_SIZE):
                        sample = idx[BATCH_SIZE * batch : BATCH_SIZE * (batch + 1)]
                        s = states[sample]
                        a = actions[sample]
                        lp = log_probs[sample]
                        adv = advantages[sample]
                        v_t = returns[sample]
                        end_frame = ob_seq_end_frames[sample]

                        pi_, v_ = model(s, end_frame)
                        lp_ = pi_.log_prob(a).sum(-1, keepdim=True)

                        ratio = torch.exp(lp_ - lp)
                        clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                        pg_loss = (
                            -torch.min(adv * ratio, adv * clipped_ratio).sum(-1).mean()
                        )
                        vf_loss = (v_ - v_t).square().mean()

                        loss = pg_loss + 0.5 * vf_loss

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(ac_parameters, 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        policy_loss.append(pg_loss.item())
                        value_loss.append(vf_loss.item())
                model.eval()
                for v in buffer.values():
                    v.clear()
                for buf in buffer_disc.values():
                    for v in buf.values():
                        v.clear()

                lifetime = env.lifetime.to(torch.float32).mean().item()
                policy_loss, value_loss = np.mean(policy_loss), np.mean(value_loss)
                if multi_critics:
                    rewards = rewards.view(*reward_weights.shape)
                    r = rewards.mean(0).cpu().tolist()
                    # reward_tot = (rewards * reward_weights).sum(-1, keepdims=True).mean(0).item()
                else:
                    r = rewards.view(-1, rewards.size(-1)).mean(0).cpu().tolist()
                if rewards_task is not None:
                    rewards_task = rewards_task.mean(0).cpu().tolist()
                print(
                    "Epoch: {}, Loss: {:.4f}/{:.4f}, Reward: {}, Lifetime: {:.4f} -- {:.4f}s".format(
                        epoch,
                        policy_loss,
                        value_loss,
                        "/".join(list(map("{:.4f}".format, r))),
                        lifetime,
                        time.time() - tic,
                    )
                )
                if logger is not None:
                    logger.add_scalar("train/lifetime", lifetime, epoch)
                    logger.add_scalar("train/reward", np.mean(r), epoch)
                    logger.add_scalar("train/loss_policy", policy_loss, epoch)
                    logger.add_scalar("train/loss_value", value_loss, epoch)
                    for name, r_loss in real_losses.items():
                        if r_loss:
                            logger.add_scalar(
                                "score_real/{}".format(name),
                                sum(r_loss) / len(r_loss),
                                epoch,
                            )
                    for name, f_loss in fake_losses.items():
                        if f_loss:
                            logger.add_scalar(
                                "score_fake/{}".format(name),
                                sum(f_loss) / len(f_loss),
                                epoch,
                            )
                    if rewards_task is not None:
                        for i in range(len(rewards_task)):
                            logger.add_scalar(
                                "train/task_reward_{}".format(i), rewards_task[i], epoch
                            )
                for v in real_losses.values():
                    v.clear()
                for v in fake_losses.values():
                    v.clear()

                if ckpt_dir is not None:
                    state = None
                    if epoch % 50 == 0:
                        state = dict(model=model.state_dict())
                        torch.save(state, os.path.join(ckpt_dir, "ckpt"))
                    if epoch % training_params.save_interval == 0:
                        if state is None:
                            state = dict(model=model.state_dict())
                        torch.save(
                            state, os.path.join(ckpt_dir, "ckpt-{}".format(epoch))
                        )
                    if epoch >= training_params.max_epochs:
                        exit()
                tic = time.time()


if __name__ == "__main__":
    # main_composite()
    main_adaptnet()
