# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from phc.utils.config import (
    set_np_formatting,
    set_seed,
    get_args,
    parse_sim_params,
    load_cfg,
)
from phc.utils.parse_task import parse_task

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from phc.utils.flags import flags

import numpy as np
import copy
import torch
import wandb

from learning import im_amp
from learning import im_amp_players
from learning import amp_agent
from learning import amp_players
from learning import amp_models
from learning import amp_network_builder
from learning import amp_network_mcp_builder
from learning import amp_network_pnn_builder


from env.tasks import humanoid_amp_task

args = None
cfg = None
cfg_train = None


def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train["params"]["config"].get("multi_gpu", False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train["params"]["seed"] = cfg_train["params"]["seed"] + rank

        args.device = "cuda"
        args.device_id = rank
        args.rl_device = "cuda:" + str(rank)

        cfg["rank"] = rank
        cfg["rl_device"] = "cuda:" + str(rank)

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print(env.num_envs)
    print(env.num_actions)
    print(env.num_obs)
    print(env.num_states)

    frames = kwargs.pop("frames", 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):

    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(
            1, self.algo.games_to_track
        ).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and "consecutive_successes" in infos:
                cons_successes = infos["consecutive_successes"].clone()
                self.consecutive_successes.update(
                    cons_successes.to(self.algo.ppo_device)
                )
            if self.use_successes and "successes" in infos:
                successes = infos["successes"].clone()
                self.consecutive_successes.update(
                    successes[done_indices].to(self.algo.ppo_device)
                )
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar(
                "successes/consecutive_successes/mean", mean_con_successes, frame
            )
            self.writer.add_scalar(
                "successes/consecutive_successes/iter", mean_con_successes, epoch_num
            )
            self.writer.add_scalar(
                "successes/consecutive_successes/time", mean_con_successes, total_time
            )
        return


class RLGPUEnv(vecenv.IVecEnv):

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](
            **kwargs
        )
        self.use_global_obs = self.env.num_states > 0

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space
        info["amp_observation_space"] = self.env.amp_observation_space

        info["enc_amp_observation_space"] = self.env.enc_amp_observation_space

        if isinstance(self.env.task, humanoid_amp_task.HumanoidAMPTask):
            info["task_obs_size"] = self.env.task.get_task_obs_size()
        else:
            info["task_obs_size"] = 0

        if self.use_global_obs:
            info["state_space"] = self.env.state_space
            print(info["action_space"], info["observation_space"], info["state_space"])
        else:
            print(info["action_space"], info["observation_space"])

        return info


vecenv.register(
    "RLGPU",
    lambda config_name, num_actors, **kwargs: RLGPUEnv(
        config_name, num_actors, **kwargs
    ),
)
env_configurations.register(
    "rlgpu",
    {
        "env_creator": lambda **kwargs: create_rlgpu_env(**kwargs),
        "vecenv_type": "RLGPU",
    },
)


def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)
    runner.player_factory.register_builder(
        "amp_discrete", lambda **kwargs: amp_players.AMPPlayerDiscrete(**kwargs)
    )

    runner.algo_factory.register_builder(
        "amp", lambda **kwargs: amp_agent.AMPAgent(**kwargs)
    )
    runner.player_factory.register_builder(
        "amp", lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs)
    )

    runner.model_builder.model_factory.register_builder(
        "amp", lambda network, **kwargs: amp_models.ModelAMPContinuous(network)
    )
    runner.model_builder.network_factory.register_builder(
        "amp", lambda **kwargs: amp_network_builder.AMPBuilder()
    )
    runner.model_builder.network_factory.register_builder(
        "amp_mcp", lambda **kwargs: amp_network_mcp_builder.AMPMCPBuilder()
    )
    runner.model_builder.network_factory.register_builder(
        "amp_pnn", lambda **kwargs: amp_network_pnn_builder.AMPPNNBuilder()
    )

    runner.algo_factory.register_builder(
        "im_amp", lambda **kwargs: im_amp.IMAmpAgent(**kwargs)
    )
    runner.player_factory.register_builder(
        "im_amp", lambda **kwargs: im_amp_players.IMAMPPlayerContinuous(**kwargs)
    )

    return runner


def main():
    global args
    global cfg
    global cfg_train

    set_np_formatting()
    args = get_args()
    cfg_env_name = args.cfg_env.split("/")[-1].split(".")[0]

    args.logdir = args.network_path
    cfg, cfg_train, logdir = load_cfg(args)
    (
        flags.debug,
        flags.follow,
        flags.fixed,
        flags.divide_group,
        flags.no_collision_check,
        flags.fixed_path,
        flags.real_path,
        flags.small_terrain,
        flags.show_traj,
        flags.server_mode,
        flags.slow,
        flags.real_traj,
        flags.im_eval,
        flags.no_virtual_display,
        flags.render_o3d,
    ) = (
        args.debug,
        args.follow,
        False,
        False,
        False,
        False,
        False,
        args.small_terrain,
        True,
        args.server_mode,
        False,
        False,
        args.im_eval,
        args.no_virtual_display,
        args.render_o3d,
    )

    flags.add_proj = args.add_proj
    flags.has_eval = args.has_eval
    flags.trigger_input = False
    flags.demo = args.demo

    if args.server_mode:
        flags.follow = args.follow = True
        flags.fixed = args.fixed = True
        flags.no_collision_check = True
        flags.show_traj = True
        cfg["env"]["episodeLength"] = 99999999999999

    if args.test and not flags.small_terrain:
        cfg["env"]["episodeLength"] = 99999999999999

    if args.real_traj:
        cfg["env"]["episodeLength"] = 99999999999999
        flags.real_traj = True

    project_name = cfg.get("project_name", "egoquest")
    if (not args.no_log) and (not args.test) and (not args.debug):
        wandb.init(
            project=project_name,
            resume=not args.resume_str is None,
            id=args.resume_str,
            notes=cfg.get("notes", "no notes"),
        )
        wandb.config.update(cfg, allow_val_change=True)
        wandb.run.name = cfg_env_name
        wandb.run.save()

    cfg_train["params"]["seed"] = set_seed(
        cfg_train["params"].get("seed", -1),
        cfg_train["params"].get("torch_deterministic", False),
    )

    if args.horovod:
        cfg_train["params"]["config"]["multi_gpu"] = args.horovod

    if args.horizon_length != -1:
        cfg_train["params"]["config"]["horizon_length"] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train["params"]["config"]["minibatch_size"] = args.minibatch_size

    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file
    flags.test = args.test

    # Create default directories for weights and statistics
    cfg_train["params"]["config"]["network_path"] = args.network_path
    args.log_path = osp.join(args.log_path, cfg["name"], cfg_env_name)
    cfg_train["params"]["config"]["log_path"] = args.log_path
    cfg_train["params"]["config"]["train_dir"] = args.log_path

    os.makedirs(args.network_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    vargs = vars(args)

    algo_observer = RLGPUAlgoObserver()

    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)

    return


if __name__ == "__main__":
    main()

"""
usage: run.py [-h] [--sim_device SIM_DEVICE] [--pipeline PIPELINE] [--graphics_device_id GRAPHICS_DEVICE_ID] [--flex | --physx] [--num_threads NUM_THREADS]
              [--subscenes SUBSCENES] [--slices SLICES] [--test] [--debug] [--play] [--epoch EPOCH] [--checkpoint CHECKPOINT] [--headless] [--horovod] [--task TASK]
              [--task_type TASK_TYPE] [--rl_device RL_DEVICE] [--logdir LOGDIR] [--experiment EXPERIMENT] [--metadata] [--cfg_env CFG_ENV] [--cfg_train CFG_TRAIN]
              [--motion_file MOTION_FILE] [--num_envs NUM_ENVS] [--episode_length EPISODE_LENGTH] [--seed SEED] [--max_iterations MAX_ITERATIONS]
              [--horizon_length HORIZON_LENGTH] [--minibatch_size MINIBATCH_SIZE] [--randomize] [--torch_deterministic] [--network_path NETWORK_PATH] [--log_path LOG_PATH]
              [--llc_checkpoint LLC_CHECKPOINT] [--no_log] [--resume_str RESUME_STR] [--follow] [--real_traj] [--show_sensors] [--small_terrain] [--server_mode] [--add_proj]
              [--im_eval] [--has_eval] [--no_virtual_display] [--render_o3d] [--demo]

RL Policy

optional arguments:
  -h, --help            show this help message and exit
  --sim_device SIM_DEVICE
                        Physics Device in PyTorch-like syntax
  --pipeline PIPELINE   Tensor API pipeline (cpu/gpu)
  --graphics_device_id GRAPHICS_DEVICE_ID
                        Graphics Device ID
  --flex                Use FleX for physics
  --physx               Use PhysX for physics
  --num_threads NUM_THREADS
                        Number of cores used by PhysX
  --subscenes SUBSCENES
                        Number of PhysX subscenes to simulate in parallel
  --slices SLICES       Number of client threads that process env slices
  --test                Run trained policy, no training
  --debug               Debugging, no training and no logging
  --play                Run trained policy, the same as test, can be used only by rl_games RL library
  --epoch EPOCH         Resume training or start testing from a checkpoint
  --checkpoint CHECKPOINT
                        Path to the saved weights, only for rl_games RL library
  --headless            Force display off at all times
  --horovod             Use horovod for multi-gpu training, have effect only with rl_games RL library
  --task TASK           Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity
  --task_type TASK_TYPE
                        Choose Python or C++
  --rl_device RL_DEVICE
                        Choose CPU or GPU device for inferencing policy network
  --logdir LOGDIR
  --experiment EXPERIMENT
                        Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be
                        added to the name
  --metadata            Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by
                        user
  --cfg_env CFG_ENV     Environment configuration file (.yaml)
  --cfg_train CFG_TRAIN
                        Training configuration file (.yaml)
  --motion_file MOTION_FILE
                        Specify reference motion file
  --num_envs NUM_ENVS   Number of environments to create - override config file
  --episode_length EPISODE_LENGTH
                        Episode length, by default is read from yaml config
  --seed SEED           Random seed
  --max_iterations MAX_ITERATIONS
                        Set a maximum number of training iterations
  --horizon_length HORIZON_LENGTH
                        Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings.
  --minibatch_size MINIBATCH_SIZE
                        Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings.
  --randomize           Apply physics domain randomization
  --torch_deterministic
                        Apply additional PyTorch settings for more deterministic behaviour
  --network_path NETWORK_PATH
                        Specify network output directory
  --log_path LOG_PATH   Specify log directory
  --llc_checkpoint LLC_CHECKPOINT
                        Path to the saved weights for the low-level controller of an HRL agent.
  --no_log              No wandb logging
  --resume_str RESUME_STR
                        Resuming training from a specific logging instance
  --follow              Follow Humanoid
  --real_traj           load real_traj
  --show_sensors        load real data mesh
  --small_terrain       load real data mesh
  --server_mode         load real data mesh
  --add_proj            adding small projectiiles or not
  --im_eval             Eval imitation
  --has_eval            Eval during training or not
  --no_virtual_display  Disable virtual display
  --render_o3d          Disable virtual display
  --demo                No SMPL_robot dependency
"""
