# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
import sys
sys.path.append(os.path.dirname(__file__))
# sys.path.append("/home/lithiumice/PHC/third-party")
# sys.path.insert(0,"/home/lithiumice/PHC/third-party/vid2player3d")

from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch
from isaacgym.torch_utils import *


from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from rl_games.torch_runner import Runner
from rl_games.algos_torch import network_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from gym import spaces
import numpy as np

sys.path.insert(0, '/root/apdcephfs/private_wallyliang/PLANT/motion_imitator/third-party')
from embodied_pose.utils import torch_utils
from embodied_pose.utils.torch_transform import heading_to_vec, rotation_matrix_to_angle_axis, rotation_matrix_to_quaternion, rot6d_to_rotmat
from embodied_pose.utils.hybrik import batch_inverse_kinematics_transform_naive, batch_inverse_kinematics_transform
from embodied_pose.utils.config import set_np_formatting, get_args, parse_sim_params, load_cfg
from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as smpl_joint_names

from humanoid_smpl_im import HumanoidSMPLIM
from humanoid_smpl_im import remove_base_rot, compute_humanoid_observations_imitation
from im_agent import ImitatorAgent
from im_player import ImitatorPlayer
import torch
import torch.nn as nn


DISC_LOGIT_INIT_SCALE = 1.0

mujoco_joint_names = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
    'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
    'R_Elbow', 'R_Wrist', 'R_Hand'
]
smpl_2_mujoco = [smpl_joint_names.index(q) for q in mujoco_joint_names]
mujoco_2_smpl = [mujoco_joint_names.index(q) for q in smpl_joint_names]




# VecEnv Wrapper for RL training
class VecTask():
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = rl_device

        print("RL device: ", rl_device)

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

# C++ CPU Class
class VecTaskCPU(VecTask):
    def __init__(self, task, rl_device, sync_frame_time=False, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations=clip_observations, clip_actions=clip_actions)
        self.sync_frame_time = sync_frame_time

    def step(self, actions):
        actions = actions.cpu().numpy()
        self.task.render(self.sync_frame_time)

        obs, rewards, resets, extras = self.task.step(np.clip(actions, -self.clip_actions, self.clip_actions))

        return (to_torch(np.clip(obs, -self.clip_obs, self.clip_obs), dtype=torch.float, device=self.rl_device),
                to_torch(rewards, dtype=torch.float, device=self.rl_device),
                to_torch(resets, dtype=torch.uint8, device=self.rl_device), [])

    def reset(self):
        actions = 0.01 * (1 - 2 * np.random.rand(self.num_envs, self.num_actions)).astype('f')

        # step the simulator
        obs, rewards, resets, extras = self.task.step(actions)

        return to_torch(np.clip(obs, -self.clip_obs, self.clip_obs), dtype=torch.float, device=self.rl_device)

# C++ GPU Class
class VecTaskGPU(VecTask):
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations=clip_observations, clip_actions=clip_actions)

        self.obs_tensor = gymtorch.wrap_tensor(self.task.obs_tensor, counts=(self.task.num_envs, self.task.num_obs))
        self.rewards_tensor = gymtorch.wrap_tensor(self.task.rewards_tensor, counts=(self.task.num_envs,))
        self.resets_tensor = gymtorch.wrap_tensor(self.task.resets_tensor, counts=(self.task.num_envs,))

    def step(self, actions):
        self.task.render(False)
        actions_clipped = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        actions_tensor = gymtorch.unwrap_tensor(actions_clipped)

        self.task.step(actions_tensor)

        return torch.clamp(self.obs_tensor, -self.clip_obs, self.clip_obs), self.rewards_tensor, self.resets_tensor, []

    def reset(self):
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions], dtype=torch.float32, device=self.rl_device))
        actions_tensor = gymtorch.unwrap_tensor(actions)

        # step the simulator
        self.task.step(actions_tensor)

        return torch.clamp(self.obs_tensor, -self.clip_obs, self.clip_obs)

# Python CPU/GPU Class
class VecTaskPython(VecTask):

    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self.task.step(actions_tensor)

        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device), self.task.rew_buf.to(self.rl_device), self.task.reset_buf.to(self.rl_device), self.task.extras

    def reset(self):
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions], dtype=torch.float32, device=self.rl_device))

        # step the simulator
        self.task.step(actions)

        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

class VecTaskCPUWrapper(VecTaskCPU):
    def __init__(self, task, rl_device, sync_frame_time=False, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, sync_frame_time, clip_observations, clip_actions)
        return

class VecTaskGPUWrapper(VecTaskGPU):
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations, clip_actions)
        return

class VecTaskPythonWrapper(VecTaskPython):
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations, clip_actions)

    def reset(self, env_ids=None):
        self.task.reset(env_ids)
        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)



class RunningNorm(nn.Module):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, dim, demean=True, destd=True, clip=5.0):
        super().__init__()
        self.dim = dim
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.register_buffer('n', torch.tensor(0, dtype=torch.long))
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.zeros(dim))
        self.register_buffer('std', torch.zeros(dim))

    def update(self, x):
        var_x, mean_x = torch.var_mean(x, dim=0, unbiased=False)
        m = x.shape[0]
        w = self.n.to(x.dtype) / (m + self.n).to(x.dtype)
        self.var[:] = w * self.var + (1 - w) * var_x + w * (1 - w) * (mean_x - self.mean).pow(2)
        self.mean[:] = w * self.mean + (1 - w) * mean_x
        self.std[:] = torch.sqrt(self.var)
        self.n += m
    
    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.update(x)
        if self.n > 0:
            if self.demean:
                x = x - self.mean
            if self.destd:
                x = x / (self.std + 1e-8)
            if self.clip:
                x = torch.clamp(x, -self.clip, self.clip)
        return x

class ImitatorModel(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('im', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ImitatorModel.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            mu, logstd, value, states, extra = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                    'extra' : extra
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : value,
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                    'extra' : extra
                }
                return result
            
class ImitatorBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            self.context_padding = params.get('context_padding', 8)
            self.humanoid_obs_dim = params.get('humanoid_obs_dim', 734)
            self.residual_action = params.get('residual_action', True)
            self.use_running_obs = params.get('use_running_obs', False)
            self.running_obs_type = params.get('running_obs_type', 'rl_game')
            self.use_ik = params.get('use_ik', False)
            self.ik_type = params.get('ik_type', 'optimized')
            self.ik_ignore_outlier = params.get('ik_ignore_outlier', False)
            self.kinematic_pretrained = params.get('kinematic_pretrained', False)
            
            self.smpl_rest_joints = kwargs['smpl_rest_joints']
            self.smpl_parents = kwargs['smpl_parents']
            self.smpl_children = kwargs['smpl_children']

            kwargs['input_shape'] = (self.humanoid_obs_dim,)
            super().__init__(params, **kwargs)

            if self.use_running_obs:
                if self.running_obs_type == 'rl_game':
                    self.running_obs = RunningMeanStd((self.humanoid_obs_dim,))
                else:
                    self.running_obs = RunningNorm(self.humanoid_obs_dim)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
            return

        def load(self, params):
            super().load(params)
            return

        def setup_env_named_dims(self, obs_names, obs_shapes, obs_dims, context_names, context_shapes, context_dims):
            self.obs_names = obs_names
            self.obs_shapes = obs_shapes
            self.obs_dims = obs_dims
            self.context_names = context_names
            self.context_shapes = context_shapes
            self.context_dims = context_dims
            return

        def perform_ik(self, body_pos, body_rot, dof_pos, phis=None, thetas=None, env_id=None):
            body_pos_flat = body_pos.view(-1, *body_pos.shape[2:])
            smpl_body_pos = body_pos_flat[:, mujoco_2_smpl]
            rest_body_pos = self.smpl_rest_joints[env_id] if env_id is not None else self.smpl_rest_joints
            rest_body_pos = rest_body_pos.repeat_interleave(body_pos.shape[1], dim=0)

            # phis
            if phis is None:
                phis = torch.tensor([1.0, 0.0], device=body_pos.device).expand(smpl_body_pos.shape[0], 23, -1)
            else:
                phis = phis.view(smpl_body_pos.shape[0], 23, 2)
                phis += torch.tensor([1.0, 0.0], device=phis.device)
            # leaf thetas
            if thetas is None:
                leaf_thetas = torch.eye(3, device=body_pos.device).expand(smpl_body_pos.shape[0], 5, -1, -1)
            else:
                new_thetas = thetas.view(smpl_body_pos.shape[0], 5, 6) + torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=thetas.device)
                leaf_thetas = rot6d_to_rotmat(new_thetas)

            root_diff = rest_body_pos[:, [0]] - smpl_body_pos[:, [0]]
            smpl_body_pos += root_diff
            if self.ik_type == 'optimized':
                smpl_rot_mats, global_rot_mat, global_body_pos = batch_inverse_kinematics_transform(smpl_body_pos, None, phis, rest_body_pos, self.smpl_children, self.smpl_parents, leaf_thetas, self.ik_ignore_outlier)   # 0.012s - 0.015s
            else:
                smpl_rot_mats, global_rot_mat = batch_inverse_kinematics_transform_naive(smpl_body_pos, None, phis, rest_body_pos, self.smpl_children, self.smpl_parents, leaf_thetas)     # 0.007s - 0.010s
                global_body_pos = None
            rot_mats = smpl_rot_mats[:, smpl_2_mujoco].view(*body_pos.shape[:2], -1, 3, 3)
            ik_dof_pos = rotation_matrix_to_angle_axis(rot_mats)    # 0.002s
            ik_body_rot = rotation_matrix_to_quaternion(global_rot_mat.contiguous())[..., [1, 2, 3, 0]]    # 0.001s
            ik_body_rot = ik_body_rot[:, smpl_2_mujoco].view(*body_pos.shape[:2], -1, 4)
            if global_body_pos is not None:
                ik_body_pos = global_body_pos - root_diff
                ik_body_pos = ik_body_pos[:, smpl_2_mujoco].view(*body_pos.shape[:2], -1, 3)
            else:
                ik_body_pos = None

            recon_err = None
            return ik_dof_pos, ik_body_rot, ik_body_pos, recon_err

        def forward_context(self, context_feat, mask, env_id=None, flatten=False):
            self.context = dict()
            context_chunks = torch.split(context_feat, self.context_dims, dim=-1)
            for name, shape, chunk in zip(self.context_names, self.context_shapes, context_chunks):
                self.context[name] = chunk.view(chunk.shape[:2] + shape)

            if self.use_ik:
                self.context['ik_dof_pos'], self.context['ik_body_rot'], self.context['ik_body_pos'], self.context['ik_err'] = self.perform_ik(self.context['body_pos'], self.context['body_rot'], self.context['dof_pos'], self.context['ik_phis'], self.context['ik_thetas'], env_id)
                self.context['dof_pos'] = self.context['ik_dof_pos'][..., 1:, :].reshape(*self.context['ik_dof_pos'].shape[:2], -1)
                self.context['body_rot'] = self.context['ik_body_rot']
                self.context['body_pos'] = self.context['ik_body_pos']

            context_names = self.context_names + (['ik_dof_pos', 'ik_body_pos'] if self.use_ik else [])
            for name in context_names:
                if self.context[name] is None:
                    continue
                if flatten:
                    self.context[name] = self.context[name][:, self.context_padding:-self.context_padding].reshape(-1, *self.context[name].shape[2:])
                else:
                    self.context[name] = self.context[name][:, self.context_padding:-self.context_padding + 1]

            return self.context

        def obtain_cur_context(self, t):
            if t is None:
                cur_context = self.context
            else:
                cur_context = {name: self.context[name][:, t] for name in self.context_names}
            return cur_context

        def compute_humanoid_obs(self, obs_feat, cur_context):
            body_pos = obs_feat['body_pos']
            body_rot = obs_feat['body_rot']
            body_vel = obs_feat['body_vel']
            body_ang_vel = obs_feat['body_ang_vel']
            dof_pos = obs_feat['dof_pos']
            dof_vel = obs_feat['dof_vel']
            motion_bodies = obs_feat['motion_bodies']
            target_pos = cur_context['body_pos']
            target_rot = cur_context['body_rot']
            target_dof_pos = cur_context['dof_pos']
            obs = compute_humanoid_observations_imitation(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, 
                                                          target_dof_pos, body_vel, body_ang_vel, motion_bodies, True, True)
            return obs

        def preprocess_input(self, obs_dict):
            if self.training:
                flatten = True
                assert 'context_feat' in obs_dict
                context_feat = obs_dict['context_feat']
                mask = obs_dict['context_mask']
                self.forward_context(context_feat, mask, env_id=obs_dict['env_id'], flatten=flatten)
            else:
                flatten = False

            obs_feat = dict()
            obs_chunks = torch.split(obs_dict['obs'], self.obs_dims, dim=-1)
            for name, shape, chunk in zip(self.obs_names, self.obs_shapes, obs_chunks):
                if flatten:
                    obs_feat[name] = chunk.view(np.prod(chunk.shape[:2]), *shape)
                else:
                    obs_feat[name] = chunk.view(chunk.shape[:1] + shape)

            t = obs_dict.get('t', None)
            cur_context = self.obtain_cur_context(t)

            obs_dict['human_obs'] = self.compute_humanoid_obs(obs_feat, cur_context)
            obs_dict['cur_context'] = cur_context
            if self.use_running_obs:
                obs_dict['obs_processed'] = self.running_obs(obs_dict['human_obs'])
            else:
                obs_dict['obs_processed'] = obs_dict['human_obs']
            return

        def forward(self, obs_dict):

            actor_outputs = self.eval_actor(obs_dict)
            value = self.eval_critic(obs_dict)

            extra = {'context': obs_dict['cur_context']}

            output = actor_outputs + (value, None, extra)

            return output

        def eval_actor(self, obs_dict):
            if 'obs_processed' not in obs_dict:
                self.preprocess_input(obs_dict)
            obs = obs_dict['obs_processed']

            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            a_out = self.actor_mlp(a_out)
                     
            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                if self.residual_action:
                    target_dof_pos = obs_dict['cur_context']['dof_pos']
                    mu[:, :target_dof_pos.shape[-1]] += target_dof_pos

                return mu, sigma
            return

        def eval_critic(self, obs_dict):
            if 'obs_processed' not in obs_dict:
                self.preprocess_input(obs_dict)
            obs = obs_dict['obs_processed']

            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

    def build(self, name, **kwargs):
        net = ImitatorBuilder.Network(self.params, **kwargs)
        return net

class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if not (args.tmp or args.no_log):
            if self.consecutive_successes.current_size > 0:
                mean_con_successes = self.consecutive_successes.get_mean()
                self.algo.log_dict.update({'successes/consecutive_successes/mean': mean_con_successes})
        return

class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

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
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info



if __name__ == '__main__':

    set_np_formatting()
    args = get_args()
    cfg, cfg_train = load_cfg(args)


    def create_rlgpu_env(**kwargs):
        use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
        if use_horovod:
            import horovod.torch as hvd

            rank = hvd.rank()
            print("Horovod rank: ", rank)

            cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

            args.device = 'cuda'
            args.device_id = rank
            args.rl_device = 'cuda:' + str(rank)

            cfg['rank'] = rank
            cfg['rl_device'] = 'cuda:' + str(rank)

        sim_params = parse_sim_params(args, cfg, cfg_train)
    #     task, env = parse_task(args, cfg, cfg_train, sim_params)
    # def parse_task(args, cfg, cfg_train, sim_params):
        # create native task and pass custom config
        device_id = args.device_id
        rl_device = args.rl_device

        cfg["seed"] = cfg_train.get("seed", -1)
        cfg_task = cfg["env"]
        cfg_task["seed"] = cfg["seed"]

        try:
            task = eval(cfg['name'])(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless)
        except NameError as e:
            print(e)
        env = VecTaskPythonWrapper(task, rl_device, cfg_train['params']['config'].get("clip_observations", np.inf), cfg_train['params']['config'].get("clip_actions_val", np.inf))

        # return task, env

        print(env.num_envs)
        print(env.num_actions)
        print(env.num_obs)
        print(env.num_states)

        frames = kwargs.pop('frames', 1)
        print(f'frames: {frames}')
        # if frames > 1:
        #     env = wrappers.FrameStack(env, frames, False)
        return env

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs), 'vecenv_type': 'RLGPU'})

    runner = Runner(RLGPUAlgoObserver())
    
    runner.algo_factory.register_builder('pose_im_rnn', lambda **kwargs : ImitatorAgent(**kwargs))        # training agent
    runner.player_factory.register_builder('pose_im_rnn', lambda **kwargs : ImitatorPlayer(**kwargs))     # testing agent
    runner.model_builder.model_factory.register_builder('pose_im', lambda network, **kwargs : ImitatorModel(network))    # network wrapper
    runner.model_builder.network_factory.register_builder('pose_im_rnn', lambda **kwargs : ImitatorBuilder())     # actuall network definition class
    # model = xxx(network)

    runner.load(cfg_train)
    runner.reset()
    runner.run(vars(args))
