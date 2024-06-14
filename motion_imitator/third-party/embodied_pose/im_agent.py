import torch 
import time
import os
import numpy as np
from torch import nn
from torch import optim
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
import datetime


class AverageMeter(object):

    def __init__(self, avg=None, count=1):
        self.reset()
        if avg is not None:
            self.val = avg
            self.avg = avg
            self.count = count
            self.sum = avg * count
    
    def __repr__(self) -> str:
        return f'{self.avg: .4f}'

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count



class AMPDataset(datasets.PPODataset):
    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len):
        super().__init__(batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len)
        self._idx_buf = torch.randperm(batch_size)
        return
    
    def update_mu_sigma(self, mu, sigma):	  
        raise NotImplementedError()
        return

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        sample_idx = self._idx_buf[start:end]

        input_dict = {}
        for k,v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                input_dict[k] = v[sample_idx]
                
        if (end >= self.batch_size):
            self._shuffle_idx_buf()

        return input_dict

    def _shuffle_idx_buf(self):
        self._idx_buf[:] = torch.randperm(self.batch_size)
        return

class ImitatorAgent(a2c_continuous.A2CAgent):

    def __init__(self, base_name, config):
        a2c_common.A2CBase.__init__(self, base_name, config)
        self.task = self.vec_env.env.task
        self.last_lr = config['learning_rate']
        self.is_discrete = False


        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
    

        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        self.clip_actions = False
        self._save_intermediate = config.get('save_intermediate', False)
        self.end_value_type = config.get('end_value_type', 'next')

        net_config = self._build_net_config()
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)
        self.model.eval()
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.normalize_input:
            obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)

        if self.has_central_value:
            cv_config = {
                'state_shape' : torch_ext.shape_whc_to_cwh(self.state_shape), 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length, 
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_len' : self.seq_len, 
                'model' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'multi_gpu' : self.multi_gpu
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = AMPDataset(self.num_actors, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        self.algo_observer.after_init(self)
        
        self.sub_rewards_names = None
        self.log_dict = dict()

        self.task.register_model(self.model)

        pretrained_model_cp = config.get('pretrained_model_cp', None)
        if pretrained_model_cp is not None and not config.get('load_checkpoint', False):
            if type(pretrained_model_cp) is list:
                for cp in pretrained_model_cp:
                    self.load_pretrained(cp)
            else:
                self.load_pretrained(pretrained_model_cp)

        self.game_rewards = torch_ext.AverageMeter((self.value_size,), self.games_to_track).to(self.ppo_device)

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'smpl_rest_joints': self.task.smpl_rest_joints,
            'smpl_parents': self.task.smpl_parents,
            'smpl_children': self.task.smpl_children
        }
        if hasattr(self.task, 'num_con_planes'):
            config['num_con_planes'] = self.task.num_con_planes
            config['num_con_bodies'] = self.task.num_con_bodies
        return config
    
    def restore(self, cp_name):
        cp_path = os.path.join(self.network_path, f"{self.config['name']}_{cp_name}.pth")
        checkpoint = torch.load(cp_path, map_location=self.device)
        self.set_full_state_weights(checkpoint)

    def load_pretrained(self, cp_path):
        checkpoint = torch.load(cp_path, map_location=self.device)

        if 'pytorch-lightning_version' in checkpoint:
            model_state_dict = self.model.a2c_network.state_dict()
            print('loading model from Lighting checkpoint...')
            print('Shared keys in model and lightning checkpoint:', [key for key in checkpoint['state_dict'] if key in model_state_dict])
            print('Lightning keys not found in current model:', [key for key in checkpoint['state_dict'] if key not in model_state_dict])
            print('Keys not found in lightning checkpoint:', [key for key in model_state_dict if key not in checkpoint['state_dict']])
            self.model.a2c_network.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model_state_dict = self.model.state_dict()
            missing_checkpoint_keys = [key for key in model_state_dict if key not in checkpoint['model']]
            print('loading model from EmbodyPose checkpoint...')
            print('Shared keys in model and checkpoint:', [key for key in model_state_dict if key in checkpoint['model']])
            print('Keys not found in current model:', [key for key in checkpoint['model'] if key not in model_state_dict])
            print('Keys not found in checkpoint:', missing_checkpoint_keys)

            discard_pretrained_sigma = self.config.get('discard_pretrained_sigma', False)
            if discard_pretrained_sigma:
                checkpoint_keys = list(checkpoint['model'])
                for key in checkpoint_keys:
                    if 'sigma' in key:
                        checkpoint['model'].pop(key)

            weights = checkpoint
            if hasattr(self.model, 'load_weights'):
                self.model.load_weights(weights['model'])
            else:
                self.model.load_state_dict(weights['model'], strict=False)
            self.set_stats_weights(weights)

            
            load_rlgame_norm = len([key for key in missing_checkpoint_keys if 'running_obs' in key]) > 0
            if load_rlgame_norm:
                if 'running_mean_std' in checkpoint:
                    print('loading running_obs from rl game running_mean_std')
                    self.model.a2c_network.running_obs.n = checkpoint['running_mean_std']['count'].long()
                    self.model.a2c_network.running_obs.mean[:] = checkpoint['running_mean_std']['running_mean'].float()
                    self.model.a2c_network.running_obs.var[:] = checkpoint['running_mean_std']['running_var'].float()
                    self.model.a2c_network.running_obs.std[:] = torch.sqrt(self.model.a2c_network.running_obs.var)
                elif 'a2c_network.running_obs.running_mean' in checkpoint['model']:
                    print('loading running_obs from rl game in model')
                    obs_len = checkpoint['model']['a2c_network.running_obs.running_mean'].shape[0]
                    self.model.a2c_network.running_obs.n = checkpoint['model']['a2c_network.running_obs.count'].long()
                    self.model.a2c_network.running_obs.mean[:obs_len] = checkpoint['model']['a2c_network.running_obs.running_mean'].float()
                    self.model.a2c_network.running_obs.var[:obs_len] = checkpoint['model']['a2c_network.running_obs.running_var'].float()
                    self.model.a2c_network.running_obs.std[:obs_len] = torch.sqrt(self.model.a2c_network.running_obs.var[:obs_len])


    def train(self):
        args = self.config['args']
        super().init_tensors()
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])
        self.tensor_list += ['next_obses']

        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        self.frame = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        while True:
            if hasattr(self.task, 'pre_epoch'):
                self.task.pre_epoch(self.epoch_num)

            epoch_num = self.update_epoch()
            train_info = self.train_epoch()

            sum_time = train_info['total_time']
            play_time = train_info['play_time']
            update_time = train_info['update_time']
            total_time += sum_time
            frame = self.frame
            if self.multi_gpu:
                self.hvd.sync_stats(self)

            # logging
            if self.rank == 0:
                self.log_dict = dict()
                scaled_time = sum_time
                scaled_play_time = train_info['play_time']
                curr_frames = self.curr_frames
                self.frame += curr_frames

                aux_losses_dict = {k: torch_ext.mean_list(v).item() for k, v in train_info.items() if k.startswith('aux_')}
                if self.print_stats:
                    fps_step = curr_frames // scaled_play_time
                    fps_total = curr_frames // scaled_time
                    mean_lengths = self.game_lengths.get_mean() if self.game_rewards.current_size > 0 else 0
                    aux_str = ' '.join(f"{k.replace('_loss', '')}: {v:.4f}" for k, v in aux_losses_dict.items())
                    if len(aux_str) > 0:
                        aux_str = '\t' + aux_str
                    

                    def convert_sec_to_time(secs):
                        return str(datetime.timedelta(seconds=round(secs)))


                    def get_eta_str(cur_iter, total_iter, time_per_iter):
                        eta = time_per_iter * (total_iter - cur_iter - 1)
                        return convert_sec_to_time(eta)
                    
                    print('{}\tT_play {:.2f}\tT_update {:.2f}\tETA {}\tstep_rewards {:.4f} {}{}\teps_len {:.2f}\talive {:.2f}\tfps step {}\tfps total {}\t{}'
                          .format(epoch_num, play_time, update_time, get_eta_str(epoch_num, self.max_epochs, sum_time), self.step_rewards.avg.item(),
                                    np.array2string(self.step_sub_rewards.avg.cpu().numpy(), formatter={'all': lambda x: '%.4f' % x}, separator=','), aux_str,
                                    mean_lengths, self.alive_ratio, fps_step, fps_total, self.config['args'].cfg))

                self.log_dict.update({'frame': frame, 'epoch_num': epoch_num, 'total_time': total_time})
                self.log_dict.update({'performance/total_fps': curr_frames / scaled_time})
                self.log_dict.update({'performance/step_fps': curr_frames / scaled_play_time})
                self.log_dict.update({'info/epochs': epoch_num})
                self.log_dict.update({'performance/update_time': train_info['update_time']})
                self.log_dict.update({'performance/play_time': train_info['play_time']})
                self.log_dict.update({'losses/a_loss': torch_ext.mean_list(train_info['actor_loss']).item()})
                self.log_dict.update({'losses/c_loss': torch_ext.mean_list(train_info['critic_loss']).item()})
                self.log_dict.update({'losses/bounds_loss': torch_ext.mean_list(train_info['b_loss']).item()})
                self.log_dict.update({'losses/entropy': torch_ext.mean_list(train_info['entropy']).item()})
                for key, value in aux_losses_dict.items():
                    self.log_dict['aux_losses/' + key] = value

                self.log_dict.update({'info/last_lr': train_info['last_lr'][-1] * train_info['lr_mul'][-1]})
                self.log_dict.update({'info/lr_mul': train_info['lr_mul'][-1]})
                self.log_dict.update({'info/e_clip': self.e_clip * train_info['lr_mul'][-1]})
                self.log_dict.update({'info/clip_frac': torch_ext.mean_list(train_info['actor_clip_frac']).item()})
                self.log_dict.update({'info/kl': torch_ext.mean_list(train_info['kl']).item()})
                self.log_dict.update({'info/alive_ratio': self.alive_ratio})

                self.log_dict.update({'step_rewards': self.step_rewards.avg.item()})
                for i, name in enumerate(self.sub_rewards_names.split(',')):
                    self.log_dict.update({f'sub_rewards/{name}': self.step_sub_rewards.avg[i].item()})

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                
                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    if len(mean_rewards.shape) == 0:
                        mean_rewards = mean_rewards[None]

                    for i in range(self.value_size):
                        self.log_dict.update({'game_rewards{0}'.format(i): mean_rewards[i]})
                    self.log_dict.update({'episode_lengths': mean_lengths})

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)
                
                # save models
                latest_model_file = os.path.join(self.network_path, f"{self.config['name']}_latest")
                if self.save_freq > 0:
                    if (epoch_num % self.save_freq == 0):
                        epoch_model_file = os.path.join(self.network_path, f"{self.config['name']}_epoch{epoch_num:05d}")
                        self.save(latest_model_file)
                        self.save(epoch_model_file)

                # import ipdb; ipdb.set_trace()
                # # self.obs = self.env_reset()
                # self.vec_env._state_init
                # obs = self.vec_env.reset(None)
                # obs = self.obs_to_tensors(obs)
                
                is_exit = self.task.post_epoch(self.epoch_num, self.step_rewards.avg.item()) 
                            
                if is_exit:
                    self.save(latest_model_file)
                    print('step reward > 0.85, success!')
                    return self.last_mean_rewards, epoch_num
                
                if epoch_num >= self.max_epochs:
                    self.save(latest_model_file)
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num
                

                    
                update_time = 0
        return

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs_dict['obs'] = self._preproc_obs(obs_dict['obs'])
        value = self.model.a2c_network.eval_critic(obs_dict)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value
    
    def play_steps(self):
        self.set_eval()
        
        epinfos = []
        done_indices = []
        update_list = self.update_list

        self.obs = self.env_reset()

        self.dones[:] = 0
        prev_dones = self.dones.clone()

        self.step_rewards = AverageMeter()
        self.step_sub_rewards = AverageMeter()

        self.current_rewards[:] = 0
        self.current_lengths[:] = 0

        for n in range(self.horizon_length):
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            self.obs['t'] = n
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                obs = self.obs
                processed_obs = self._preproc_obs(obs['obs'])
                self.model.eval()
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs,
                    'rnn_states' : self.rnn_states,
                    't': obs['t']
                }

                with torch.no_grad():
                    res_dict = self.model(input_dict)
                    if self.has_central_value:
                        states = obs['states']
                        input_dict = {
                            'is_train': False,
                            'states' : states,
                        }
                        value = self.get_central_value(input_dict)
                        res_dict['values'] = value
                if self.normalize_value:
                    res_dict['values'] = self.value_mean_std(res_dict['values'], True)
                # return res_dict
                    

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic({'obs': self.obs['obs'], 't': n + 1})
            if self.end_value_type == 'next':
                next_vals *= (1.0 - terminated)
            elif self.end_value_type == 'zero':
                next_vals *= (1.0 - self.dones.unsqueeze(-1))
            elif self.end_value_type == 'mean':
                next_vals *= (1.0 - self.dones.unsqueeze(-1))
                reset = self.dones.unsqueeze(-1) * (1 - terminated)
                ind = reset == 1.0
                next_vals[ind] = self.experience_buffer.tensor_dict['values'][:, ind][:n + 1].mean(dim=0)
            else:
                raise ValueError('unknown end_value_type')
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1

            step_dones = self.dones * (1 - prev_dones)
            all_done_indices = step_dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            
            rewards_not_done = rewards[prev_dones == 0]
            n = rewards_not_done.shape[0]
            self.step_rewards.update(rewards_not_done.mean(dim=0), n)
            if 'sub_rewards' in infos:
                sub_rewards_not_done = infos['sub_rewards'][prev_dones == 0]
                self.step_sub_rewards.update(sub_rewards_not_done.mean(dim=0), n)
                self.sub_rewards_names = infos['sub_rewards_names']

            self.algo_observer.process_infos(infos, done_indices)

            done_indices = done_indices[:, 0]

            prev_dones = self.dones.clone()

            if torch.all(self.dones == 1):
                break

        self.game_rewards.update(self.current_rewards[self.dones == 0])
        self.game_lengths.update(self.current_lengths[self.dones == 0])

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        

    #     mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
    # def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)
            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam
        # return mb_advs

        mb_returns = mb_advs + mb_values


        def swap01(arr):
            """
            swap axes 0 and 1
            """
            if arr is None:
                return arr
            s = arr.size()
            return arr.transpose(0, 1)


        batch_dict = self.experience_buffer.get_transformed_list(swap01, self.tensor_list)
        batch_dict['returns'] = swap01(mb_returns)
        batch_dict['alive'] = 1.0 - batch_dict['dones']
        batch_dict['played_frames'] = self.batch_size
        batch_dict['context_feat'] = self.task.context_feat
        batch_dict['context_mask'] = self.task.context_mask

        self.alive_ratio = batch_dict['alive'].sum().item() / batch_dict['alive'].numel()

        return batch_dict

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)
        
        advantages = self._calc_advs(batch_dict)

        if self.normalize_value:
            orig_shape = values.shape
            values = self.value_mean_std(values.reshape(-1, values.shape[-1])).view(orig_shape)
            returns = self.value_mean_std(returns.reshape(-1, values.shape[-1])).view(orig_shape)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        dataset_dict['dones'] = dones
        dataset_dict['alive'] = batch_dict['alive']
        dataset_dict['context_feat'] = batch_dict['context_feat']
        dataset_dict['context_mask'] = batch_dict['context_mask']

        assert self.num_actors == batch_dict['context_feat'].shape[0]
        dataset_dict['env_id'] = torch.arange(self.num_actors, device=self.device)

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

        return

    def _calc_advs(self, batch_dict):
        returns = batch_dict['returns']
        values = batch_dict['values']
        alive = batch_dict['alive']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=-1)

        if self.normalize_advantage:
            valid_advs = advantages[alive == 1]
            advantages = (advantages - valid_advs.mean()) / (valid_advs.std() + 1e-8)

        return advantages

    def calc_gradients(self, input_dict):
        self.set_train()

        for key in input_dict:
            if key not in ['obs', 'context_feat', 'context_mask']:
                new_shape = (-1, input_dict[key].shape[-1]) if input_dict[key].dim() > 2 else (-1,)
                input_dict[key] = input_dict[key].reshape(new_shape)

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        
        alive = input_dict['alive']

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'context_feat': input_dict['context_feat'],
            'context_mask': input_dict['context_mask'],
            'env_id': input_dict['env_id'], 
            'obs': obs_batch
        }

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            #####actor loss
            # a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            # def _actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
            ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip,1.0 + curr_e_clip)
            a_loss = torch.max(-surr1, -surr2)

            clipped = torch.abs(ratio - 1.0) > curr_e_clip
            clipped = clipped.detach()
            
            a_info = {
                'actor_loss': a_loss,
                'actor_clipped': clipped.detach()
            }
            # return info
            # a_loss = a_info['actor_loss']


            ########critic loss
            # c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            # def _critic_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
            if self.clip_value:
                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
                value_losses = (values - return_batch)**2
                value_losses_clipped = (value_pred_clipped - return_batch)**2
                c_loss = torch.max(value_losses, value_losses_clipped)
            else:
                c_loss = (return_batch - values)**2

            c_info = {'critic_loss': c_loss}
            #     return info
            # c_loss = c_info['critic_loss']


            ######bound loss
        #     b_loss = self.bound_loss(mu)
        # def bound_loss(self, mu):
            if self.bounds_loss_coef is not None:
                soft_bound = 1.0
                mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
                mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
                b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
            else:
                b_loss = 0
            # return b_loss
        

            b_loss_unsq = b_loss.unsqueeze(1) if isinstance(b_loss, torch.Tensor) else 0.0
            b_loss_coef = 0.0 if self.bounds_loss_coef is None else self.bounds_loss_coef

            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss_unsq], mask=alive)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + b_loss_coef * b_loss

            if hasattr(self.task, 'get_aux_losses'):
                aux_losses, aux_losses_weighted = self.task.get_aux_losses(res_dict)
                for aux_loss in aux_losses_weighted.values():
                    loss += aux_loss
            else:
                aux_losses = aux_losses_weighted = None
            
            a_clip_frac = torch.mean(a_info['actor_clipped'].float())
            
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()

        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce=False)
            kl_dist = (kl_dist * alive).sum() / alive.numel()
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)

        if aux_losses is not None:
            self.train_result.update(aux_losses)

        return

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps() 

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)
        
        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])
                
                if self.schedule_type == 'legacy':  
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)
            
            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time

        return train_info

    def env_reset(self, env_ids=None):
        obs = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs

    def _log_train_info(self, train_info, frame):
        self.writer.add_scalar('performance/update_time', train_info['update_time'], frame)
        self.writer.add_scalar('performance/play_time', train_info['play_time'], frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(train_info['actor_loss']).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(train_info['critic_loss']).item(), frame)
        self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(train_info['b_loss']).item(), frame)
        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(train_info['entropy']).item(), frame)
        self.writer.add_scalar('info/last_lr', train_info['last_lr'][-1] * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/lr_mul', train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/clip_frac', torch_ext.mean_list(train_info['actor_clip_frac']).item(), frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(train_info['kl']).item(), frame)
        return



