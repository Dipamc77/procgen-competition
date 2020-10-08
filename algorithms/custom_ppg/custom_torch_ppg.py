from ray.rllib.policy.torch_policy import TorchPolicy
import numpy as np
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, convert_to_torch_tensor
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from collections import deque
from .utils import *
import time

torch, nn = try_import_torch()
import torch.distributions as td

class CustomTorchPolicy(TorchPolicy):
    """Example of a random policy
    If you are using tensorflow/pytorch to build custom policies,
    you might find `build_tf_policy` and `build_torch_policy` to
    be useful.
    Adopted from examples from https://docs.ray.io/en/master/rllib-concepts.html
    """

    def __init__(self, observation_space, action_space, config):
        self.config = config

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"], framework="torch")
        self.model = ModelCatalog.get_model_v2(
                        obs_space=observation_space,
                        action_space=action_space,
                        num_outputs=logit_dim,
                        model_config=self.config["model"],
                        framework="torch",
                        device=self.device,
                     )

        TorchPolicy.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            config=config,
            model=self.model,
            loss=None,
            action_distribution_class=dist_class,
        )

        self.framework = "torch"
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.aux_optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.max_reward = self.config['env_config']['return_max']
        self.rewnorm = RewardNormalizer(cliprew=self.max_reward) ## TODO: Might need to go to custom state
        self.reward_deque = deque(maxlen=100)
        self.best_reward = -np.inf
        self.best_weights = None
        self.time_elapsed = 0
        self.batch_end_time = time.time()
        self.timesteps_total = 0
        self.best_rew_tsteps = 0
        
        nw = self.config['num_workers'] if self.config['num_workers'] > 0 else 1
        nenvs = nw * self.config['num_envs_per_worker']
        nsteps = self.config['rollout_fragment_length']
        n_pi = self.config['n_pi']
        self.nbatch = nenvs * nsteps
        self.actual_batch_size = self.nbatch // self.config['updates_per_batch']
        self.accumulate_train_batches = int(np.ceil( self.actual_batch_size / self.config['max_minibatch_size'] ))
        self.mem_limited_batch_size = self.actual_batch_size // self.accumulate_train_batches
        if self.nbatch % self.actual_batch_size != 0 or self.nbatch % self.mem_limited_batch_size != 0:
            print("#################################################")
            print("WARNING: MEMORY LIMITED BATCHING NOT SET PROPERLY")
            print("#################################################")
        self.retune_selector = RetuneSelector(nenvs, observation_space, action_space, 
                                              skips = self.config['skips'], 
                                              n_pi = n_pi,
                                              num_retunes = self.config['num_retunes'])
        
        replay_shape = (n_pi, nsteps, nenvs)
        self.exp_replay = np.empty((*replay_shape, *observation_space.shape), dtype=np.uint8)
        self.vtarg_replay = np.empty(replay_shape, dtype=np.float32)
        self.save_success = 0
        self.target_timesteps = 8_000_000
        self.buffer_time = 20 # TODO: Could try to do a median or mean time step check instead
        self.max_time = 7200
        self.maxrewep_lenbuf = deque(maxlen=100)
        self.gamma = self.config['gamma']
        self.adaptive_discount_tuner = AdaptiveDiscountTuner(self.gamma, momentum=0.98, eplenmult=3)
        
        self.lr = config['lr']
        self.ent_coef = config['entropy_coeff']
        
        self.last_dones = np.zeros((nw * self.config['num_envs_per_worker'],))
        self.make_distr = dist_build(action_space)
        
    def to_tensor(self, arr):
        return torch.from_numpy(arr).to(self.device)
        
    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        """Fused compute gradients and apply gradients call.
        Either this or the combination of compute/apply grads must be
        implemented by subclasses.
        Returns:
            grad_info: dictionary of extra metadata from compute_gradients().
        Examples:
            >>> batch = ev.sample()
            >>> ev.learn_on_batch(samples)
        Reference: https://github.com/ray-project/ray/blob/master/rllib/policy/policy.py#L279-L316
        """
        ## Config data values
        nbatch = self.nbatch
        nbatch_train = self.mem_limited_batch_size 
        gamma, lam = self.gamma, self.config['lambda']
        nsteps = self.config['rollout_fragment_length']
        nenvs = nbatch//nsteps
        ts = (nenvs, nsteps)
        mb_dones = unroll(samples['dones'], ts)
        
        ## Reward Normalization - No reward norm works well for many envs
        if self.config['standardize_rewards']:
            mb_origrewards = unroll(samples['rewards'], ts)
            mb_rewards =  np.zeros_like(mb_origrewards)
            mb_rewards[0] = self.rewnorm.normalize(mb_origrewards[0], self.last_dones)
            for ii in range(1, nsteps):
                mb_rewards[ii] = self.rewnorm.normalize(mb_origrewards[ii], mb_dones[ii-1])
            self.last_dones = mb_dones[-1]
        else:
            mb_rewards = unroll(samples['rewards'], ts)
       
        # Weird hack that helps in many envs (Yes keep it after normalization)
        rew_scale = self.config["scale_reward"]
        if rew_scale != 1.0:
            mb_rewards *= rew_scale
        
        should_skip_train_step = self.best_reward_model_select(samples)
        if should_skip_train_step:
            self.update_batch_time()
            return {} # Not doing last optimization step - This is intentional due to noisy gradients
          
        obs = samples['obs']

        ## Value prediction
        next_obs = unroll(samples['new_obs'], ts)[-1]
        last_values, _ = self.model.vf_pi(next_obs, ret_numpy=True, no_grad=True, to_torch=True)
        values = np.empty((nbatch,), dtype=np.float32)
        for start in range(0, nbatch, nbatch_train): # Causes OOM up if trying to do all at once
            end = start + nbatch_train
            values[start:end], _ = self.model.vf_pi(samples['obs'][start:end], ret_numpy=True, no_grad=True, to_torch=True)
        
        ## GAE
        mb_values = unroll(values, ts)
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextvalues = last_values
            else:
                nextvalues = mb_values[t+1]
            nextnonterminal = 1.0 - mb_dones[t]
            delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        
        ## Data from config
        cliprange, vfcliprange = self.config['clip_param'], self.config['vf_clip_param']
        lrnow = self.lr
        max_grad_norm = self.config['grad_clip']
        ent_coef, vf_coef = self.ent_coef, self.config['vf_loss_coeff']
        
        neglogpacs = -samples['action_logp'] ## np.isclose seems to be True always, otherwise compute again if needed
        noptepochs = self.config['num_sgd_iter']
        actions = samples['actions']
        returns = roll(mb_returns)
        
        advs = returns - values
        normalized_advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8) 
        
        ## Train multiple epochs
        optim_count = 0
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (self.to_tensor(arr[mbinds]) for arr in (obs, returns, actions, values, neglogpacs, normalized_advs))
                optim_count += 1
                apply_grad = (optim_count % self.accumulate_train_batches) == 0
                self._batch_train(apply_grad, self.accumulate_train_batches,
                                  lrnow, cliprange, vfcliprange, max_grad_norm, ent_coef, vf_coef, *slices)

        ## Distill with aux head
        should_retune = self.retune_selector.update(unroll(obs, ts), mb_returns, self.exp_replay, self.vtarg_replay)
        if should_retune:
            self.aux_train()
            self.update_batch_time()
            return {}
        
        self.update_gamma(samples)
        self.update_lr()
        self.update_ent_coef()
            
        self.update_batch_time()
        return {}
    
    def update_batch_time(self):
        self.time_elapsed += time.time() - self.batch_end_time
        self.batch_end_time = time.time()
        
    def _batch_train(self, apply_grad, num_accumulate, 
                     lr, cliprange, vfcliprange, max_grad_norm,
                     ent_coef, vf_coef,
                     obs, returns, actions, values, neglogpac_old, advs):
        
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        vpred, pi_logits = self.model.vf_pi(obs, ret_numpy=False, no_grad=False, to_torch=False)
        pd = self.make_distr(pi_logits)
        neglogpac = -pd.log_prob(actions[...,None]).squeeze(1)
        entropy = torch.mean(pd.entropy())

        vf_loss = .5 * torch.mean(torch.pow((vpred - returns), 2))

        ratio = torch.exp(neglogpac_old - neglogpac)
        pg_losses1 = -advs * ratio
        pg_losses2 = -advs * torch.clamp(ratio, 1-cliprange, 1+cliprange)
        pg_loss = torch.mean(torch.max(pg_losses1, pg_losses2))

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        
        loss = loss / num_accumulate

        loss.backward()
        if apply_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

        
    def aux_train(self):
        for g in self.aux_optimizer.param_groups:
            g['lr'] = self.lr
        nbatch_train = self.mem_limited_batch_size 
        retune_epochs = self.config['retune_epochs']
        replay_shape = self.vtarg_replay.shape
        replay_pi = np.empty((*replay_shape, self.retune_selector.ac_space.n), dtype=np.float32)

        for nnpi in range(self.retune_selector.n_pi):
            for ne in range(self.retune_selector.nenvs):
                _, replay_pi[nnpi, :, ne] = self.model.vf_pi(self.exp_replay[nnpi, :, ne], 
                                                             ret_numpy=True, no_grad=True, to_torch=True)
        
        # Tune vf and pi heads to older predictions with (augmented?) observations
        for ep in range(retune_epochs):
            for slices in self.retune_selector.make_minibatches_with_rollouts(self.exp_replay, self.vtarg_replay, replay_pi):
                self.tune_policy(slices[0], self.to_tensor(slices[1]), self.to_tensor(slices[2]))

        self.retune_selector.retune_done()
 
    def tune_policy(self, obs, target_vf, target_pi):
        if self.config['augment_buffer']:
            obs_aug = np.empty(obs.shape, obs.dtype)
            aug_idx = np.random.randint(6, size=len(obs))
            obs_aug[aug_idx == 0] = pad_and_random_crop(obs[aug_idx == 0], 64, 10)
            obs_aug[aug_idx == 1] = random_cutout_color(obs[aug_idx == 1], 10, 30)
            obs_aug[aug_idx >= 2] = obs[aug_idx >= 2]
            obs_in = self.to_tensor(obs_aug)
        else:
            obs_in = self.to_tensor(obs)
        
        vpred, pi_logits = self.model.vf_pi(obs_in, ret_numpy=False, no_grad=False, to_torch=False)
        aux_vpred = self.model.aux_value_function()
        vf_loss = .5 * torch.mean(torch.pow(vpred - target_vf, 2))
        aux_loss = .5 * torch.mean(torch.pow(aux_vpred - target_vf, 2))
        
        target_pd = self.make_distr(target_pi)
        pd = self.make_distr(pi_logits)
        pi_loss = td.kl_divergence(target_pd, pd).mean()
        
        loss = vf_loss + pi_loss + aux_loss
        
        loss.backward()
        self.aux_optimizer.step()
        self.aux_optimizer.zero_grad()
        
    def best_reward_model_select(self, samples):
        self.timesteps_total += len(samples['dones'])
        
        ## Best reward model selection
        eprews = [info['episode']['r'] for info in samples['infos'] if 'episode' in info]
        self.reward_deque.extend(eprews)
        mean_reward = safe_mean(eprews) if len(eprews) >= 100 else safe_mean(self.reward_deque)
        if self.best_reward < mean_reward:
            self.best_reward = mean_reward
            self.best_weights = self.get_weights()["current_weights"]
            self.best_rew_tsteps = self.timesteps_total
           
        if self.timesteps_total > self.target_timesteps or (self.time_elapsed + self.buffer_time) > self.max_time:
            if self.best_weights is not None:
                self.set_model_weights(self.best_weights)
                return True
            
        return False
    
    def update_lr(self):
        if self.config['lr_schedule'] == 'linear':
            self.lr = linear_schedule(initial_val=self.config['lr'],
                                      final_val=self.config['final_lr'],
                                      current_steps=self.timesteps_total,
                                      total_steps=self.target_timesteps)
            
        elif self.config['lr_schedule'] == 'exponential':
            self.lr = 0.997 * self.lr 

    
    def update_ent_coef(self):
        if self.config['entropy_schedule']:
            self.ent_coef = linear_schedule(initial_val=self.config['entropy_coeff'], 
                                            final_val=self.config['final_entropy_coeff'], 
                                            current_steps=self.timesteps_total, 
                                            total_steps=self.target_timesteps)
    
    def update_gamma(self, samples):
        if self.config['adaptive_gamma']:
            epinfobuf = [info['episode'] for info in samples['infos'] if 'episode' in info]
            self.maxrewep_lenbuf.extend([epinfo['l'] for epinfo in epinfobuf if epinfo['r'] >= self.max_reward])
            sorted_nth = lambda buf, n: np.nan if len(buf) < 100 else sorted(self.maxrewep_lenbuf.copy())[n]
            target_horizon = sorted_nth(self.maxrewep_lenbuf, 80)
            self.gamma = self.adaptive_discount_tuner.update(target_horizon)

        
    def get_custom_state_vars(self):
        return {
            "time_elapsed": self.time_elapsed,
            "timesteps_total": self.timesteps_total,
            "best_weights": self.best_weights,
            "reward_deque": self.reward_deque,
            "batch_end_time": self.batch_end_time,
            "retune_selector": self.retune_selector,
            "gamma": self.gamma,
            "maxrewep_lenbuf": self.maxrewep_lenbuf,
            "lr": self.lr,
            "ent_coef": self.ent_coef,
            "rewnorm": self.rewnorm,
            "best_rew_tsteps": self.best_rew_tsteps,
            "best_reward": self.best_reward,
            "last_dones": self.last_dones,
        }
    
    def set_custom_state_vars(self, custom_state_vars):
        self.time_elapsed = custom_state_vars["time_elapsed"]
        self.timesteps_total = custom_state_vars["timesteps_total"]
        self.best_weights = custom_state_vars["best_weights"]
        self.reward_deque = custom_state_vars["reward_deque"]
        self.batch_end_time = custom_state_vars["batch_end_time"]
        self.retune_selector = custom_state_vars["retune_selector"]
        self.gamma = self.adaptive_discount_tuner.gamma = custom_state_vars["gamma"]
        self.maxrewep_lenbuf = custom_state_vars["maxrewep_lenbuf"]
        self.lr =custom_state_vars["lr"]
        self.ent_coef = custom_state_vars["ent_coef"]
        self.rewnorm = custom_state_vars["rewnorm"]
        self.best_rew_tsteps = custom_state_vars["best_rew_tsteps"]
        self.best_reward = custom_state_vars["best_reward"]
        self.last_dones = custom_state_vars["last_dones"]
    
    @override(TorchPolicy)
    def get_weights(self):
        weights = {}
        weights["current_weights"] = {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }
#         weights["optimizer_state"] = {
#             k: v
#             for k, v in self.optimizer.state_dict().items()
#         }
#         weights["aux_optimizer_state"] = {
#             k: v
#             for k, v in self.aux_optimizer.state_dict().items()
#         }
#         weights["custom_state_vars"] = self.get_custom_state_vars()
        return weights
        
    
    @override(TorchPolicy)
    def set_weights(self, weights):
        self.set_model_weights(weights["current_weights"])
#         self.set_optimizer_state(weights["optimizer_state"])
#         self.set_aux_optimizer_state(weights["aux_optimizer_state"])
#         self.set_custom_state_vars(weights["custom_state_vars"])
        
    def set_aux_optimizer_state(self, aux_optimizer_state):
        aux_optimizer_state = convert_to_torch_tensor(aux_optimizer_state, device=self.device)
        self.aux_optimizer.load_state_dict(aux_optimizer_state)
        
    def set_optimizer_state(self, optimizer_state):
        optimizer_state = convert_to_torch_tensor(optimizer_state, device=self.device)
        self.optimizer.load_state_dict(optimizer_state)
        
    def set_model_weights(self, model_weights):
        model_weights = convert_to_torch_tensor(model_weights, device=self.device)
        self.model.load_state_dict(model_weights)