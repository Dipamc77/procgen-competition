from ray.rllib.policy.torch_policy import TorchPolicy
import numpy as np
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, convert_to_torch_tensor
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from collections import deque

torch, nn = try_import_torch()

def neglogp_actions(pi_logits, actions):
    return nn.functional.cross_entropy(pi_logits, actions, reduction='none')

def sample_actions(logits, device):
    u = torch.rand(logits.shape, dtype=logits.dtype).to(device)
    return torch.argmax(logits - torch.log(-torch.log(u)), dim=1)

def pi_entropy(logits):
    a0 = logits - torch.max(logits, dim=1, keepdim=True)[0]
    ea0 = torch.exp(a0)
    z0 = torch.sum(ea0, dim=1, keepdim=True)
    p0 = ea0 / z0
    return torch.sum(p0 * (torch.log(z0) - a0), axis=1)

def roll(arr):
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def unroll(arr, targetshape):
    s = arr.shape
    return arr.reshape(*targetshape, *s[1:]).swapaxes(0, 1)

def safe_mean(xs):
    return -np.inf if len(xs) == 0 else np.mean(xs)

from skimage.util import view_as_windows
import numpy as np
def pad_and_random_crop(imgs, out, pad):
    """
    Vectorized pad and random crop
    Assumes square images?
    args:
    imgs: shape (B,H,W,C)
    out: output size (e.g. 64)
    """
    # n: batch size.
    imgs = np.pad(imgs, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    n = imgs.shape[0]
    img_size = imgs.shape[1] # e.g. 64
    crop_max = img_size - out
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding window
    # combinations of size (out)
    windows = view_as_windows(imgs, (1, out, out, 1))[..., 0,:,:, 0]
    # selects a random window
    # for each batch element
    cropped = windows[np.arange(n), w1, h1]
    cropped = cropped.transpose(0,2,3,1)
    return cropped


class RetuneSelector:
    def __init__(self, nbatch, ob_space, ac_space, skips = 800_000, replay_size = 200_000, num_retunes = 5):
        self.skips = skips + (-skips) % nbatch
        self.replay_size = replay_size + (-replay_size) % nbatch
        self.exp_replay = np.empty((self.replay_size, *ob_space.shape), dtype=np.uint8)
        self.batch_size = nbatch
        self.batches_in_replay = self.replay_size // nbatch
        
        self.num_retunes = num_retunes
        self.ac_space = ac_space
        self.ob_space = ob_space
        
        self.cooldown_counter = self.skips // self.batch_size
        self.replay_index = 0
        self.buffer_full = False

    def update(self, obs_batch):
        if self.num_retunes == 0:
            return False
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        start = self.replay_index * self.batch_size
        end = start + self.batch_size
        self.exp_replay[start:end] = obs_batch
        
        self.replay_index = (self.replay_index + 1) % self.batches_in_replay
        self.buffer_full = self.buffer_full or (self.replay_index == 0)
        
        return self.buffer_full
        
    def retune_done(self):
        self.cooldown_counter = self.skips // self.batch_size
        self.num_retunes -= 1
        self.replay_index = 0
        self.buffer_full = False
    

class RewardNormalizer(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, gamma=0.99, cliprew=10.0, epsilon=1e-8):
        self.epsilon = epsilon
        self.gamma = gamma
        self.ret_rms = RunningMeanStd(shape=())
        self.cliprew = cliprew
        self.ret = 0 # size updates after first pass
        
    def normalize(self, rews):
        self.ret = self.ret * self.gamma + rews
        self.ret_rms.update(self.ret)
        rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return rews
    
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class CustomTorchPolicy(TorchPolicy):
    """Example of a random policy
    If you are using tensorflow/pytorch to build custom policies,
    you might find `build_tf_policy` and `build_torch_policy` to
    be useful.
    Adopted from examples from https://docs.ray.io/en/master/rllib-concepts.html
    """

    def __init__(self, observation_space, action_space, config):
        self.config = config

        dist_class, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"], framework="torch")
        self.model = ModelCatalog.get_model_v2(
            obs_space=observation_space,
            action_space=action_space,
            num_outputs=logit_dim,
            model_config=self.config["model"],
            framework="torch")

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
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.rewnorm = RewardNormalizer(cliprew=25.0)
        self.reward_deque = deque(maxlen=100)
        self.best_reward = -np.inf
        self.best_weights = None
        self.timesteps_total = 0
        self.target_timesteps = 8_000_000
        nbatch = self.config['train_batch_size']
        self.retune_selector = RetuneSelector(nbatch, observation_space, action_space, 
                                              skips = 300_000, replay_size = 200_000, num_retunes = 14)
        
    def _torch_tensor(self, arr):
        return torch.tensor(arr).to(self.device)
    
    def _value_function(self, arr, ret_pi=False):
        tt = self._torch_tensor(arr)
        with torch.no_grad():
            pi, _ = self.model.forward({"obs": tt}, None, None)
            v = self.model.value_function()
        if not ret_pi:
            return v.cpu().numpy()
        else:
            return v.cpu().numpy(), pi.cpu().numpy()
    
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
        
        nbatch = len(samples['dones'])
        self.timesteps_total += nbatch
        
        ## Best reward model selection
        eprews = [info['episode']['r'] for info in samples['infos'] if 'episode' in info]
        self.reward_deque.extend(eprews)
        mean_reward = safe_mean(eprews) if len(eprews) >= 100 else safe_mean(self.reward_deque)
        if self.best_reward < mean_reward:
            self.best_reward = mean_reward
            self.best_weights = self.get_weights()
           
        if self.timesteps_total > self.target_timesteps:
            if self.best_weights is not None:
                self.set_weights(self.best_weights)
                return {} # Not doing last optimization step - This is intentional due to noisy gradients
          
        obs = samples['obs']
        ## Distill with augmentation
        should_retune = self.retune_selector.update(obs)
        if should_retune:
            self.retune_with_augmentation(obs)
            return {}
         
        ## Config data values
        nbatch_train = self.config['sgd_minibatch_size']
        gamma, lam = self.config['gamma'], self.config['lambda']
        nsteps = self.config['rollout_fragment_length']
        nenvs = nbatch//nsteps
        ts = (nenvs, nsteps)
        
        ## Value prediction
        next_obs = unroll(samples['new_obs'], ts)[-1]
        last_values = self._value_function(next_obs)
        values = np.empty((nbatch,), dtype=np.float32)
        for start in range(0, nbatch, nbatch_train): # Causes OOM up if trying to do all at once (TODO: Try bigger than nbatch_train)
            end = start + nbatch_train
            values[start:end] = self._value_function(samples['obs'][start:end])
        
        ## Reward Normalization - No reward norm works well for many envs
        mb_values = unroll(values, ts)
#         mb_origrewards = unroll(samples['rewards'], ts)
#         mb_rewards =  np.zeros_like(mb_origrewards)
#         for ii in range(nsteps):
#             mb_rewards[ii] = self.rewnorm.normalize(mb_origrewards[ii])
        mb_rewards = unroll(samples['rewards'], ts)
        mb_dones = unroll(samples['dones'], ts)
        
        ## GAE
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
        lrnow = self.config['lr']
        max_grad_norm = self.config['grad_clip']
        ent_coef, vf_coef = self.config['entropy_coeff'], self.config['vf_loss_coeff']
        
#         obs = samples['obs']
        neglogpacs = -samples['action_logp'] ## np.isclose seems to be True always, otherwise compute again if needed
        actions = samples['actions']
        returns = roll(mb_returns)
        nminibatches = nbatch // nbatch_train
        noptepochs = self.config['num_sgd_iter']

        ## Train multiple epochs
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (torch.from_numpy(arr[mbinds]).to(self.device)
                          for arr in (obs, returns, actions, values, neglogpacs))
                self._batch_train(lrnow, 
                                  cliprange, vfcliprange, max_grad_norm,
                                  ent_coef, vf_coef,
                                  *slices)
                
        return {}

    def _batch_train(self, lr, 
                     cliprange, vfcliprange, max_grad_norm,
                     ent_coef, vf_coef,
                     obs, returns, actions, values, neglogpac_old):
        
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.optimizer.zero_grad()

        advs = returns - values
        advs = (advs - torch.mean(advs)) / (torch.std(advs) + 1e-8)

        pi_logits, _ = self.model.forward({"obs": obs}, None, None)
        vpred = self.model.value_function()
        neglogpac = neglogp_actions(pi_logits, actions)
        entropy = torch.mean(pi_entropy(pi_logits))

        vpredclipped = values + torch.clamp(vpred - values, -cliprange, cliprange)
        vf_losses1 = torch.pow((vpred - returns), 2)
        vf_losses2 = torch.pow((vpredclipped - returns), 2)
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))

        ratio = torch.exp(neglogpac_old - neglogpac)
        pg_losses1 = -advs * ratio
        pg_losses2 = -advs * torch.clamp(ratio, 1-cliprange, 1+cliprange)
        pg_loss = torch.mean(torch.max(pg_losses1, pg_losses2))

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
        
    def retune_with_augmentation(self, obs):
        nbatch_train = self.config['sgd_minibatch_size']
        retune_epochs = 2
        replay_size = self.retune_selector.replay_size
        replay_vf = np.empty((replay_size,), dtype=np.float32)
        replay_pi = np.empty((replay_size, self.retune_selector.ac_space.n), dtype=np.float32)

        # Store current value function and policy logits
        for start in range(0, replay_size, nbatch_train):
            end = start + nbatch_train
            replay_batch = self.retune_selector.exp_replay[start:end]
            replay_vf[start:end], replay_pi[start:end] = self._value_function(replay_batch, ret_pi=True)

        # Tune vf and pi heads to older predictions with augmented observations
        inds = np.arange(len(self.retune_selector.exp_replay))
        for ep in range(retune_epochs):
            np.random.shuffle(inds)
            for start in range(0, replay_size, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = [self.retune_selector.exp_replay[mbinds], 
                          torch.from_numpy(replay_vf[mbinds]).to(self.device), 
                          torch.from_numpy(replay_pi[mbinds]).to(self.device)]

        self.retune_selector.retune_done()
 
    def tune_policy(self, obs, target_vf, target_pi, retune_vf_loss_coeff):
        obs_aug = torch.from_numpy(pad_and_random_crop(obs, 64, 10)).to(self.device)
        with torch.no_grad():
            tpi_log_softmax = nn.functional.log_softmax(target_pi, dim=1)
            tpi_softmax = torch.exp(tpi_log_softmax)
        self.optimizer.zero_grad()
        pi_logits, _ = self.model.forward({"obs": obs_aug}, None, None)
        vpred = self.model.value_function()
        pi_log_softmax =  nn.functional.log_softmax(pi_logits, dim=1)
#         pi_loss = nn.functional.kl_div(pi_softmax, tpi_log_softmax, reduction='batchmean', log_target=True)
        # kl_div in torch 1.3.1 has numerical issues
        pi_loss = torch.mean(torch.sum(tpi_softmax * (tpi_log_softmax - pi_log_softmax) , dim=1)) 
        vf_loss = .5 * torch.mean(torch.pow(vf - target_vf, 2))
        loss = retune_vf_loss_coeff * vf_loss + pi_loss
        
        loss.backward()
        self.optimizer.step()
    
    @override(TorchPolicy)
    def get_weights(self):
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }
    
    @override(TorchPolicy)
    def set_weights(self, weights):
        weights = convert_to_torch_tensor(weights, device=self.device)
        self.model.load_state_dict(weights)