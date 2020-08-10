from ray.rllib.policy.torch_policy import TorchPolicy
import numpy as np
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, convert_to_torch_tensor
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override

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
        self.rewnorm = RewardNormalizer()
        
    def _torch_tensor(self, arr):
        return torch.tensor(arr).to(self.device)
    
    def _value_function(self, arr):
        tt = self._torch_tensor(arr)
        with torch.no_grad():
            self.model.forward({"obs": tt}, None, None)
            v = self.model.value_function()
        return v.cpu().numpy()
    
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
        nbatch_train = self.config['sgd_minibatch_size']
        gamma, lam = self.config['gamma'], self.config['lambda']
        nsteps = self.config['rollout_fragment_length']
        nenvs = nbatch//nsteps
        ts = (nenvs, nsteps)
        
        next_obs = unroll(samples['new_obs'], ts)[-1]
        last_values = self._value_function(next_obs)
        values = np.empty((nbatch,), dtype=np.float32)
        for start in range(0, nbatch, nbatch_train):
            end = start + nbatch_train
            values[start:end] = self._value_function(samples['obs'][start:end])
        
        mb_values = unroll(values, ts)
        mb_origrewards = unroll(samples['rewards'], ts)
        mb_rewards =  np.zeros_like(mb_origrewards)
        for ii in range(nsteps):
            mb_rewards[ii] = self.rewnorm.normalize(mb_origrewards[ii])
        mb_dones = unroll(samples['dones'], ts)
        
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
        
        
        cliprange, vfcliprange = self.config['clip_param'], self.config['vf_clip_param']
        lrnow = self.config['lr']
        max_grad_norm = self.config['grad_clip']
        ent_coef, vf_coef = self.config['entropy_coeff'], self.config['vf_loss_coeff']
        
        obs = samples['obs']
        ## np.isclose seems to be True always, otherwise compute again if needed
        neglogpacs = -samples['action_logp'] 
        actions = samples['actions']
        returns = roll(mb_returns)
        values = roll(mb_values)
        nminibatches = nbatch // nbatch_train
        noptepochs = self.config['num_sgd_iter']

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
        
#         with torch.no_grad():
#             approxkl = .5 * torch.mean(torch.pow((neglogpac - neglogpac_old), 2))
#             clipfrac = torch.mean((torch.abs(ratio - 1.0) > cliprange).float())
            
#         return pg_loss.item(), vf_loss.item(), entropy.item(), approxkl.item(), clipfrac.item()
    
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