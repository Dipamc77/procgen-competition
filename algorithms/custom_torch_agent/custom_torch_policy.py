from ray.rllib.policy import Policy
import numpy as np
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, convert_to_torch_tensor
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog

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

class CustomTorchPolicy(Policy):
    """Example of a random policy
    If you are using tensorflow/pytorch to build custom policies,
    you might find `build_tf_policy` and `build_torch_policy` to
    be useful.
    Adopted from examples from https://docs.ray.io/en/master/rllib-concepts.html
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

        # You can replace this with whatever variable you want to save
        # the state of the policy in. `get_weights` and `set_weights`
        # are used for checkpointing the states and restoring the states
        # from a checkpoint.
        model = ModelCatalog.get_model_v2(obs_space=observation_space,
                                           action_space=action_space,
                                           num_outputs=action_space.n,
                                           model_config=config["model"],
                                           framework="torch")
        self.framework = "torch"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
    def _torch_tensor(self, arr):
        return torch.tensor(arr).to(self.device)

    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        """Return the action for a batch
        Returns:
            action_batch: List of actions for the batch
            rnn_states: List of RNN states if any
            info: Additional info
        """
        rnn_states = []
        info = {}
        obs_tensor = self._torch_tensor(obs_batch)
        with torch.no_grad():
            pi_logits, vf = self.model.forward(obs_tensor)
            actions = sample_actions(pi_logits, self.device)
            neglogp = neglogp_actions(pi_logits, actions)
        
        info['neglogpacs'] = neglogp.tolist()
        info['values'] = vf.tolist()
        action_batch = actions.tolist()
#         import pdb; pdb.set_trace()

        return action_batch, rnn_states, info

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
        gamma, lam = self.config['gamma'], self.config['lambda']
        nsteps = self.config['rollout_fragment_length']
        nenvs = nbatch//nsteps
        ts = (nenvs, nsteps)
        
        next_obs = self._torch_tensor(unroll(samples['new_obs'], ts)[-1])
        with torch.no_grad():
            _, last_values = self.model.forward(next_obs)
        last_values = last_values.cpu().numpy()
        
        
        mb_values = unroll(samples['values'], ts)
        mb_rewards = unroll(samples['rewards'], ts)
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
        
#         import pdb; pdb.set_trace()
        
        cliprange, vfcliprange = self.config['clip_param'], self.config['vf_clip_param']
        lrnow = self.config['lr']
        max_grad_norm = self.config['grad_clip']
        ent_coef, vf_coef = self.config['vf_loss_coeff'], self.config['entropy_coeff']
        
        obs = samples['obs']
        neglogpacs = samples['neglogpacs']
        actions = samples['actions']
        returns = roll(mb_returns)
        values = roll(mb_values)
        nbatch_train = self.config['sgd_minibatch_size']
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

        pi_logits, vpred = self.model.forward(obs)
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

    def get_weights(self):
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }

    def set_weights(self, weights):
        weights = convert_to_torch_tensor(weights, device=self.device)
        self.model.load_state_dict(weights)