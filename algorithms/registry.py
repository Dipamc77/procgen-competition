"""
Registry of custom implemented algorithms names

Please refer to the following examples to add your custom algorithms : 

- AlphaZero : https://github.com/ray-project/ray/tree/master/rllib/contrib/alpha_zero
- bandits : https://github.com/ray-project/ray/tree/master/rllib/contrib/bandits
- maddpg : https://github.com/ray-project/ray/tree/master/rllib/contrib/maddpg
- random_agent: https://github.com/ray-project/ray/tree/master/rllib/contrib/random_agent

An example integration of the random agent is shown here : 
- https://github.com/AIcrowd/neurips2020-procgen-starter-kit/tree/master/algorithms/custom_random_agent
"""


def _import_custom_random_agent():
    from .custom_random_agent.custom_random_agent import CustomRandomAgent
    return CustomRandomAgent

def _import_random_policy():
    from .random_policy.trainer import RandomPolicyTrainer
    return RandomPolicyTrainer

def _import_custom_ppo_agent():
    from .custom_ppo.ppo import PPOTrainer
    return PPOTrainer

def _import_custom_torch_agent():
    from .custom_torch_agent.ppo import PPOTrainer
    return PPOTrainer

def _import_custom_torch_ppg():
    from .custom_ppg.ppg import PPGTrainer
    return PPGTrainer


CUSTOM_ALGORITHMS = {
    "custom/CustomRandomAgent": _import_custom_random_agent,
    "RandomPolicy": _import_random_policy,
    "CustomPPOAgent": _import_custom_ppo_agent,
    "CustomTorchPPOAgent": _import_custom_torch_agent,
    "CustomTorchPPGAgent": _import_custom_torch_ppg
}
