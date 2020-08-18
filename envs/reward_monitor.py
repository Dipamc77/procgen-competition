from gym.spaces import Box
from gym import Wrapper
import numpy as np

from envs.procgen_env_wrapper import ProcgenEnvWrapper
from ray.tune import registry

class RewardMonitor(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.epret = 0
        self.eplen = 0

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.epret = 0
        self.eplen = 0
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.epret += reward
        self.eplen += 1
        newinfo = info.copy()
        if done:
            epinfo = {'r': self.epret, 'l': self.eplen}
            newinfo['episode'] = epinfo
            self.epret = 0
            self.eplen = 0
        return observation, reward, done, newinfo

    
registry.register_env(
    "reward_monitor",
    lambda config: RewardMonitor(ProcgenEnvWrapper(config))
)