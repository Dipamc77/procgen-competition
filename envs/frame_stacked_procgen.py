import numpy as np

from gym.spaces import Box
from gym import Wrapper

from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper
from envs.reward_monitor import RewardMonitor

class FrameStackByChannels(Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        
        low = np.tile(env.observation_space.low, num_stack)
        high = np.tile(env.observation_space.high, num_stack)
        
        self.frames = low.copy()

        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.stackedobs = np.roll(self.stackedobs, shift=-observation.shape[-1], axis=-1)
        self.stackedobs[...,-observation.shape[-1]:] = observation
        return self.stackedobs, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.stackedobs = np.tile(observation, self.num_stack)
        return self.stackedobs

class FasterFrameStack2(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        low = np.tile(env.observation_space.low, 2)
        high = np.tile(env.observation_space.high, 2)
        
        self.frames = low.copy()
        self.old_obs = env.observation_space.low.copy()
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.stackedobs[...,:-observation.shape[-1]] = self.old_obs.copy()
        self.stackedobs[...,-observation.shape[-1]:] = observation
        self.old_obs = observation
        return self.stackedobs.copy(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.stackedobs = np.tile(observation, 2)
        return self.stackedobs.copy()
    
def maybe_framestack(config):
    config_copy = config.copy()
    fs = config_copy.pop('frame_stack')
    if fs == 2:
        return FasterFrameStack2(RewardMonitor(ProcgenEnvWrapper(config_copy)))
    elif fs > 1:
        return FrameStackByChannels(RewardMonitor(ProcgenEnvWrapper(config_copy)), fs)
    else:
        return RewardMonitor(ProcgenEnvWrapper(config_copy))
    
# Register Env in Ray
registry.register_env("frame_stacked_procgen", maybe_framestack)