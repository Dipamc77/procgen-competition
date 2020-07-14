from collections import deque
import numpy as np

from gym.spaces import Box
from gym import Wrapper

from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

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
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        self.stackedobs[...,-observation.shape[-1]:] = observation
        return self.stackedobs, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.stackedobs = np.tile(observation, self.num_stack)
        return self.stackedobs
    
    
# Register Env in Ray
registry.register_env(
    "frame_stacked_procgen",  # This should be different from procgen_env_wrapper
    lambda config: FrameStackByChannels(ProcgenEnvWrapper(config), 4)
)