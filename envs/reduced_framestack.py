import numpy as np
from gym.spaces import Box
from gym import Wrapper
from skimage.measure import block_reduce

from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

class ReducedFrameStack(Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.drop = 2
        wos = env.observation_space  # wrapped ob space
        assert len(wos.shape) == 3, "Only supports images"
        assert np.prod(wos.shape[:2]) % (self.drop*self.drop) == 0, "Only supports for 4x4 reduction"
        
        self.flatlen = np.prod(wos.shape)
#         self.oldlen = np.prod(wos.shape[:2])//16
        self.oldlen = np.prod(wos.shape)//(self.drop*self.drop)
        oblen = self.flatlen + self.oldlen * (num_stack - 1)
        
        lval, hval = np.min(wos.low), np.max(wos.high)
        low = np.ones((1,1,oblen), wos.dtype)*lval
        high = np.ones((1,1,oblen), wos.dtype)*hval
        self.observation_space = Box(low=low, high=high, dtype=wos.dtype)

        self.stackedobs = np.zeros(low.shape, wos.dtype)
        self.prev_obs = np.zeros(wos.shape, wos.dtype)
        
        self.newobssh = (1, 1, self.flatlen)
        self.oldobssh = (1, 1, self.oldlen)
    

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.stackedobs[...,self.flatlen + self.oldlen:] = self.stackedobs[...,self.flatlen:-self.oldlen]
#         gray = np.mean(self.prev_obs, axis=-1)
        gray_res = block_reduce(self.prev_obs, (self.drop, self.drop, 1), np.mean)
        self.stackedobs[...,self.flatlen:self.flatlen + self.oldlen] = np.reshape(gray_res, self.oldobssh)
        self.stackedobs[...,:self.flatlen] = np.reshape(observation, self.newobssh)
        self.prev_obs = observation
        return self.stackedobs.copy(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.stackedobs[...] = 0
        self.stackedobs[...,:self.flatlen] = np.reshape(observation, self.newobssh)
        self.prev_obs = observation
        return self.stackedobs.copy()
    
# Register Env in Ray
registry.register_env(
    "reduced_frame_stack",
    lambda config: ReducedFrameStack(ProcgenEnvWrapper(config), 4)
)