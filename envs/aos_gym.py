import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AOSGym(gym.Env):
    def __init__(self, fobj, dim=10, pop_size=50):
        super().__init__()
        self.fobj = fobj
        self.dim = dim
        self.pop_size = pop_size
        self.action_space = spaces.Discrete(4)       # 4 operators
        self.observation_space = spaces.Box(-1,1,(pop_size,dim+1),dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        self.pop = np.random.uniform(-100,100,(self.pop_size,self.dim))
        self.fit = np.array([self.fobj(x) for x in self.pop])
        return self._get_state(), {}

    def step(self, action):
        # tạm thời chỉ copy population
        self.pop = np.copy(self.pop)
        reward = -np.mean(self.fit)
        done = False
        return self._get_state(), reward, done, False, {}

    def _get_state(self):
        norm_fit = (self.fit - self.fit.min()) / (self.fit.max()-self.fit.min()+1e-9)
        return np.concatenate([self.pop/100, norm_fit[:,None]], axis=1)
