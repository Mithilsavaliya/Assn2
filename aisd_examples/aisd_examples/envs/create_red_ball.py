import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CreateRedBallEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(640)  # pixel x position (0-639)
        self.action_space = spaces.Discrete(3)         # left, stay, right
        self.current_step = 0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        obs = np.random.randint(0, 640)
        return obs, {}

    def step(self, action):
        self.current_step += 1
        obs = np.random.randint(0, 640)
        reward = 0
        terminated = self.current_step >= 100
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
