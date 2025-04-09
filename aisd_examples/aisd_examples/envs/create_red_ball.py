import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node

class CreateRedBallEnv(gym.Env):
    def __init__(self):
        super(CreateRedBallEnv, self).__init__()

        rclpy.init()
        self.node = rclpy.create_node('create_redball_env')

        # Arbitrary spaces
        self.observation_space = spaces.Discrete(100)
        self.action_space = spaces.Discrete(3)  # e.g., left, stay, right

        self.current_state = 0
        self.step_counter = 0

    def reset(self, seed=None, options=None):
        self.step_counter = 0
        self.current_state = 50
        return self.current_state, {}

    def step(self, action):
        self.current_state = np.random.randint(0, 100)
        self.step_counter += 1
        terminated = self.step_counter >= 100
        truncated = False
        reward = 0  # Placeholder
        return self.current_state, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
