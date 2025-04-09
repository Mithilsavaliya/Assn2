import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'aisd_examples')))

from aisd_examples.envs.create_red_ball import CreateRedBallEnv

env = CreateRedBallEnv()

observation, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Step: obs={observation}, reward={reward}, done={terminated or truncated}")
    if terminated or truncated:
        observation, info = env.reset()

env.close()
