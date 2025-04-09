import gymnasium as gym
import aisd_examples.envs.create_red_ball  # ensures env registration
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

# Create and check environment
env = gym.make("aisd_examples/CreateRedBall-v0")
check_env(env)

# Train DQN agent
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate DQN agent
obs, _ = env.reset()
rewards = []
total_reward = 0

for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    rewards.append(total_reward)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()

# Plotting
plt.plot(rewards)
plt.title("DQN Agent Performance")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.grid()
plt.savefig("dqn_rewards.png")
plt.show()
