import gymnasium as gym
import aisd_examples.envs.create_red_ball  # register env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

# Create environment
env = gym.make("aisd_examples/CreateRedBall-v0")

# Check environment compliance (optional but helpful)
check_env(env)

# Train the PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate the trained model
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
plt.title("PPO Agent Performance")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.grid()
plt.savefig("ppo_rewards.png")
plt.show()
