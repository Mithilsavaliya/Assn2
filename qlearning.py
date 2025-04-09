import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import aisd_examples.envs.create_red_ball  # Register env

env = gym.make("aisd_examples/CreateRedBall-v0")

# Parameters
num_episodes = 200
alpha = 0.1       # learning rate
gamma = 0.99      # discount factor
epsilon = 1.0     # exploration rate
epsilon_decay = 0.995
min_epsilon = 0.05

# Assuming discrete observation/action space
obs_space_size = env.observation_space.n
action_space_size = env.action_space.n

q_table = np.zeros((obs_space_size, action_space_size))
rewards_per_episode = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0

    for _ in range(100):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[obs])

        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Q-learning update
        q_table[obs, action] = q_table[obs, action] + alpha * (
            reward + gamma * np.max(q_table[next_obs]) - q_table[obs, action])

        obs = next_obs
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)
    print(f"Episode {episode} - Total Reward: {total_reward}")

env.close()

# Plotting
plt.plot(rewards_per_episode)
plt.title("Q-learning Total Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.savefig("qlearning_rewards.png")
plt.show()
