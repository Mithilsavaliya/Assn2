import gymnasium as gym
import aisd_examples.envs.create_red_ball  # Register env
import matplotlib.pyplot as plt

env = gym.make("aisd_examples/CreateRedBall-v0")
obs, _ = env.reset()

step_limit = 100
returns = []
total_reward = 0

for step in range(step_limit):
    ball_x = obs  # This is the pixel x-position of red ball

    # Heuristic rule: center is at 320, adjust accordingly
    if ball_x < 300:
        action = 250  # turn left
    elif ball_x > 340:
        action = 390  # turn right
    else:
        action = 320  # go straight (no turn)

    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    returns.append(total_reward)

    if terminated or truncated:
        break

env.close()

# Plot
plt.plot(returns)
plt.title("Non-RL Agent Performance")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.grid()
plt.savefig("non_rl_rewards.png")
plt.show()
