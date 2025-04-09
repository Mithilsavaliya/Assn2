import gymnasium as gym
import aisd_examples.envs.create_red_ball  # Required to register environment
import matplotlib.pyplot as plt


def simple_action_policy(observation):
    """Return an action that turns toward the red ball based on its pixel X position."""
    ball_x = observation  # Observation is expected to be the X coordinate of the ball (0–640)
    if ball_x < 300:
        return 290  # Turn left
    elif ball_x > 340:
        return 350  # Turn right
    else:
        return 320  # Stay straight


def main():
    env = gym.make("aisd_examples/CreateRedBall-v0")
    episode_rewards = []

    for episode in range(10):  # Run 10 episodes
        observation, info = env.reset()
        total_reward = 0

        for step in range(100):
            action = simple_action_policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} | Total Reward: {total_reward}")

    env.close()

    # Plot reward over episodes
    plt.plot(episode_rewards, marker='o')
    plt.title("Non-RL Agent Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig("non_rl_rewards.png")
    plt.show()


if __name__ == "__main__":
    main()
