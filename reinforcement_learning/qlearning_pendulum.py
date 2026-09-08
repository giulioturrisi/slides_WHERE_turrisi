"""A small, educational example of tabular Q-learning on Pendulum-v1.

Install: python3 -m pip install "gymnasium[classic-control]" numpy matplotlib
Run:     python3 qlearning_pendulum.py
Quick check without windows:
    python3 qlearning_pendulum.py --episodes 10 --no-render --no-plot

The goal is to swing the pendulum up and keep it upright. Rewards are
negative: a return closer to zero means better control. Discretization makes
tabular Q-learning possible, but also limits the accuracy of the controller.
"""

import argparse

import gymnasium as gym  # Gymnasium is the maintained successor to Gym.
import numpy as np


# Try changing these values to see how learning changes.
ANGLE_BINS = 40
VELOCITY_BINS = 40
TORQUES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
ALPHA = 0.1  # Learning rate: how much each new target changes a Q value.
GAMMA = 0.99  # Discount factor: how much future rewards matter.
SEED = 42


def discretize(observation):
    """Convert [cos(angle), sin(angle), angular velocity] into two indices."""
    cosine, sine, velocity = observation
    angle = np.arctan2(sine, cosine)  # Zero means upright.
    # Map the angle to a circular grid: -pi and +pi represent the same pose.
    angle_index = int((angle + np.pi) / (2 * np.pi) * ANGLE_BINS) % ANGLE_BINS
    # Pendulum-v1 limits angular velocity to [-8, 8] rad/s.
    velocity_index = int((velocity + 8.0) / 16.0 * VELOCITY_BINS)
    velocity_index = min(max(velocity_index, 0), VELOCITY_BINS - 1)
    return angle_index, velocity_index


def train(episodes):
    rng = np.random.default_rng(SEED)
    # Q[angle bin, velocity bin, action] estimates discounted future reward.
    q = np.zeros((ANGLE_BINS, VELOCITY_BINS, len(TORQUES)))
    env = gym.make("Pendulum-v1")  # No rendering during training: much faster.
    recent_returns = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=SEED if episode == 0 else None)
            state = discretize(observation)
            total_reward = 0.0
            # Start with random exploration; gradually favor learned actions.
            #epsilon = max(0.05, 1.0 - episode / max(1, 0.8 * episodes))
            epsilon = 0.3
            while True:
                if rng.random() < epsilon:
                    action = int(rng.integers(len(TORQUES)))
                else:
                    action = int(np.argmax(q[state]))

                # Gym expects a continuous torque array of shape (1,).
                observation, reward, terminated, truncated, _ = env.step(
                    np.array([TORQUES[action]], dtype=np.float32)
                )
                next_state = discretize(observation)

                # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
                # Bootstrap at time limits (truncated), but not true terminal states.
                future_value = 0.0 if terminated else np.max(q[next_state])
                target = reward + GAMMA * future_value
                q[state + (action,)] += ALPHA * (target - q[state + (action,)])

                state = next_state
                total_reward += reward
                if terminated or truncated:
                    break

            recent_returns.append(total_reward)
            if (episode + 1) % 100 == 0 or episode == episodes - 1:
                print(f"Episode {episode + 1:5d}/{episodes} | "
                      f"mean return (last 100): {np.mean(recent_returns[-100:]):8.1f} | "
                      f"epsilon: {epsilon:.2f}")
    finally:
        env.close()
    return q, recent_returns


def plot_rewards(returns):
    """Plot training returns and a moving average to make the trend clearer."""
    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(returns) + 1)
    # Use a shorter window when training for fewer than 100 episodes.
    window = min(100, len(returns))
    moving_average = np.convolve(returns, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots()
    ax.plot(episodes, returns, alpha=0.3, label="Episode return")
    # The first average belongs to the last episode of the first full window.
    ax.plot(episodes[window - 1:], moving_average,
            label=f"Moving average ({window} episodes)")
    ax.set_title("Pendulum Q-learning: training rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward (higher is better)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    # Close the plot window to continue to the pendulum demonstration.
    plt.show()
    plt.close(fig)


def visualize(q):
    # Human mode opens a window and handles animation timing automatically.
    env = gym.make("Pendulum-v1", render_mode="human")
    try:
        for episode in range(3):
            observation, _ = env.reset(seed=SEED + 1000 + episode)
            total_reward = 0.0
            while True:
                # Evaluation uses only the best known action: no exploration.
                action = int(np.argmax(q[discretize(observation)]))
                observation, reward, terminated, truncated, _ = env.step(
                    np.array([TORQUES[action]], dtype=np.float32)
                )
                total_reward += reward
                if terminated or truncated:
                    break
            print(f"Demo {episode + 1}: return = {total_reward:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="Skip the reward plot")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    try:
        q_table, returns = train(args.episodes)
        if not args.no_plot:
            plot_rewards(returns)
        if not args.no_render:
            visualize(q_table)
    except KeyboardInterrupt:
        print("\nStopped by user.")
