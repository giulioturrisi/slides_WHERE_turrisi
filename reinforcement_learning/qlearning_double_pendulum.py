"""Tabular Q-learning for the fully actuated double pendulum.

Install: python3 -m pip install numpy matplotlib
Run a quick check: python3 qlearning_double_pendulum.py --episodes 5 --no-plot --no-render
Run training: python3 qlearning_double_pendulum.py --episodes 30000

The two absolute link angles are measured from the downward vertical. Both
joints are actuated. Since the state is continuous, Q-learning uses a
discretized state: two angle bins and two angular-velocity bins.
"""

import argparse

import numpy as np


ANGLE_BINS = 12
VELOCITY_BINS = 8
VELOCITY_LIMIT = 8.0
TORQUES = np.array([-12.0, 0.0, 12.0], dtype=np.float64)
ACTION_TORQUES = np.array(
    [(first, second) for first in TORQUES for second in TORQUES],
    dtype=np.float64,
)
ALPHA = 0.1
GAMMA = 0.97
EPSILON_START = 1.0
EPSILON_END = 0.05
SEED = 7
DT = 0.04
HORIZON = 120
TARGET = np.array([np.pi, 0.0])


def wrap(angle):
    """Map angles to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def reset(rng):
    """Start close to the hanging configuration."""
    return np.concatenate((rng.normal(0, 0.08, 2), rng.normal(0, 0.04, 2)))


def step(state, torque):
    """Integrate one control interval and return (next_state, reward)."""
    # Each state contains two absolute angles followed by two angular speeds.
    angles = state[:2].copy()
    velocity = state[2:].copy()
    substep = DT / 4
    # Multiple semi-implicit Euler substeps make the simulation more stable.
    for _ in range(4):
        difference = angles[0] - angles[1]
        sine, cosine = np.sin(difference), np.cos(difference)
        rhs1 = (torque[0] - torque[1] - sine * velocity[1] ** 2
                - 2 * 9.81 * np.sin(angles[0]) - 0.3 * velocity[0])
        rhs2 = (torque[1] + sine * velocity[0] ** 2
                - 9.81 * np.sin(angles[1]) - 0.3 * velocity[1])
        determinant = 2 - cosine ** 2
        acceleration = np.array(
            [(rhs1 - cosine * rhs2) / determinant,
             (2 * rhs2 - cosine * rhs1) / determinant]
        )
        velocity += substep * acceleration
        angles = wrap(angles + substep * velocity)

    next_state = np.concatenate((angles, velocity))
    # The reward is the negative regulation cost, so values closer to zero
    # correspond to better tracking of the target posture.
    error = wrap(angles - TARGET)
    cost = (np.sum(error ** 2) + 0.03 * np.sum(velocity ** 2)
            + 0.001 * np.sum(torque ** 2))
    return next_state, -cost / HORIZON


def discretize(state):
    """Convert [q1, q2, dq1, dq2] into a four-dimensional table index."""
    # Angles are periodic; angular velocities are clipped to the table range.
    angles = wrap(state[:2])
    angle_indices = ((angles + np.pi) / (2 * np.pi) * ANGLE_BINS).astype(int)
    angle_indices %= ANGLE_BINS
    velocity_indices = ((np.clip(state[2:], -VELOCITY_LIMIT, VELOCITY_LIMIT)
                         + VELOCITY_LIMIT)
                        / (2 * VELOCITY_LIMIT) * VELOCITY_BINS).astype(int)
    velocity_indices = np.clip(velocity_indices, 0, VELOCITY_BINS - 1)
    return tuple(np.concatenate((angle_indices, velocity_indices)))


def train(episodes):
    """Train a tabular Q-function with epsilon-greedy exploration."""
    rng = np.random.default_rng(SEED)
    # The final axis stores one Q value for each pair of joint torques.
    q = np.zeros((ANGLE_BINS, ANGLE_BINS, VELOCITY_BINS, VELOCITY_BINS,
                  len(ACTION_TORQUES)))
    returns = []
    for episode in range(episodes):
        state = reset(rng)
        total_reward = 0.0
        fraction = episode / max(1, episodes - 1)
        epsilon = EPSILON_START + fraction * (EPSILON_END - EPSILON_START)
        #epsilon = 0.3
        for _ in range(HORIZON):
            state_index = discretize(state)
            # Explore randomly early in training, then use the greedy action
            # more often as epsilon decreases.
            if rng.random() < epsilon:
                action = int(rng.integers(len(ACTION_TORQUES)))
            else:
                action = int(np.argmax(q[state_index]))
            next_state, reward = step(state, ACTION_TORQUES[action])
            next_index = discretize(next_state)
            # Tabular Q-learning Bellman update:
            # Q(s,a) <- Q(s,a) + alpha [r + gamma max Q(s',a') - Q(s,a)].
            target = reward + GAMMA * np.max(q[next_index])
            q[state_index + (action,)] += ALPHA * (
                target - q[state_index + (action,)]
            )
            state = next_state
            total_reward += reward
        returns.append(total_reward)
        if (episode + 1) % 100 == 0 or episode == episodes - 1:
            print(f"Episode {episode + 1:5d}/{episodes} | "
                  f"mean return (last 100): {np.mean(returns[-100:]):8.3f} | "
                  f"epsilon: {epsilon:.3f}")
    return q, np.asarray(returns)


def plot_rewards(returns):
    import matplotlib.pyplot as plt

    # The moving average makes the long-term training trend easier to see.
    window = min(100, len(returns))
    moving_average = np.convolve(
        returns, np.ones(window) / window, mode="valid"
    )
    fig, ax = plt.subplots()
    ax.plot(returns, alpha=0.25, label="Episode return")
    ax.plot(np.arange(window - 1, len(returns)), moving_average,
            label=f"Moving average ({window})")
    ax.set_title("Double pendulum Q-learning")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (higher is better)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def visualize(q, episodes=3):
    """Show learned trajectories as an HTML animation in a notebook or plot."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    rng = np.random.default_rng(SEED + 1000)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), aspect="equal")
    line, = ax.plot([], [], "o-", lw=3, color="tab:blue")
    title = ax.set_title("")
    frames = []
    for episode in range(episodes):
        state = reset(rng)
        trajectory = []
        total_reward = 0.0
        for _ in range(HORIZON):
            # Evaluation is fully greedy: exploration is disabled here.
            q1, q2 = state[:2]
            trajectory.append((np.array([0, np.sin(q1), np.sin(q1) + np.sin(q2)]),
                               np.array([0, -np.cos(q1), -np.cos(q1) - np.cos(q2)])))
            action = int(np.argmax(q[discretize(state)]))
            state, reward = step(state, ACTION_TORQUES[action])
            total_reward += reward
        print(f"Demo {episode + 1}: return = {total_reward:.3f}")
        frames.extend(trajectory)

    def update(index):
        x, y = frames[index]
        line.set_data(x, y)
        title.set_text(f"Learned policy, frame {index + 1}/{len(frames)}")
        return line, title

    # Keep the animation object alive until the window finishes rendering.
    animation = FuncAnimation(fig, update, frames=len(frames), interval=30,
                              blit=True)
    plt.show()
    return animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    q_table, returns = train(args.episodes)
    if not args.no_plot:
        plot_rewards(returns)
    if not args.no_render:
        visualize(q_table)