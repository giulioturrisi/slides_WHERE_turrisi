"""Vanilla policy gradient for the fully actuated double pendulum.

Install: python3 -m pip install numpy matplotlib
Quick check: python3 vanilla_policy_gradient_double_pendulum.py --updates 2 --no-plot
Run training: python3 vanilla_policy_gradient_double_pendulum.py

This is REINFORCE with reward-to-go and a Gaussian policy whose mean is a
linear combination of fixed nonlinear features. There is no critic, baseline,
return normalization, entropy bonus, or gradient clipping.
"""

import argparse

import numpy as np


SEED = 7
UPDATES = 3000
BATCH_SIZE = 256
HORIZON = 120
DT = 0.04
GAMMA = 0.97
LEARNING_RATE = 0.03
SIGMA = 0.45
MAX_TORQUE = 12.0
TARGET = np.array([np.pi, 0.0])
EVAL_EPISODES = 64
EVAL_EVERY = 10


def wrap(angle):
    """Map angles to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def reset(batch_size, rng):
    """Start a batch near the hanging position."""
    return np.concatenate((rng.normal(0, 0.08, (batch_size, 2)),
                           rng.normal(0, 0.04, (batch_size, 2))), axis=1)


def step(state, torque):
    """Vectorized double-pendulum dynamics for one control interval."""
    # The first axis stores independent trajectories from the same batch.
    angles = state[:, :2].copy()
    velocity = state[:, 2:].copy()
    substep = DT / 4
    # Several semi-implicit Euler substeps improve numerical stability.
    for _ in range(4):
        difference = angles[:, 0] - angles[:, 1]
        sine, cosine = np.sin(difference), np.cos(difference)
        rhs1 = (torque[:, 0] - torque[:, 1] - sine * velocity[:, 1] ** 2
                - 2 * 9.81 * np.sin(angles[:, 0]) - 0.3 * velocity[:, 0])
        rhs2 = (torque[:, 1] + sine * velocity[:, 0] ** 2
                - 9.81 * np.sin(angles[:, 1]) - 0.3 * velocity[:, 1])
        determinant = 2 - cosine ** 2
        acceleration = np.column_stack(((rhs1 - cosine * rhs2) / determinant,
                                         (2 * rhs2 - cosine * rhs1) / determinant))
        velocity += substep * acceleration
        angles = wrap(angles + substep * velocity)

    next_state = np.column_stack((angles, velocity))
    error = wrap(angles - TARGET)
    cost = (np.sum(error ** 2, axis=1) + 0.03 * np.sum(velocity ** 2, axis=1)
            + 0.001 * np.sum(torque ** 2, axis=1))
    return next_state, -cost / HORIZON


def features(state):
    """Fixed bounded nonlinear basis functions for the policy."""
    # These features are fixed; only the policy weights are learned.
    error = wrap(state[:, :2] - TARGET)
    velocity = np.tanh(state[:, 2:] / 3.0)
    return np.column_stack((
        np.ones(len(state)),
        np.sin(error), np.cos(error), velocity,
        np.sin(error[:, 0] - error[:, 1]),
        np.cos(error[:, 0] - error[:, 1]),
        np.sin(error[:, 0] + error[:, 1]),
        np.cos(error[:, 0] + error[:, 1]),
    ))


N_FEATURES = features(np.zeros((1, 4))).shape[1]


def rollout(weights, rng, batch_size, stochastic=True):
    """Collect on-policy trajectories with weights frozen for the batch."""
    state = reset(batch_size, rng)
    states = np.empty((HORIZON + 1, batch_size, 4))
    rewards = np.empty((HORIZON, batch_size))
    scores = np.empty((HORIZON, batch_size, N_FEATURES, 2))
    states[0] = state
    for time_step in range(HORIZON):
        phi = features(state)
        mean = phi @ weights
        # Sample in an unconstrained latent space, then bound each torque.
        noise = rng.normal(size=mean.shape) if stochastic else np.zeros_like(mean)
        latent_action = mean + SIGMA * noise
        torque = MAX_TORQUE * np.tanh(latent_action)
        # Score of the Gaussian policy with respect to its mean weights.
        scores[time_step] = phi[:, :, None] * (noise / SIGMA)[:, None, :]
        state, rewards[time_step] = step(state, torque)
        states[time_step + 1] = state
    return states, rewards, scores


def reward_to_go(rewards):
    """Compute discounted Monte Carlo returns separately per episode."""
    returns = np.empty_like(rewards)
    running = np.zeros(rewards.shape[1])
    # Backward recursion computes G_t without storing every future sum.
    for time_step in reversed(range(len(rewards))):
        running = rewards[time_step] + GAMMA * running
        returns[time_step] = running
    return returns


def policy_gradient(rewards, scores):
    """Estimate the vanilla policy gradient with reward-to-go."""
    returns = reward_to_go(rewards)
    discount = GAMMA ** np.arange(len(rewards))
    # Average score-weighted returns over the batch and ascend the objective.
    return np.einsum("tb,tbfa,t->fa", returns, scores, discount) / rewards.shape[1]


def evaluate(weights, stochastic):
    """Evaluate the policy on fixed initial-state samples."""
    # Fixed seeds make different policy checkpoints directly comparable.
    _, rewards, _ = rollout(weights, np.random.default_rng(2026),
                            EVAL_EPISODES, stochastic=stochastic)
    return rewards.sum(axis=0).mean()


def train(updates):
    """Train the policy and return weights plus diagnostic histories."""
    rng = np.random.default_rng(SEED)
    weights = np.zeros((N_FEATURES, 2))
    train_mean, train_std, objective_history = [], [], []
    eval_updates = [0]
    eval_stochastic = [evaluate(weights, True)]
    eval_mean_action = [evaluate(weights, False)]

    for update in range(1, updates + 1):
        _, rewards, scores = rollout(weights, rng, BATCH_SIZE)
        # Vanilla policy gradient uses raw returns: no critic or baseline.
        weights += LEARNING_RATE * policy_gradient(rewards, scores)
        if not np.isfinite(weights).all():
            raise FloatingPointError("Nonfinite policy weights")
        # Training reward is the undiscounted episode sum, while the update
        # above optimizes the discounted reward-to-go objective.
        episode_rewards = rewards.sum(axis=0)
        train_mean.append(episode_rewards.mean())
        train_std.append(episode_rewards.std())
        objective_history.append(np.mean(
            np.sum(rewards * (GAMMA ** np.arange(HORIZON))[:, None], axis=0)
        ))
        # Evaluate both the noisy policy and its deterministic mean-action form.
        if update % EVAL_EVERY == 0 or update == updates:
            eval_updates.append(update)
            eval_stochastic.append(evaluate(weights, True))
            eval_mean_action.append(evaluate(weights, False))
        if update % 50 == 0 or update == updates:
            print(f"Update {update:4d}/{updates} | "
                  f"training reward {train_mean[-1]: .3f} | "
                  f"mean-action evaluation {eval_mean_action[-1]: .3f}")

    history = (np.asarray(train_mean), np.asarray(train_std),
               np.asarray(objective_history), np.asarray(eval_updates),
               np.asarray(eval_stochastic), np.asarray(eval_mean_action))
    return weights, history


def plot_history(history):
    import matplotlib.pyplot as plt

    train_mean, train_std, objective, eval_updates, eval_stochastic, eval_mean = history
    updates = np.arange(1, len(train_mean) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(updates, train_mean, alpha=0.65, label="Training batch mean")
    axes[0].fill_between(updates, train_mean - train_std,
                         train_mean + train_std, alpha=0.12,
                         label="Training episodes: +/- 1 std")
    axes[0].plot(eval_updates, eval_stochastic, lw=2,
                 label="Stochastic evaluation")
    axes[0].plot(eval_updates, eval_mean, lw=2,
                 label="Mean-action evaluation")
    axes[0].set(xlabel="Policy update", ylabel="Episode reward",
                title="Reward over training")
    axes[0].legend(fontsize=8)
    axes[1].plot(updates, objective)
    axes[1].set(xlabel="Policy update", ylabel="Mean discounted return",
                title="Sampled training objective")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def coordinates(angles):
    """Return Cartesian coordinates for the two links."""
    return (np.array([0, np.sin(angles[0]),
                      np.sin(angles[0]) + np.sin(angles[1])]),
            np.array([0, -np.cos(angles[0]),
                      -np.cos(angles[0]) - np.cos(angles[1])]))


def visualize(weights):
    """Animate the zero-torque and learned mean-action policies."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Use the same initial state for both trajectories.
    rng = np.random.default_rng(9001)
    initial_weights = np.zeros_like(weights)
    initial_states, _, _ = rollout(initial_weights, rng, 1, stochastic=False)
    learned_states, _, _ = rollout(weights, np.random.default_rng(9001),
                                   1, stochastic=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    lines = []
    for axis, title in zip(
            axes, ["Initial policy: zero torque", "Learned mean-action policy"]):
        axis.plot(*coordinates(TARGET), "o--", color="gray", alpha=0.6,
                   label="Target")
        line, = axis.plot([], [], "o-", lw=3, markersize=9,
                          label="Pendulum")
        lines.append(line)
        axis.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), title=title,
                 xlabel="x (m)", ylabel="y (m)")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
    clock_text = fig.suptitle("")
    fig.tight_layout()

    def animate(frame):
        for line, trajectory in zip(lines, [initial_states, learned_states]):
            line.set_data(*coordinates(trajectory[frame, 0, :2]))
        clock_text.set_text(f"Time: {frame * DT:.2f} s")
        return [*lines, clock_text]

    # Skip every other frame to keep the animation responsive.
    animation = FuncAnimation(fig, animate, frames=range(0, HORIZON + 1, 2),
                              interval=2000 * DT, blit=False)
    plt.show()
    return animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be at least 1")
    learned_weights, training_history = train(args.updates)
    print(f"\nStochastic evaluation: "
          f"{training_history[4][0]:.3f} -> {training_history[4][-1]:.3f}")
    print(f"Mean-action evaluation: "
          f"{training_history[5][0]:.3f} -> {training_history[5][-1]:.3f}")
    if not args.no_plot:
        plot_history(training_history)
    visualize(learned_weights)
