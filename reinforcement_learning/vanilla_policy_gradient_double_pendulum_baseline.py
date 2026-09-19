"""Vanilla policy gradient with a learned value baseline.

Install: python3 -m pip install numpy matplotlib
Quick check: python3 vanilla_policy_gradient_double_pendulum_baseline.py --updates 2 --no-plot --no-render

The actor is the same Gaussian policy used by
vanilla_policy_gradient_double_pendulum.py. A linear value baseline is fitted
on each batch to the reward-to-go, and the actor uses the resulting advantage.
"""

import argparse

import numpy as np

from vanilla_policy_gradient_double_pendulum import (
    BATCH_SIZE,
    DT,
    EVAL_EPISODES,
    EVAL_EVERY,
    GAMMA,
    HORIZON,
    LEARNING_RATE,
    N_FEATURES,
    SIGMA,
    features,
    evaluate,
    plot_history,
    rollout,
    visualize,
    reward_to_go,
)


UPDATES = 3000
BASELINE_REGULARIZATION = 1e-4


def fit_baseline(states, returns):
    """Fit V(s) to Monte Carlo reward-to-go targets with ridge regularization."""
    # Use the same fixed features as the policy, but learn separate value weights.
    design = features(states.reshape(-1, 4))
    targets = returns.reshape(-1)
    # Ridge regularization keeps the least-squares system well-conditioned.
    regularizer = BASELINE_REGULARIZATION * np.eye(N_FEATURES)
    baseline_weights = np.linalg.solve(
        design.T @ design + regularizer, design.T @ targets
    )
    prediction = design @ baseline_weights
    return prediction.reshape(returns.shape), baseline_weights


def baseline_policy_gradient(rewards, scores, advantages):
    """Estimate the policy gradient using baseline-subtracted returns."""
    # Subtracting V(s) reduces gradient variance without changing its expectation.
    discount = GAMMA ** np.arange(len(rewards))
    return np.einsum("tb,tbfa,t->fa", advantages, scores, discount) / rewards.shape[1]


def train(updates):
    """Train the actor and return weights plus diagnostic histories."""
    rng = np.random.default_rng(7)
    weights = np.zeros((N_FEATURES, 2))
    train_mean, train_std, objective_history = [], [], []
    baseline_error_history = []
    eval_updates = [0]
    eval_stochastic = [evaluate(weights, True)]
    eval_mean_action = [evaluate(weights, False)]

    for update in range(1, updates + 1):
        # Collect on-policy trajectories with the current actor parameters.
        states, rewards, scores = rollout(weights, rng, BATCH_SIZE)
        returns = reward_to_go(rewards)
        # Fit the baseline on this batch, then compute the policy advantages.
        baseline, _ = fit_baseline(states[:-1], returns)
        advantages = returns - baseline
        # Update the actor by stochastic gradient ascent.
        gradient = baseline_policy_gradient(rewards, scores, advantages)
        weights += LEARNING_RATE * gradient
        if not np.isfinite(weights).all():
            raise FloatingPointError("Nonfinite policy weights")

        episode_rewards = rewards.sum(axis=0)
        train_mean.append(episode_rewards.mean())
        train_std.append(episode_rewards.std())
        objective_history.append(np.mean(
            np.sum(rewards * (GAMMA ** np.arange(HORIZON))[:, None], axis=0)
        ))
        baseline_error_history.append(np.mean((returns - baseline) ** 2))

        # Evaluate periodically without affecting the training random stream.
        if update % EVAL_EVERY == 0 or update == updates:
            eval_updates.append(update)
            eval_stochastic.append(evaluate(weights, True))
            eval_mean_action.append(evaluate(weights, False))
        if update % 50 == 0 or update == updates:
            print(f"Update {update:4d}/{updates} | "
                  f"training reward {train_mean[-1]: .3f} | "
                  f"mean-action evaluation {eval_mean_action[-1]: .3f} | "
                  f"baseline MSE {baseline_error_history[-1]: .4f}")

    history = (np.asarray(train_mean), np.asarray(train_std),
               np.asarray(objective_history), np.asarray(eval_updates),
               np.asarray(eval_stochastic), np.asarray(eval_mean_action),
               np.asarray(baseline_error_history))
    return weights, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be at least 1")

    learned_weights, training_history = train(args.updates)
    print(f"\nStochastic evaluation: "
          f"{training_history[4][0]:.3f} -> {training_history[4][-1]:.3f}")
    print(f"Mean-action evaluation: "
          f"{training_history[5][0]:.3f} -> {training_history[5][-1]:.3f}")
    if not args.no_plot:
        plot_history(training_history[:6])
    if not args.no_render:
        visualize(learned_weights)
