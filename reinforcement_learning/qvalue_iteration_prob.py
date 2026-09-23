"""Q-value iteration on the 4x4 Gridworld shown in the reference picture.

Install: python3 -m pip install matplotlib
Run:     python3 qvalue_iteration_prob.py

Coordinates are (x, y), starting at (1, 1) in the bottom-left corner.
Actions succeed with probability 0.8. With probability 0.1 each, the agent
moves in either perpendicular direction instead (no backward slip).
Hitting a wall leaves the agent in the same cell, without redistributing
that outcome's probability. Rewards belong to the CURRENT state R(s):
    (4, 4): +1, terminal (treasure)
    (4, 3): -1, terminal (fire)
    all other cells: 0
Terminal rewards are received once, with no continuation: Q(terminal, a) = R(terminal).
The start cell (2, 1) is marked S, but Q-value iteration updates ALL state-action pairs.
"""

SIZE = 4
GAMMA = 0.9  # Discount future rewards: reaching the treasure sooner is better.
ITERATIONS = 10
PLOT_EVERY = 1
START = (2, 1)
TERMINALS = {(4, 4): 1.0, (4, 3): -1.0}
ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]
SUCCESS_PROB = 0.8
STATES = [(x, y) for y in range(1, SIZE + 1) for x in range(1, SIZE + 1)]


def transition(state, action):
    """Return the next state and current-state reward R(s) for one movement."""
    if state in TERMINALS:
        return state, TERMINALS[state]  # Terminal payoff; no continuation in Bellman.
    x, y = state
    dx, dy = action
    next_state = (min(SIZE, max(1, x + dx)), min(SIZE, max(1, y + dy)))
    reward = TERMINALS.get(state, 0.0)
    return next_state, reward


def outcomes(state, action):
    """Yield (probability, next state, reward) for the three possible moves."""
    dx, dy = action
    slip_prob = (1.0 - SUCCESS_PROB) / 2.0
    for probability, movement in [
        (SUCCESS_PROB, action),
        (slip_prob, (-dy, dx)),  # Rotate the chosen direction 90 degrees.
        (slip_prob, (dy, -dx)),  # Rotate it -90 degrees.
    ]:
        next_state, reward = transition(state, movement)
        yield probability, next_state, reward


def initial_q_values():
    """Use terminal payoffs as boundary values for every action slot."""
    return {(state, action): TERMINALS.get(state, 0.0)
            for state in STATES for action in ACTIONS}


def bellman_update(q_values):
    """Update all state-action pairs synchronously using the known model."""
    new_q_values = {}
    for state in STATES:
        for action in ACTIONS:
            if state in TERMINALS:
                # No actions are executed here: these slots store the final payoff.
                new_q_values[state, action] = TERMINALS[state]
                continue
            # Q_new(s,a) = R(s) + gamma * sum P(s'|s,a) * max_b Q_old(s',b).
            # Choose the best future action separately for each successor.
            # Average over exact probabilities; no sampled experience is used.
            new_q_values[state, action] = sum(
                probability * (reward + GAMMA * max(
                    q_values[next_state, next_action] for next_action in ACTIONS
                ))
                for probability, next_state, reward in outcomes(state, action)
            )
    return new_q_values


def plot_q_values(ax, q_values, iteration):
    """Show each action value in its movement direction inside the cell."""
    from matplotlib.patches import Rectangle

    ax.clear()
    for x, y in STATES:
        state = (x, y)
        color = "white"
        if state in TERMINALS:
            color = "#d8f3dc" if TERMINALS[state] > 0 else "#ffdad6"
        elif state == START:
            color = "#dceeff"
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                               facecolor=color, edgecolor="black"))
        if state in TERMINALS:
            ax.text(x, y, f"Terminal\nQ = R = {TERMINALS[state]:+.0f}",
                    ha="center", va="center", fontsize=10)
            continue
        ax.plot([x - 0.5, x + 0.5], [y - 0.5, y + 0.5], color="0.8", lw=0.7)
        ax.plot([x - 0.5, x + 0.5], [y + 0.5, y - 0.5], color="0.8", lw=0.7)
        best = max(q_values[state, action] for action in ACTIONS)
        for action in ACTIONS:
            dx, dy = action
            value = q_values[state, action]
            is_best = abs(value - best) < 1e-10
            ax.text(x + 0.32 * dx, y + 0.32 * dy, f"{value:.3f}",
                    ha="center", va="center", fontsize=9,
                    color="#126b36" if is_best else "black",
                    fontweight="bold" if is_best else "normal")
        if state == START:
            ax.text(x, y, "S", ha="center", va="center", fontsize=9)
    ax.set(xlim=(0.5, SIZE + 0.5), ylim=(0.5, SIZE + 0.5),
           xticks=range(1, SIZE + 1), yticks=range(1, SIZE + 1),
           xlabel="x", ylabel="y",
           title=f"Stochastic Q-value iteration — sweep {iteration}\n"
                 f"gamma = {GAMMA}, action success = {SUCCESS_PROB:.0%}\n"
                 "Q values: up / down / left / right; best actions in green")
    ax.set_aspect("equal")


def main():
    import matplotlib.pyplot as plt

    assert ITERATIONS >= 1 and PLOT_EVERY >= 1
    q_values = initial_q_values()
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.text(0.5, 0.02, "Current-state rewards R(s); terminal payoff received once.",
             ha="center", fontsize=10)
    plot_q_values(ax, q_values, 0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show(block=False)
    plt.pause(1.0)
    for iteration in range(1, ITERATIONS + 1):
        if not plt.fignum_exists(fig.number):
            return
        new_q_values = bellman_update(q_values)
        delta = max(abs(new_q_values[key] - q_values[key]) for key in q_values)
        q_values = new_q_values
        if iteration % PLOT_EVERY == 0 or iteration == ITERATIONS:
            print(f"Sweep {iteration:3d}: max Q-value change = {delta:.6f}")
            plot_q_values(ax, q_values, iteration)
            plt.pause(1.0)
    # Increase ITERATIONS if the final change is still large.
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
