"""Value iteration on the 4x4 Gridworld shown in the reference picture.

Install: python3 -m pip install matplotlib
Run:     python3 value_iteration.py

Coordinates are (x, y), starting at (1, 1) in the bottom-left corner.
Actions are deterministic: up, down, left, right. Hitting a wall leaves
the agent in the same cell. Rewards are received when ENTERING a cell:
    (4, 4): +1, terminal (treasure)
    (4, 3): -1, terminal (fire)
    all other cells: 0
The start cell (2, 1) is marked S, but value iteration updates ALL states.
"""

SIZE = 4
GAMMA = 0.9  # Discount future rewards: reaching the treasure sooner is better.
ITERATIONS = 10
PLOT_EVERY = 1
START = (2, 1)
TERMINALS = {(4, 4): 1.0, (4, 3): -1.0}
ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]
STATES = [(x, y) for y in range(1, SIZE + 1) for x in range(1, SIZE + 1)]


def transition(state, action):
    """Return the next state and reward for a deterministic action."""
    if state in TERMINALS:
        return state, 0.0  # The episode has ended; no further rewards.
    x, y = state
    dx, dy = action
    next_state = (min(SIZE, max(1, x + dx)), min(SIZE, max(1, y + dy)))
    reward = TERMINALS.get(next_state, 0.0)
    return next_state, reward


def bellman_update(values):
    """Perform one synchronous sweep of the Bellman optimality equation."""
    new_values = {}
    for state in STATES:
        if state in TERMINALS:
            # V(terminal) = 0: its entry reward has already been received.
            new_values[state] = 0.0
            continue
        action_values = []
        for action in ACTIONS:
            next_state, reward = transition(state, action)
            # Q(s,a) = R(s,a,s') + gamma * V(s'). No sampling is needed.
            action_values.append(reward + GAMMA * values[next_state])
        new_values[state] = max(action_values)
    # Every state uses the PREVIOUS sweep, independent of iteration order.
    return new_values


def plot_values(ax, values, iteration):
    """Draw a table with coordinates matching the reference picture."""
    ax.clear()
    for x, y in STATES:
        state = (x, y)
        color = "white"
        label = f"V = {values[state]:.3f}"
        if state in TERMINALS:
            reward = TERMINALS[state]
            color = "#d8f3dc" if reward > 0 else "#ffdad6"
            label += f"\nR = {reward:+.0f} (terminal)"
        elif state == START:
            color = "#dceeff"
            label += "\nS (start)"
        # Matplotlib is only needed for plotting, not for the algorithm.
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                               facecolor=color, edgecolor="black"))
        ax.text(x, y, label, ha="center", va="center", fontsize=10)
    ax.set(xlim=(0.5, SIZE + 0.5), ylim=(0.5, SIZE + 0.5),
           xticks=range(1, SIZE + 1), yticks=range(1, SIZE + 1),
           xlabel="x", ylabel="y",
           title=f"Value iteration — sweep {iteration} (gamma = {GAMMA})")
    ax.set_aspect("equal")


def main():
    import matplotlib.pyplot as plt

    values = {state: 0.0 for state in STATES}
    plt.ion()  # Refresh the same window as value iteration progresses.
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.text(0.5, 0.02, "Rewards are paid on entry; terminal values are zero.",
             ha="center", fontsize=10)
    plot_values(ax, values, 0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show(block=False)
    # Keep the GUI responsive while waiting for a key in the plot window.
    # Mouse clicks do not advance the algorithm.
    waiting_for_key = False

    def on_key(event):
        nonlocal waiting_for_key
        waiting_for_key = False

    connection = fig.canvas.mpl_connect("key_press_event", on_key)
    print("Click the plot window to focus it, then press a key to start.")
    while waiting_for_key and plt.fignum_exists(fig.number):
        plt.pause(0.1)
    fig.canvas.mpl_disconnect(connection)

    for iteration in range(1, ITERATIONS + 1):
        if not plt.fignum_exists(fig.number):
            return  # Closing the window stops the demonstration.
        new_values = bellman_update(values)
        delta = max(abs(new_values[s] - values[s]) for s in STATES)
        values = new_values
        if iteration % PLOT_EVERY == 0 or iteration == ITERATIONS:
            print(f"Sweep {iteration:3d}: max value change = {delta:.6f}")
            plot_values(ax, values, iteration)
            plt.pause(1.0)

    # This tiny deterministic world converges before sweep 10. We keep doing
    # sweeps to show that subsequent snapshots remain unchanged at convergence.
    plt.ioff()
    plt.show()  # Keep the final table visible until the user closes it.


if __name__ == "__main__":
    main()
