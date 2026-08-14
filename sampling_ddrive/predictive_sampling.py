import os

# Use the CPU by default, while allowing callers such as the benchmark script
# to select another JAX backend before importing this module.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.animation import FuncAnimation, PillowWriter

from robot_model import Robot


print("jax.default_backend()", jax.default_backend())
print("jax.devices()", jax.devices())


class Sampling_MPC:
    """Sampling MPC for point-to-point differential-drive navigation.

    Obstacles are fixed circles represented by ``[x, y, radius]``. The
    controller receives only the current robot state and one goal pose; it does
    not require a precomputed reference trajectory.
    """

    def __init__(
        self,
        horizon=80,
        dt=0.05,
        num_computations=100,
        init_jax=True,
        interpolation="zero_order",
        obstacles=None,
        robot_radius=0.15,
        safety_margin=0.15,
        goal_tolerance=0.10,
        seed=42,
        sample_deltas=True,
        delta_v_max=0.25,
        delta_w_max=0.50,
        print_computation_time=True,
    ):
        self.horizon = horizon
        self.dt = dt
        self.state_dim = 3
        self.control_dim = 2
        self.num_computations = num_computations
        self.robot = Robot(self.dt)

        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.goal_tolerance = goal_tolerance
        self.sample_deltas = sample_deltas
        self.delta_v_max = delta_v_max
        self.delta_w_max = delta_w_max
        self.print_computation_time = print_computation_time
        self.previous_parameters = None
        self.seed = seed
        self.rng_key = random.PRNGKey(seed)
        self.obstacles = jnp.asarray(
            [] if obstacles is None else obstacles, dtype=jnp.float32
        ).reshape((-1, 3))

        self.goal_weight = 1.0
        self.terminal_goal_weight = 30.0
        self.heading_weight = 0.05
        self.control_weight = 0.01
        self.obstacle_weight = 250.0
        self.collision_weight = 10000.0

        if interpolation not in {"zero_order", "linear", "cubic"}:
            raise ValueError(
                "interpolation must be 'zero_order', 'linear', or 'cubic'"
            )
        self.interpolation = interpolation
        # Zero order samples one independent (v, w) pair per horizon step.
        # Linear and cubic interpolation use eight knots per control profile.
        self.num_knots = self.horizon if interpolation == "zero_order" else 20
        self.num_parameters = 2 * self.num_knots
        interpolation_functions = {
            "zero_order": self.compute_zero_order_hold,
            "linear": self.compute_linear_spline,
            "cubic": self.compute_cubic_spline,
        }
        self.control_profile_fun = jax.jit(
            interpolation_functions[self.interpolation]
        )

        self._resample_parameter_maps()

        vectorized_forward_sim = jax.vmap(
            self.compute_forward_simulations, in_axes=(0, None, 0), out_axes=0
        )
        self.jit_vectorized_forward_sim = jax.jit(vectorized_forward_sim)
        vectorized_rollout = jax.vmap(
            self.compute_rollout_trajectory, in_axes=(None, 0), out_axes=0
        )
        self.jit_vectorized_rollout = jax.jit(vectorized_rollout)

        if init_jax:
            states = jnp.zeros((self.num_computations, self.state_dim))
            goal = jnp.zeros(self.state_dim)
            self.jit_vectorized_forward_sim(states, goal, self.parameters_map)

    def reset(self):
        self.previous_parameters = None
        self.rng_key = random.PRNGKey(self.seed)

    def _resample_parameter_maps(self):
        """Sample a fresh candidate set and advance the internal PRNG key."""
        self.rng_key, key_v, key_w, key_delta_v, key_delta_w = random.split(
            self.rng_key, 5
        )
        v_parameters = random.uniform(
            key_v,
            (self.num_computations, self.num_knots),
            minval=-1.2,
            maxval=1.2,
        )
        w_parameters = random.uniform(
            key_w,
            (self.num_computations, self.num_knots),
            minval=-2.0,
            maxval=2.0,
        )
        self.parameters_map = jnp.column_stack((v_parameters, w_parameters))

        delta_v_parameters = random.uniform(
            key_delta_v,
            (self.num_computations, self.num_knots),
            minval=-self.delta_v_max,
            maxval=self.delta_v_max,
        )
        delta_w_parameters = random.uniform(
            key_delta_w,
            (self.num_computations, self.num_knots),
            minval=-self.delta_w_max,
            maxval=self.delta_w_max,
        )
        self.delta_parameters_map = jnp.column_stack(
            (delta_v_parameters, delta_w_parameters)
        )

    def _shift_previous_parameters(self):
        """Advance the previous knot sequence by one controller time step."""
        if self.interpolation == "zero_order":
            previous = jnp.asarray(self.previous_parameters, dtype=jnp.float32)
            previous_v = previous[: self.num_knots]
            previous_w = previous[self.num_knots :]
            shifted_v = jnp.concatenate((previous_v[1:], previous_v[-1:]))
            shifted_w = jnp.concatenate((previous_w[1:], previous_w[-1:]))
            return jnp.concatenate((shifted_v, shifted_w))

        knot_grid = np.arange(self.num_knots, dtype=np.float32)
        knot_shift = (self.num_knots - 1) / max(self.horizon - 1, 1)
        shifted_grid = np.minimum(knot_grid + knot_shift, knot_grid[-1])
        previous = np.asarray(self.previous_parameters)
        shifted_v = np.interp(
            shifted_grid, knot_grid, previous[: self.num_knots]
        )
        shifted_w = np.interp(
            shifted_grid, knot_grid, previous[self.num_knots :]
        )
        return jnp.asarray(np.concatenate((shifted_v, shifted_w)), dtype=jnp.float32)

    def _candidate_parameters(self):
        """Build absolute samples or delta samples around the previous optimum."""
        if not self.sample_deltas or self.previous_parameters is None:
            return self.parameters_map

        nominal_parameters = self._shift_previous_parameters()
        candidates = nominal_parameters[None, :] + self.delta_parameters_map
        candidate_v = jnp.clip(candidates[:, : self.num_knots], -1.2, 1.2)
        candidate_w = jnp.clip(candidates[:, self.num_knots :], -2.0, 2.0)
        return jnp.column_stack((candidate_v, candidate_w))

    def _spline_position(self, step):
        """Map a rollout step to a knot segment and its local coordinate."""
        clipped_step = jnp.clip(step, 0, max(self.horizon - 1, 0))
        knot_position = (
            clipped_step
            * (self.num_knots - 1)
            / max(self.horizon - 1, 1)
        )
        segment = jnp.floor(knot_position).astype(jnp.int32)
        segment = jnp.clip(segment, 0, self.num_knots - 2)
        alpha = jnp.clip(knot_position - segment, 0.0, 1.0)
        return segment, alpha

    def compute_zero_order_hold(self, parameters, step):
        """Return the independently sampled control for one horizon step."""
        clipped_step = jnp.clip(step, 0, max(self.horizon - 1, 0))
        knot_index = clipped_step.astype(jnp.int32)
        v = parameters[knot_index]
        w = parameters[self.num_knots + knot_index]
        return v, w

    def compute_linear_spline(self, parameters, step):
        """Piecewise-linear interpolation through every velocity knot."""
        segment, alpha = self._spline_position(step)
        v_knots = parameters[: self.num_knots]
        w_knots = parameters[self.num_knots :]

        v = (1.0 - alpha) * v_knots[segment] + alpha * v_knots[segment + 1]
        w = (1.0 - alpha) * w_knots[segment] + alpha * w_knots[segment + 1]
        return v, w

    def compute_cubic_spline(self, parameters, step):
        """Piecewise cubic Hermite interpolation through every velocity knot.

        Centred finite differences define the internal tangents; one-sided
        differences are used at the two endpoints. The resulting profile is
        continuous in both value and first derivative (C1).
        """
        segment, alpha = self._spline_position(step)

        def interpolate(knots):
            previous_index = jnp.maximum(segment - 1, 0)
            next_next_index = jnp.minimum(segment + 2, self.num_knots - 1)
            p0 = knots[previous_index]
            p1 = knots[segment]
            p2 = knots[segment + 1]
            p3 = knots[next_next_index]

            tangent_1 = jnp.where(
                segment == 0, p2 - p1, 0.5 * (p2 - p0)
            )
            tangent_2 = jnp.where(
                segment == self.num_knots - 2,
                p2 - p1,
                0.5 * (p3 - p1),
            )

            alpha_2 = alpha * alpha
            alpha_3 = alpha_2 * alpha
            h00 = 2.0 * alpha_3 - 3.0 * alpha_2 + 1.0
            h10 = alpha_3 - 2.0 * alpha_2 + alpha
            h01 = -2.0 * alpha_3 + 3.0 * alpha_2
            h11 = alpha_3 - alpha_2
            return h00 * p1 + h10 * tangent_1 + h01 * p2 + h11 * tangent_2

        v = interpolate(parameters[: self.num_knots])
        w = interpolate(parameters[self.num_knots :])
        return v, w

    def _obstacle_cost(self, state):
        delta = state[:2] - self.obstacles[:, :2]
        centre_distance = jnp.sqrt(jnp.sum(delta**2, axis=1) + 1e-8)
        clearance = centre_distance - self.obstacles[:, 2] - self.robot_radius

        safety_violation = jnp.maximum(self.safety_margin - clearance, 0.0)
        soft_cost = self.obstacle_weight * jnp.sum(safety_violation**2)
        collision_cost = self.collision_weight * jnp.sum(clearance <= 0.0)
        return soft_cost + collision_cost

    def compute_forward_simulations(self, initial_state, goal, parameters):
        """Return the cost of one candidate control rollout."""

        def iterate_fun(step, carry):
            cost, state = carry
            v, w = self.control_profile_fun(parameters, step)
            v = jnp.clip(v, -1.2, 1.2)
            w = jnp.clip(w, -2.0, 2.0)
            state_next = self.robot.integrate_jax(state, v, w)

            position_error = state_next[:2] - goal[:2]
            goal_cost = self.goal_weight * jnp.sum(position_error**2)
            effort_cost = self.control_weight * (v**2 + 0.1 * w**2)
            obstacle_cost = self._obstacle_cost(state_next)
            return cost + goal_cost + effort_cost + obstacle_cost, state_next

        cost, final_state = jax.lax.fori_loop(
            0, self.horizon, iterate_fun, (0.0, initial_state)
        )
        final_position_error = final_state[:2] - goal[:2]
        heading_error = jnp.arctan2(
            jnp.sin(final_state[2] - goal[2]),
            jnp.cos(final_state[2] - goal[2]),
        )
        return (
            cost
            + self.terminal_goal_weight * jnp.sum(final_position_error**2)
            + self.heading_weight * heading_error**2
        )

    def compute_rollout_trajectory(self, initial_state, parameters):
        """Simulate one rollout and return all its predicted states."""

        def integrate_step(state, step):
            v, w = self.control_profile_fun(parameters, step)
            v = jnp.clip(v, -1.0, 1.0)
            w = jnp.clip(w, -1.0, 1.0)
            state_next = self.robot.integrate_jax(state, v, w)
            return state_next, state_next

        _, predicted_states = jax.lax.scan(
            integrate_step, initial_state, jnp.arange(self.horizon)
        )
        return jnp.vstack((initial_state, predicted_states))

    def get_rollout_trajectories(self, state, parameter_indices):
        """Return selected sampled rollouts for visualization only."""
        candidate_parameters = self._candidate_parameters()
        selected_parameters = candidate_parameters[jnp.asarray(parameter_indices)]
        trajectories = self.jit_vectorized_rollout(
            jnp.asarray(state, dtype=jnp.float32), selected_parameters
        )
        return np.asarray(trajectories)

    def compute_control(self, state, goal):
        """Compute ``(linear_velocity, angular_velocity)`` toward one goal."""
        state = jnp.asarray(state, dtype=jnp.float32)
        goal = jnp.asarray(goal, dtype=jnp.float32)

        distance = np.linalg.norm(np.asarray(goal[:2] - state[:2]))
        if distance <= self.goal_tolerance:
            return 0.0, 0.0
        start_time = time.time()
        self._resample_parameter_maps()
        candidate_parameters = self._candidate_parameters()
        state_batch = jnp.tile(state, (self.num_computations, 1))
        costs = self.jit_vectorized_forward_sim(
            state_batch, goal, candidate_parameters
        )
        best_parameters = candidate_parameters[jnp.nanargmin(costs)]
        if self.sample_deltas:
            self.previous_parameters = best_parameters
        v, w = self.control_profile_fun(best_parameters, 0)
        if self.print_computation_time:
            print("computation time:", time.time() - start_time)
        return float(v), float(w)


def run_demo():
    # Set to True to sample delta-v and delta-w around the previous solution.
    obstacles = np.array(
        [
            [1.20, 0.35, 0.28],
            [2.15, 1.05, 0.35],
            [3.05, 1.35, 0.30],
        ],
        dtype=np.float32,
    )
    goal = jnp.array([4.0, 2.0, 0.0])
    state = jnp.array([0.0, 0.0, 0.0])
    controller = Sampling_MPC(
        obstacles=obstacles,
        interpolation="linear",
        delta_v_max=0.05,
        delta_w_max=0.05,
        sample_deltas=True,
    )

    state_history = [np.asarray(state)]
    control_history = []
    rollout_history = []
    visualization_rng = np.random.default_rng(7)
    number_of_visible_rollouts = min(10, controller.num_computations)

    for step in range(300):
        if np.linalg.norm(np.asarray(state[:2] - goal[:2])) <= controller.goal_tolerance:
            print(f"Goal raggiunto in {step} passi")
            break

        visible_indices = visualization_rng.choice(
            controller.num_computations,
            size=number_of_visible_rollouts,
            replace=False,
        )
        rollout_history.append(
            controller.get_rollout_trajectories(state, visible_indices)
        )
        v, w = controller.compute_control(state, goal)
        control_history.append([v, w])
        state = controller.robot.integrate_jax(state, v, w)
        state_history.append(np.asarray(state))
    else:
        print("Goal non raggiunto entro il numero massimo di passi")

    visible_indices = visualization_rng.choice(
        controller.num_computations,
        size=number_of_visible_rollouts,
        replace=False,
    )
    rollout_history.append(controller.get_rollout_trajectories(state, visible_indices))

    state_history = np.asarray(state_history)
    control_history = np.asarray(control_history)
    output_directory = Path(__file__).resolve().parent
    gif_path = output_directory / "ddrive_navigation.gif"
    initial_scene_path = output_directory / "ddrive_initial_environment.png"
    controls_path = output_directory / "ddrive_control_profiles.png"
    figure_dpi = 220
    gif_dpi = 160

    figure, axis = plt.subplots(num="Point-to-point obstacle avoidance")
    axis.scatter(state_history[0, 0], state_history[0, 1], marker="o", label="start")
    axis.scatter(float(goal[0]), float(goal[1]), marker="*", s=160, label="goal")

    for obstacle_x, obstacle_y, obstacle_radius in obstacles:
        axis.add_patch(
            plt.Circle((obstacle_x, obstacle_y), obstacle_radius, color="tab:red", alpha=0.6)
        )
        axis.add_patch(
            plt.Circle(
                (obstacle_x, obstacle_y),
                obstacle_radius + controller.robot_radius + controller.safety_margin,
                fill=False,
                linestyle="--",
                color="tab:red",
                alpha=0.5,
            )
        )

    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)
    axis.legend(loc="upper left")

    environment_points = np.vstack((state_history[:, :2], obstacles[:, :2], np.asarray(goal[:2])[None, :]))
    lower_bounds = np.min(environment_points, axis=0) - 0.8
    upper_bounds = np.max(environment_points, axis=0) + 0.8
    axis.set_xlim(lower_bounds[0], upper_bounds[0])
    axis.set_ylim(lower_bounds[1], upper_bounds[1])

    path_line, = axis.plot([], [], color="tab:blue", linewidth=2.5, label="robot path")
    robot_body = plt.Circle(
        (state_history[0, 0], state_history[0, 1]),
        controller.robot_radius,
        color="tab:blue",
        alpha=0.85,
        zorder=5,
    )
    axis.add_patch(robot_body)
    heading_line, = axis.plot([], [], color="white", linewidth=2.0, zorder=6)

    initial_heading_end = state_history[0, :2] + controller.robot_radius * np.array(
        [np.cos(state_history[0, 2]), np.sin(state_history[0, 2])]
    )
    heading_line.set_data(
        [state_history[0, 0], initial_heading_end[0]],
        [state_history[0, 1], initial_heading_end[1]],
    )
    #axis.set_title(f"Sampling MPC")
    figure.savefig(initial_scene_path, dpi=figure_dpi, bbox_inches="tight")
    print(f"Immagine iniziale salvata in: {initial_scene_path}")

    rollout_colors = visualization_rng.random((number_of_visible_rollouts, 3))
    rollout_lines = [
        axis.plot([], [], color=color, alpha=0.65, linewidth=1.0)[0]
        for color in rollout_colors
    ]

    def update_animation(frame):
        current_state = state_history[frame]
        path_line.set_data(state_history[: frame + 1, 0], state_history[: frame + 1, 1])
        robot_body.center = (current_state[0], current_state[1])
        heading_end = current_state[:2] + controller.robot_radius * np.array(
            [np.cos(current_state[2]), np.sin(current_state[2])]
        )
        heading_line.set_data(
            [current_state[0], heading_end[0]],
            [current_state[1], heading_end[1]],
        )
        for line, rollout in zip(rollout_lines, rollout_history[frame]):
            line.set_data(rollout[:, 0], rollout[:, 1])
        axis.set_title(
            f"K = {controller.num_computations}, "
            f"t = {frame * controller.dt:.2f} s"
        )
        return [path_line, robot_body, heading_line, *rollout_lines]

    animation = FuncAnimation(
        figure,
        update_animation,
        frames=len(state_history),
        interval=1000 / 12,
        blit=False,
        cache_frame_data=False,
    )
    animation.save(gif_path, writer=PillowWriter(fps=12), dpi=gif_dpi)
    print(f"GIF salvata in: {gif_path}")

    control_figure, (axis_v, axis_w) = plt.subplots(
        2, 1, sharex=True, num="Control profiles", constrained_layout=True
    )
    control_time = np.arange(len(control_history)) * controller.dt
    axis_v.plot(control_time, control_history[:, 0], color="tab:blue")
    axis_v.set_ylabel("v [m/s]")
    axis_v.grid(True)
    axis_w.plot(control_time, control_history[:, 1], color="tab:orange")
    axis_w.set_xlabel("time [s]")
    axis_w.set_ylabel("ω [rad/s]")
    axis_w.grid(True)
    control_figure.savefig(controls_path, dpi=figure_dpi, bbox_inches="tight")
    print(f"Profili di controllo salvati in: {controls_path}")

    plt.show()
    return gif_path, initial_scene_path, controls_path


if __name__ == "__main__":
    run_demo()
