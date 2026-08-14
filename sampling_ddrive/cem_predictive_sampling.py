import os

# Use the CPU by default, while allowing callers such as benchmark scripts to
# select another JAX backend before importing this module.
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
    """Cross-Entropy Method MPC for differential-drive navigation.

    At every control step, ``num_cem_iterations`` (M) CEM updates are made.
    Each update evaluates ``num_computations`` samples and fits a diagonal
    Gaussian distribution to the ``num_elites`` lowest-cost samples.

    Obstacles are circles represented by ``[x, y, radius]``. An optional
    obstacle can be inserted near the goal as soon as the robot enters a
    configurable activation radius.
    """

    def __init__(
        self,
        horizon=80,
        dt=0.05,
        num_computations=100,
        num_cem_iterations=3,
        M=None,
        num_elites=10,
        reset_covariance_each_step=True,
        cem_alpha=1.0,
        min_std=0.01,
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
        sudden_obstacle=False,
        sudden_obstacle_distance=1.25,
        sudden_obstacle_radius=0.18,
        sudden_obstacle_offset=(-0.65, 0.0),
        sudden_obstacle_position=None,
        print_computation_time=True,
    ):
        if M is not None:
            num_cem_iterations = M
        if num_cem_iterations < 1:
            raise ValueError("num_cem_iterations (M) must be at least 1")
        if not 1 <= num_elites <= num_computations:
            raise ValueError(
                "num_elites must be between 1 and num_computations"
            )
        if not 0.0 < cem_alpha <= 1.0:
            raise ValueError("cem_alpha must be in (0, 1]")
        if min_std <= 0.0:
            raise ValueError("min_std must be positive")

        self.horizon = horizon
        self.dt = dt
        self.state_dim = 3
        self.control_dim = 2
        self.num_computations = num_computations
        self.num_cem_iterations = num_cem_iterations
        # Convenient alias matching the usual CEM notation requested for M.
        self.M = num_cem_iterations
        self.num_elites = num_elites
        self.reset_covariance_each_step = reset_covariance_each_step
        self.cem_alpha = cem_alpha
        self.min_std = min_std
        self.robot = Robot(self.dt)

        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.goal_tolerance = goal_tolerance
        self.sample_deltas = sample_deltas
        self.delta_v_max = delta_v_max
        self.delta_w_max = delta_w_max
        self.print_computation_time = print_computation_time
        self.previous_parameters = None
        self.distribution_std = None
        self.seed = seed
        self.rng_key = random.PRNGKey(seed)

        self.base_obstacles = np.asarray(
            [] if obstacles is None else obstacles, dtype=np.float32
        ).reshape((-1, 3))
        self.obstacles = jnp.asarray(self.base_obstacles)
        self.sudden_obstacle = sudden_obstacle
        self.sudden_obstacle_distance = sudden_obstacle_distance
        self.sudden_obstacle_radius = sudden_obstacle_radius
        self.sudden_obstacle_offset = np.asarray(
            sudden_obstacle_offset, dtype=np.float32
        )
        if self.sudden_obstacle_offset.shape != (2,):
            raise ValueError("sudden_obstacle_offset must contain two values")
        self.sudden_obstacle_position = (
            None
            if sudden_obstacle_position is None
            else np.asarray(sudden_obstacle_position, dtype=np.float32)
        )
        if (
            self.sudden_obstacle_position is not None
            and self.sudden_obstacle_position.shape != (2,)
        ):
            raise ValueError("sudden_obstacle_position must contain two values")
        self.sudden_obstacle_active = False
        self.active_sudden_obstacle = None

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

        self.initial_std = jnp.concatenate(
            (
                jnp.full(
                    self.num_knots,
                    self.delta_v_max if sample_deltas else 1.2,
                ),
                jnp.full(
                    self.num_knots,
                    self.delta_w_max if sample_deltas else 2.0,
                ),
            )
        )
        self.parameters_map = jnp.zeros(
            (self.num_computations, self.num_parameters), dtype=jnp.float32
        )
        self.last_candidate_parameters = self.parameters_map
        self.last_costs = jnp.full((self.num_computations,), jnp.inf)

        self._build_forward_simulator()
        vectorized_rollout = jax.vmap(
            self.compute_rollout_trajectory, in_axes=(None, 0), out_axes=0
        )
        self.jit_vectorized_rollout = jax.jit(vectorized_rollout)

        if init_jax:
            states = jnp.zeros((self.num_computations, self.state_dim))
            goal = jnp.zeros(self.state_dim)
            self.jit_vectorized_forward_sim(states, goal, self.parameters_map)

    def _build_forward_simulator(self):
        """Compile rollouts against the currently active obstacle set."""
        vectorized_forward_sim = jax.vmap(
            self.compute_forward_simulations, in_axes=(0, None, 0), out_axes=0
        )
        self.jit_vectorized_forward_sim = jax.jit(vectorized_forward_sim)

    def reset(self):
        """Reset warm starts, covariance, random generator, and surprise obstacle."""
        self.previous_parameters = None
        self.distribution_std = None
        self.rng_key = random.PRNGKey(self.seed)
        self.sudden_obstacle_active = False
        self.active_sudden_obstacle = None
        self.obstacles = jnp.asarray(self.base_obstacles)
        self._build_forward_simulator()

    def _shift_parameter_vector(self, parameters):
        """Advance one knot vector by one controller time step."""
        if self.interpolation == "zero_order":
            parameters = jnp.asarray(parameters, dtype=jnp.float32)
            values_v = parameters[: self.num_knots]
            values_w = parameters[self.num_knots :]
            shifted_v = jnp.concatenate((values_v[1:], values_v[-1:]))
            shifted_w = jnp.concatenate((values_w[1:], values_w[-1:]))
            return jnp.concatenate((shifted_v, shifted_w))

        knot_grid = np.arange(self.num_knots, dtype=np.float32)
        knot_shift = (self.num_knots - 1) / max(self.horizon - 1, 1)
        shifted_grid = np.minimum(knot_grid + knot_shift, knot_grid[-1])
        parameters = np.asarray(parameters)
        shifted_v = np.interp(
            shifted_grid, knot_grid, parameters[: self.num_knots]
        )
        shifted_w = np.interp(
            shifted_grid, knot_grid, parameters[self.num_knots :]
        )
        return jnp.asarray(
            np.concatenate((shifted_v, shifted_w)), dtype=jnp.float32
        )

    def _shift_previous_parameters(self):
        """Advance the previous optimal sequence by one control time step."""
        return self._shift_parameter_vector(self.previous_parameters)

    def _initial_distribution(self):
        """Return the warm-started mean and covariance diagonal for this step."""
        if self.sample_deltas and self.previous_parameters is not None:
            mean = self._shift_previous_parameters()
        else:
            mean = jnp.zeros(self.num_parameters, dtype=jnp.float32)

        if self.reset_covariance_each_step or self.distribution_std is None:
            std = self.initial_std
        else:
            std = self._shift_parameter_vector(self.distribution_std)
        return mean, jnp.maximum(std, self.min_std)

    def _clip_parameters(self, parameters):
        parameters_v = jnp.clip(parameters[..., : self.num_knots], -1.2, 1.2)
        parameters_w = jnp.clip(parameters[..., self.num_knots :], -2.0, 2.0)
        return jnp.concatenate((parameters_v, parameters_w), axis=-1)

    def _sample_candidates(self, mean, std):
        """Draw one CEM population and always include its current mean."""
        self.rng_key, sample_key = random.split(self.rng_key)
        noise = random.normal(
            sample_key,
            (self.num_computations, self.num_parameters),
            dtype=jnp.float32,
        )
        candidates = self._clip_parameters(mean[None, :] + noise * std[None, :])
        return candidates.at[0].set(self._clip_parameters(mean))

    def _resample_parameter_maps(self):
        """Refresh the visualization population from the current distribution."""
        mean, std = self._initial_distribution()
        self.parameters_map = self._sample_candidates(mean, std)
        self.last_candidate_parameters = self.parameters_map

    def _candidate_parameters(self):
        """Return the most recently sampled CEM population."""
        return self.last_candidate_parameters

    def _spline_position(self, step):
        """Map a rollout step to a knot segment and its local coordinate."""
        clipped_step = jnp.clip(step, 0, max(self.horizon - 1, 0))
        knot_position = (
            clipped_step * (self.num_knots - 1) / max(self.horizon - 1, 1)
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
        """Piecewise cubic Hermite interpolation through the velocity knots."""
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
            v = jnp.clip(v, -1.2, 1.2)
            w = jnp.clip(w, -2.0, 2.0)
            state_next = self.robot.integrate_jax(state, v, w)
            return state_next, state_next

        _, predicted_states = jax.lax.scan(
            integrate_step, initial_state, jnp.arange(self.horizon)
        )
        return jnp.vstack((initial_state, predicted_states))

    def get_rollout_trajectories(self, state, parameter_indices):
        """Return selected rollouts from the latest CEM population."""
        selected_parameters = self._candidate_parameters()[
            jnp.asarray(parameter_indices)
        ]
        trajectories = self.jit_vectorized_rollout(
            jnp.asarray(state, dtype=jnp.float32), selected_parameters
        )
        return np.asarray(trajectories)

    def _maybe_add_sudden_obstacle(self, state, goal):
        """Activate the optional hidden obstacle when the robot nears the goal."""
        if not self.sudden_obstacle or self.sudden_obstacle_active:
            return False
        distance_to_goal = float(np.linalg.norm(np.asarray(goal[:2] - state[:2])))
        if distance_to_goal > self.sudden_obstacle_distance:
            return False

        if self.sudden_obstacle_position is None:
            position = np.asarray(goal[:2], dtype=np.float32) + self.sudden_obstacle_offset
        else:
            position = self.sudden_obstacle_position
        obstacle = np.array(
            [position[0], position[1], self.sudden_obstacle_radius],
            dtype=np.float32,
        )
        self.active_sudden_obstacle = obstacle
        self.obstacles = jnp.asarray(
            np.vstack((np.asarray(self.obstacles), obstacle)), dtype=jnp.float32
        )
        self.sudden_obstacle_active = True
        # Obstacles are captured by the jitted cost function. Rebuilding the
        # wrapper ensures that the newly revealed obstacle is used immediately.
        self._build_forward_simulator()
        print("Ostacolo improvviso attivato:", obstacle)
        return True

    def compute_control(self, state, goal):
        """Run M CEM iterations and return ``(linear_velocity, angular_velocity)``."""
        state = jnp.asarray(state, dtype=jnp.float32)
        goal = jnp.asarray(goal, dtype=jnp.float32)
        distance = np.linalg.norm(np.asarray(goal[:2] - state[:2]))
        if distance <= self.goal_tolerance:
            return 0.0, 0.0

        start_time = time.time()
        self._maybe_add_sudden_obstacle(state, goal)
        mean, std = self._initial_distribution()
        state_batch = jnp.tile(state, (self.num_computations, 1))
        best_cost = jnp.inf
        best_parameters = mean

        for _ in range(self.M):
            candidate_parameters = self._sample_candidates(mean, std)
            costs = self.jit_vectorized_forward_sim(
                state_batch, goal, candidate_parameters
            )
            costs = jnp.nan_to_num(costs, nan=jnp.inf, posinf=jnp.inf)
            elite_indices = jnp.argsort(costs)[: self.num_elites]
            elites = candidate_parameters[elite_indices]
            elite_mean = jnp.mean(elites, axis=0)
            elite_std = jnp.sqrt(jnp.var(elites, axis=0) + self.min_std**2)
            mean = self._clip_parameters(
                (1.0 - self.cem_alpha) * mean + self.cem_alpha * elite_mean
            )
            std = jnp.maximum(
                (1.0 - self.cem_alpha) * std + self.cem_alpha * elite_std,
                self.min_std,
            )
            #mean, std = self._initial_distribution()

            iteration_best_index = jnp.argmin(costs)
            iteration_best_cost = costs[iteration_best_index]
            iteration_best = candidate_parameters[iteration_best_index]
            use_iteration_best = iteration_best_cost < best_cost
            best_cost = jnp.minimum(best_cost, iteration_best_cost)
            best_parameters = jnp.where(
                use_iteration_best, iteration_best, best_parameters
            )
            self.last_candidate_parameters = candidate_parameters
            self.parameters_map = candidate_parameters
            self.last_costs = costs

        # Keeping the best evaluated member makes the command no worse than
        # the best elite, while the fitted mean/covariance warm-start next step.
        self.previous_parameters = best_parameters
        self.distribution_std = std
        self.distribution_mean = mean
        v, w = self.control_profile_fun(best_parameters, 0)
        if self.print_computation_time:
            print(
                f"computation time ({self.M} CEM iterations):",
                time.time() - start_time,
            )
        return float(v), float(w)


def run_demo():
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
        num_cem_iterations=3,
        num_elites=10,
        reset_covariance_each_step=True,
        delta_v_max=1.2,
        delta_w_max=2.0,
        sample_deltas=True,
        sudden_obstacle=False,
    )

    state_history = [np.asarray(state)]
    control_history = []
    rollout_history = []
    obstacle_history = []
    visualization_rng = np.random.default_rng(7)
    number_of_visible_rollouts = min(10, controller.num_computations)

    for step in range(300):
        if np.linalg.norm(np.asarray(state[:2] - goal[:2])) <= controller.goal_tolerance:
            print(f"Goal raggiunto in {step} passi")
            break
        v, w = controller.compute_control(state, goal)
        visible_indices = visualization_rng.choice(
            controller.num_computations,
            size=number_of_visible_rollouts,
            replace=False,
        )
        rollout_history.append(
            controller.get_rollout_trajectories(state, visible_indices)
        )
        obstacle_history.append(np.asarray(controller.obstacles))
        control_history.append([v, w])
        state = controller.robot.integrate_jax(state, v, w)
        state_history.append(np.asarray(state))
    else:
        print("Goal non raggiunto entro il numero massimo di passi")

    state_history = np.asarray(state_history)
    control_history = np.asarray(control_history)
    output_directory = Path(__file__).resolve().parent
    gif_path = output_directory / "cem_ddrive_navigation.gif"
    initial_scene_path = output_directory / "cem_ddrive_initial_environment.png"
    controls_path = output_directory / "cem_ddrive_control_profiles.png"
    figure, axis = plt.subplots(num="CEM point-to-point obstacle avoidance")
    axis.scatter(state_history[0, 0], state_history[0, 1], marker="o", label="start")
    axis.scatter(float(goal[0]), float(goal[1]), marker="*", s=160, label="goal")

    obstacle_patches = []
    max_obstacles = len(obstacles) + int(controller.sudden_obstacle)
    for obstacle_index in range(max_obstacles):
        visible = obstacle_index < len(obstacles)
        obstacle = obstacles[obstacle_index] if visible else np.zeros(3)
        obstacle_color = (
            "tab:red" if obstacle_index < len(obstacles) else "tab:purple"
        )
        patch = plt.Circle(
            obstacle[:2],
            obstacle[2],
            color=obstacle_color,
            alpha=0.75,
            visible=visible,
        )
        axis.add_patch(patch)
        obstacle_patches.append(patch)

    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)
    axis.legend(loc="upper left")
    environment_points = np.vstack(
        (state_history[:, :2], obstacles[:, :2], np.asarray(goal[:2])[None, :])
    )
    axis.set_xlim(*(np.min(environment_points[:, 0]) - 0.8, np.max(environment_points[:, 0]) + 0.8))
    axis.set_ylim(*(np.min(environment_points[:, 1]) - 0.8, np.max(environment_points[:, 1]) + 0.8))
    figure.savefig(initial_scene_path, dpi=220, bbox_inches="tight")

    path_line, = axis.plot([], [], color="tab:blue", linewidth=2.5)
    robot_body = plt.Circle(
        state_history[0, :2], controller.robot_radius,
        color="tab:blue", alpha=0.85, zorder=5,
    )
    axis.add_patch(robot_body)
    heading_line, = axis.plot([], [], color="white", linewidth=2.0, zorder=6)
    rollout_lines = [
        axis.plot([], [], color=color, alpha=0.65, linewidth=1.0)[0]
        for color in visualization_rng.random((number_of_visible_rollouts, 3))
    ]

    def update_animation(frame):
        current_state = state_history[frame]
        path_line.set_data(state_history[: frame + 1, 0], state_history[: frame + 1, 1])
        robot_body.center = current_state[:2]
        heading_end = current_state[:2] + controller.robot_radius * np.array(
            [np.cos(current_state[2]), np.sin(current_state[2])]
        )
        heading_line.set_data(
            [current_state[0], heading_end[0]],
            [current_state[1], heading_end[1]],
        )
        rollout_index = min(frame, len(rollout_history) - 1)
        for line, rollout in zip(rollout_lines, rollout_history[rollout_index]):
            line.set_data(rollout[:, 0], rollout[:, 1])
        current_obstacles = obstacle_history[rollout_index]
        for obstacle_index, patch in enumerate(obstacle_patches):
            visible = obstacle_index < len(current_obstacles)
            patch.set_visible(visible)
            if visible:
                patch.center = current_obstacles[obstacle_index, :2]
                patch.set_radius(current_obstacles[obstacle_index, 2])
        axis.set_title(
            f"CEM: K={controller.num_computations}, M={controller.M}, "
            f"elite={controller.num_elites}, t={frame * controller.dt:.2f} s"
        )
        return [path_line, robot_body, heading_line, *rollout_lines, *obstacle_patches]

    animation = FuncAnimation(
        figure,
        update_animation,
        frames=len(state_history),
        interval=1000 / 12,
        blit=False,
        cache_frame_data=False,
    )
    animation.save(gif_path, writer=PillowWriter(fps=12), dpi=160)

    control_figure, (axis_v, axis_w) = plt.subplots(
        2, 1, sharex=True, num="CEM control profiles", constrained_layout=True
    )
    control_time = np.arange(len(control_history)) * controller.dt
    axis_v.plot(control_time, control_history[:, 0], color="tab:blue")
    axis_v.set_ylabel("v [m/s]")
    axis_v.grid(True)
    axis_w.plot(control_time, control_history[:, 1], color="tab:orange")
    axis_w.set_xlabel("time [s]")
    axis_w.set_ylabel("ω [rad/s]")
    axis_w.grid(True)
    control_figure.savefig(controls_path, dpi=220, bbox_inches="tight")
    print(f"GIF salvata in: {gif_path}")
    print(f"Immagine iniziale salvata in: {initial_scene_path}")
    print(f"Profili di controllo salvati in: {controls_path}")
    plt.show()
    return gif_path, initial_scene_path, controls_path


if __name__ == "__main__":
    run_demo()
