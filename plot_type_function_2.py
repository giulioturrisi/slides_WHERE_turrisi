import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def quadratic_cost_u(u):
	"""Caso 1: funzione convessa 2D (u, J)."""
	return 0.08 * (u + 1.0) ** 2 + 0.6


def nonconvex_cost_u(u):
	"""Caso 2: funzione non convessa 2D (u, J) con molte valli locali."""
	base = 0.05 * u**2 + 2.2
	ripple_1 = 0.90 * np.sin(2.5 * u)
	ripple_2 = 0.35 * np.cos(5.4 * u + 0.4)
	well = -1.40 * np.exp(-((u - 2.2) ** 2) / 0.85)
	return base + ripple_1 + ripple_2 + well


def mixed_nonconvex_cost(u, k):
	"""Caso 3 (copiato): non convessa mixed-integer con u continuo e k intero."""
	base = 0.035 * (u - 0.7 * k) ** 2 + 0.12 * (k - 1.0) ** 2
	ripple_xk = 0.95 * np.sin(3.3 * u + 0.8 * k) * np.cos(0.7 * u - 1.5 * k)
	ripple_hi = 0.42 * np.cos(5.2 * u - 0.3 * k)
	ripple_k = 0.28 * np.cos(2.2 * k)
	return base + ripple_xk + ripple_hi + ripple_k


def nonconvex_grad_u(u):
	"""Derivata analitica del caso 2 rispetto a u."""
	well_exp = np.exp(-((u - 2.2) ** 2) / 0.85)
	return (
		0.10 * u
		+ 2.25 * np.cos(2.5 * u)
		- 1.89 * np.sin(5.4 * u + 0.4)
		+ (2.80 / 0.85) * (u - 2.2) * well_exp
	)


def mixed_nonconvex_grad_u(u, k):
	"""Derivata analitica del caso 3 rispetto a u (k fissato)."""
	a = 3.3 * u + 0.8 * k
	b = 0.7 * u - 1.5 * k
	term_base = 0.07 * (u - 0.7 * k)
	term_ripple_xk = 0.95 * (3.3 * np.cos(a) * np.cos(b) - 0.7 * np.sin(a) * np.sin(b))
	term_ripple_hi = -2.184 * np.sin(5.2 * u - 0.3 * k)
	return term_base + term_ripple_xk + term_ripple_hi


def gradient_descent_1d(cost_fn, grad_fn, u0, lr, steps):
	"""Discesa del gradiente 1D con clipping per stabilita numerica."""
	us = [float(u0)]
	js = [float(cost_fn(u0))]
	u = float(u0)
	for _ in range(steps):
		g = float(grad_fn(u))
		u = u - lr * g
		u = float(np.clip(u, -8.0, 8.0))
		us.append(u)
		js.append(float(cost_fn(u)))
	return np.array(us), np.array(js)


def find_local_minima_1d(values):
	"""Indici dei minimi locali in una sequenza 1D."""
	idx = []
	for i in range(1, len(values) - 1):
		if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
			idx.append(i)
	return np.array(idx, dtype=int)


def main():
	u = np.linspace(-8.0, 8.0, 2400)

	# Caso 1: convesso
	J_quad = quadratic_cost_u(u)
	u_star_q = -1.0
	J_star_q = quadratic_cost_u(u_star_q)

	# Caso 2: non convesso continuo
	J_non = nonconvex_cost_u(u)
	local_min_non = find_local_minima_1d(J_non)
	gidx_non = int(np.argmin(J_non))
	# Inizializzazione scelta per convergere a un minimo locale specifico.
	gd2_u, gd2_j = gradient_descent_1d(nonconvex_cost_u, nonconvex_grad_u, u0=-6.5, lr=0.035, steps=140)

	# Caso 3: mixed-integer copiato (k = 1, 2)
	k_values = np.array([1, 2], dtype=int)
	J_mixed = np.array([mixed_nonconvex_cost(u, k) for k in k_values])
	global_idx_mixed = np.unravel_index(np.argmin(J_mixed), J_mixed.shape)
	k_g = int(k_values[global_idx_mixed[0]])
	u_g = float(u[global_idx_mixed[1]])
	J_g = float(J_mixed[global_idx_mixed])
	# Discesa del gradiente sul caso 3 con k fissato per ottenere un minimo diverso dal caso 2.
	k_gd3 = 2
	gd3_cost = lambda uu: mixed_nonconvex_cost(uu, k_gd3)
	gd3_grad = lambda uu: mixed_nonconvex_grad_u(uu, k_gd3)
	gd3_u, gd3_j = gradient_descent_1d(gd3_cost, gd3_grad, u0=5.8, lr=0.030, steps=140)

	fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

	# Plot 1
	axes[0].plot(u, J_quad, color="tab:blue", lw=1.8)
	axes[0].scatter([u_star_q], [J_star_q], color="red", s=60, zorder=5)
	axes[0].set_xlabel("u")
	axes[0].set_ylabel("J")
	axes[0].grid(True, alpha=0.25)

	# Plot 2
	axes[1].plot(u, J_non, color="tab:purple", lw=1.5)
	if local_min_non.size > 0:
		axes[1].scatter(u[local_min_non], J_non[local_min_non], s=12, color="black", alpha=0.75)
	axes[1].scatter([u[gidx_non]], [J_non[gidx_non]], color="red", s=60, zorder=6)
	(gd2_traj_line,) = axes[1].plot([], [], color="tab:green", lw=2.0, alpha=0.95, label="GD path")
	(gd2_point,) = axes[1].plot([], [], "o", color="tab:green", ms=6)
	axes[1].set_xlabel("u")
	axes[1].set_ylabel("J")
	axes[1].grid(True, alpha=0.25)
	axes[1].legend(loc="best")

	# Plot 3
	for row, k in enumerate(k_values):
		axes[2].plot(u, J_mixed[row], lw=1.5, label=f"k={k}")
		local_min_idx = find_local_minima_1d(J_mixed[row])
		if local_min_idx.size > 0:
			axes[2].scatter(u[local_min_idx], J_mixed[row, local_min_idx], s=12, color="black", alpha=0.65)
	axes[2].scatter([u_g], [J_g], color="red", s=70, zorder=7, label=f"global min (k={k_g})")
	(gd3_traj_line,) = axes[2].plot([], [], color="tab:green", lw=2.0, alpha=0.95, label=f"GD path (k={k_gd3})")
	(gd3_point,) = axes[2].plot([], [], "o", color="tab:green", ms=6)
	axes[2].set_xlabel("u")
	axes[2].set_ylabel("J")
	axes[2].grid(True, alpha=0.25)
	axes[2].legend(loc="best")

	output_png = "plot_type_function_2_three_cases_2d.png"
	output_gif = "plot_type_function_2_three_cases_2d_gd.gif"
	fig.tight_layout()
	fig.savefig(output_png, dpi=220)

	frames = max(len(gd2_u), len(gd3_u))

	def update(frame):
		idx2 = min(frame, len(gd2_u) - 1)
		idx3 = min(frame, len(gd3_u) - 1)

		gd2_traj_line.set_data(gd2_u[: idx2 + 1], gd2_j[: idx2 + 1])
		gd2_point.set_data([gd2_u[idx2]], [gd2_j[idx2]])

		gd3_traj_line.set_data(gd3_u[: idx3 + 1], gd3_j[: idx3 + 1])
		gd3_point.set_data([gd3_u[idx3]], [gd3_j[idx3]])

		return gd2_traj_line, gd2_point, gd3_traj_line, gd3_point

	anim = FuncAnimation(fig, update, frames=frames, interval=120, blit=False)
	anim.save(output_gif, writer=PillowWriter(fps=8))

	print("PNG salvata:", output_png)
	print("GIF salvata:", output_gif)
	print(f"Caso 1 -> u*={u_star_q:.4f}, J*={J_star_q:.4f}")
	print(f"Caso 2 -> u*~{u[gidx_non]:.4f}, J*~{J_non[gidx_non]:.4f}")
	print(f"Caso 3 -> u*~{u_g:.4f}, k*={k_g}, J*~{J_g:.4f}")
	print(f"GD caso 2 -> u_final={gd2_u[-1]:.4f}, J_final={gd2_j[-1]:.4f}")
	print(f"GD caso 3 (k={k_gd3}) -> u_final={gd3_u[-1]:.4f}, J_final={gd3_j[-1]:.4f}")

	plt.show()


if __name__ == "__main__":
	main()
