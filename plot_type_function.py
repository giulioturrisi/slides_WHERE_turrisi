import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def quadratic_cost(x, u):
	# Superficie quadratica convessa (forma a campana/paraboloide)
	return 0.35 * (x - 2.0) ** 2 + 0.08 * (u + 1.0) ** 2 + 0.6


def quadratic_grad(x, u):
	dj_dx = 0.70 * (x - 2.0)
	dj_du = 0.16 * (u + 1.0)
	return dj_dx, dj_du


def nonconvex_cost(x, u):
	# Due cunette gaussiane con profondita diversa: minima destra migliore.
	base = 0.10 * (x**2 + u**2) + 3.0
	w1 = np.exp(-(((x + 2.5) ** 2) / 0.9 + ((u + 1.8) ** 2) / 0.9))
	w2 = np.exp(-(((x - 2.3) ** 2) / 1.0 + ((u - 1.9) ** 2) / 1.0))
	# Pozzo sinistro poco profondo, pozzo destro molto profondo.
	return base - 1.20 * w1 - 3.40 * w2


def nonconvex_grad(x, u):
	w1 = np.exp(-(((x + 2.5) ** 2) / 0.9 + ((u + 1.8) ** 2) / 0.9))
	w2 = np.exp(-(((x - 2.3) ** 2) / 1.0 + ((u - 1.9) ** 2) / 1.0))
	dj_dx = 0.20 * x + 1.20 * w1 * (2.0 * (x + 2.5) / 0.9) + 3.40 * w2 * (2.0 * (x - 2.3) / 1.0)
	dj_du = 0.20 * u + 1.20 * w1 * (2.0 * (u + 1.8) / 0.9) + 3.40 * w2 * (2.0 * (u - 1.9) / 1.0)
	return dj_dx, dj_du


def gradient_descent_2d(cost_fn, grad_fn, x0, u0, lr, steps):
	xs = [float(x0)]
	us = [float(u0)]
	js = [float(cost_fn(x0, u0))]

	x = float(x0)
	u = float(u0)
	for _ in range(steps):
		dj_dx, dj_du = grad_fn(x, u)
		x -= lr * dj_dx
		u -= lr * dj_du
		xs.append(float(x))
		us.append(float(u))
		js.append(float(cost_fn(x, u)))

	return np.array(xs), np.array(us), np.array(js)


def one_shot_quadratic_path(cost_fn, x0, u0, steps):
	# Per questa quadratica il minimo e noto in forma chiusa: (x*, u*) = (2, -1).
	x_star, u_star = 2.0, -1.0
	xs = [float(x0), x_star]
	us = [float(u0), u_star]
	js = [float(cost_fn(x0, u0)), float(cost_fn(x_star, u_star))]

	# Mantiene la lunghezza della traiettoria compatibile con i frame della GIF.
	for _ in range(max(0, steps - 1)):
		xs.append(x_star)
		us.append(u_star)
		js.append(float(cost_fn(x_star, u_star)))

	return np.array(xs), np.array(us), np.array(js)


def main():
	# Parametri discesa del gradiente
	steps = 100
	x0_q, u0_q, lr_q = -8.0, 8.0, 0.12
	x0_n_bad, u0_n_bad, lr_n_bad = -3.8, -2.8, 0.07
	x0_n_good, u0_n_good, lr_n_good = 3.9, 2.9, 0.07

	# Tracce del GD
	xq, uq, jq = one_shot_quadratic_path(quadratic_cost, x0_q, u0_q, steps)
	xn_bad, un_bad, jn_bad = gradient_descent_2d(
		nonconvex_cost, nonconvex_grad, x0_n_bad, u0_n_bad, lr_n_bad, steps
	)
	xn_good, un_good, jn_good = gradient_descent_2d(
		nonconvex_cost, nonconvex_grad, x0_n_good, u0_n_good, lr_n_good, steps
	)

	# Dominio 3D: quadratica piu ampia sugli assi per evidenziare la forma
	x_quad = np.linspace(-12.0, 12.0, 150)
	u_quad = np.linspace(-12.0, 12.0, 150)
	Xq, Uq = np.meshgrid(x_quad, u_quad)
	Jq = quadratic_cost(Xq, Uq)

	x_non = np.linspace(-4.5, 4.5, 180)
	u_non = np.linspace(-4.5, 4.5, 180)
	Xn, Un = np.meshgrid(x_non, u_non)
	Jn = nonconvex_cost(Xn, Un)

	fig = plt.figure(figsize=(14, 6))
	ax1 = fig.add_subplot(1, 2, 1, projection="3d")
	ax2 = fig.add_subplot(1, 2, 2, projection="3d")

	# Pannello 1: superficie quadratica
	ax1.plot_surface(Xq, Uq, Jq, cmap="Blues", alpha=0.75, linewidth=0, antialiased=True)
	(traj1,) = ax1.plot([], [], [], "o-", color="tab:red", ms=3, lw=1.2)
	(point1,) = ax1.plot([], [], [], "o", color="black", ms=7)
	ax1.set_xlabel("x")
	ax1.set_ylabel("u")
	ax1.set_zlabel("J")
	ax1.view_init(elev=8, azim=-65)

	# Pannello 2: superficie non convessa
	ax2.plot_surface(Xn, Un, Jn, cmap="Greens", alpha=0.75, linewidth=0, antialiased=True)
	(traj2_bad,) = ax2.plot(
		[], [], [], "o-", color="tab:orange", ms=3, lw=1.2
	)
	(point2_bad,) = ax2.plot([], [], [], "o", color="black", ms=7)
	(traj2_good,) = ax2.plot(
		[], [], [], "o-", color="tab:cyan", ms=3, lw=1.2
	)
	(point2_good,) = ax2.plot([], [], [], "o", color="navy", ms=7)
	ax2.set_xlabel("x")
	ax2.set_ylabel("u")
	ax2.set_zlabel("J")
	ax2.view_init(elev=8, azim=-65)

	# Limiti assi espliciti per mantenere prospettiva stabile
	ax1.set_xlim(x_quad.min(), x_quad.max())
	ax1.set_ylim(u_quad.min(), u_quad.max())
	ax1.set_zlim(Jq.min() - 0.5, Jq.max() + 0.5)
	ax2.set_xlim(x_non.min(), x_non.max())
	ax2.set_ylim(u_non.min(), u_non.max())
	ax2.set_zlim(Jn.min() - 0.5, Jn.max() + 0.5)

	frames = max(len(xq), len(xn_bad), len(xn_good))

	def update(frame):
		idx_q = min(frame, len(xq) - 1)
		idx_n_bad = min(frame, len(xn_bad) - 1)
		idx_n_good = min(frame, len(xn_good) - 1)

		traj1.set_data(xq[: idx_q + 1], uq[: idx_q + 1])
		traj1.set_3d_properties(jq[: idx_q + 1])
		point1.set_data([xq[idx_q]], [uq[idx_q]])
		point1.set_3d_properties([jq[idx_q]])

		traj2_bad.set_data(xn_bad[: idx_n_bad + 1], un_bad[: idx_n_bad + 1])
		traj2_bad.set_3d_properties(jn_bad[: idx_n_bad + 1])
		point2_bad.set_data([xn_bad[idx_n_bad]], [un_bad[idx_n_bad]])
		point2_bad.set_3d_properties([jn_bad[idx_n_bad]])

		traj2_good.set_data(xn_good[: idx_n_good + 1], un_good[: idx_n_good + 1])
		traj2_good.set_3d_properties(jn_good[: idx_n_good + 1])
		point2_good.set_data([xn_good[idx_n_good]], [un_good[idx_n_good]])
		point2_good.set_3d_properties([jn_good[idx_n_good]])

		return traj1, point1, traj2_bad, point2_bad, traj2_good, point2_good

	anim = FuncAnimation(fig, update, frames=frames, interval=120, blit=False)

	output_png = "gradient_descent_costs_3d.png"
	output_gif = "gradient_descent_costs_3d.gif"
	fig.savefig(output_png, dpi=200, bbox_inches="tight")
	anim.save(output_gif, writer=PillowWriter(fps=10))

	print("PNG salvata:", output_png)
	print("GIF salvata:", output_gif)
	print(f"Quadratica -> (x*, u*) ~ ({xq[-1]:.6f}, {uq[-1]:.6f}), J* ~ {jq[-1]:.6f}")
	print(
		f"Non convessa (init sbagliata) -> (x*, u*) ~ ({xn_bad[-1]:.6f}, {un_bad[-1]:.6f}), J* ~ {jn_bad[-1]:.6f}"
	)
	print(
		f"Non convessa (init corretta) -> (x*, u*) ~ ({xn_good[-1]:.6f}, {un_good[-1]:.6f}), J* ~ {jn_good[-1]:.6f}"
	)

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
