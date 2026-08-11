import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def mixed_nonconvex_cost(u, k):
	"""
	Funzione mixed-integer non convessa:
	- u continuo
	- k intero
	Con molte ondulazioni per creare tanti minimi locali.
	"""
	base = 0.035 * (u - 0.7 * k) ** 2 + 0.12 * (k - 1.0) ** 2
	ripple_xk = 0.95 * np.sin(3.3 * u + 0.8 * k) * np.cos(0.7 * u - 1.5 * k)
	ripple_hi = 0.42 * np.cos(5.2 * u - 0.3 * k)
	ripple_k = 0.28 * np.cos(2.2 * k)
	return base + ripple_xk + ripple_hi + ripple_k


def find_local_minima_1d(values):
	"""Indici dei minimi locali in una sequenza 1D."""
	idx = []
	for i in range(1, len(values) - 1):
		if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
			idx.append(i)
	return np.array(idx, dtype=int)


def main():
	# Dominio mixed-integer
	u = np.linspace(-8.0, 8.0, 2400)
	k_values = np.array([1, 2], dtype=int)

	# Griglia valori: righe = k interi, colonne = u continui
	F = np.array([mixed_nonconvex_cost(u, k) for k in k_values])

	# Minimo globale su tutta la griglia campionata
	global_idx = np.unravel_index(np.argmin(F), F.shape)
	k_g = k_values[global_idx[0]]
	u_g = u[global_idx[1]]
	j_g = F[global_idx]

	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(1, 1, 1, projection="3d")

	# Unica vista 3D: curve J(u, k) per k discreti
	for row, k in enumerate(k_values):
		k_line = np.full_like(u, float(k))
		ax.plot(u, k_line, F[row], lw=1.4, alpha=0.9, label=f"k={k}")

		local_min_idx = find_local_minima_1d(F[row])
		if local_min_idx.size > 0:
			ax.scatter(
				u[local_min_idx],
				np.full(local_min_idx.size, float(k)),
				F[row, local_min_idx],
				s=10,
				alpha=0.65,
				color="black",
			)

	ax.scatter([u_g], [k_g], [j_g], s=120, marker="*", color="red", edgecolor="black", linewidth=0.8, label="global min")

	# Punti che verranno campionati e mostrati progressivamente nella GIF.
	rng = np.random.default_rng(7)
	n_samples = 180
	u_samples = rng.uniform(u.min(), u.max(), size=n_samples)
	k_samples = rng.choice(k_values, size=n_samples)
	j_samples = mixed_nonconvex_cost(u_samples, k_samples)

	(samples_plot,) = ax.plot([], [], [], "o", color="tab:green", ms=5, alpha=0.95, label="sampled points")

	ax.set_xlabel("u")
	ax.set_ylabel("k")
	ax.set_zlabel("J(u, k)")
	ax.set_yticks(k_values)
	ax.view_init(elev=24, azim=-58)
	ax.legend(loc="upper left")

	output_png = "nonconvex_mixed_integer_many_minima.png"
	output_gif = "nonconvex_mixed_integer_many_minima_sampling.gif"
	fig.tight_layout()
	fig.savefig(output_png, dpi=220)

	def update(frame):
		idx = min(frame // 4 + 1, n_samples)
		samples_plot.set_data(u_samples[:idx], k_samples[:idx])
		samples_plot.set_3d_properties(j_samples[:idx])
		return (samples_plot,)

	# Quattro frame per ogni punto: durata raddoppiata rispetto alla versione precedente.
	anim = FuncAnimation(fig, update, frames=4 * n_samples, interval=1000 / 15, blit=False)
	anim.save(output_gif, writer=PillowWriter(fps=15))

	print("Grafico salvato:", output_png)
	print("GIF salvata:", output_gif)
	print(f"Minimo globale (campionato): u={u_g:.4f}, k={k_g}, J={j_g:.4f}")

	plt.show()


if __name__ == "__main__":
	main()

