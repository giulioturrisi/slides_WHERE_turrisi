"""Benchmark Sampling_MPC.compute_control with zero-order control profiles.

Examples:
    python3 test_ms_predictive_sampling.py
    python3 test_ms_predictive_sampling.py --cpu-cores 1
    python3 test_ms_predictive_sampling.py --repetitions 200 --cpu-cores 0

``--cpu-cores 0`` leaves CPU parallelism under JAX/XLA control. A positive
value limits the common CPU thread pools before JAX is imported.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-cores",
        type=int,
        choices=(0, 1),
        default=1,
        help="0 uses JAX parallelism; 1 confines the process to one CPU core.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=100,
        help="Timed compute_control calls for each K (default: 100).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Untimed warm-up calls for each K (default: 5).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=80,
        help="Prediction horizon; zero-order samples horizon v and w values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for CSV and PNG results.",
    )
    return parser.parse_args()


def configure_cpu(cpu_cores):
    """Configure thread limits before importing JAX."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    if cpu_cores == 0:
        return

    if hasattr(os, "sched_getaffinity"):
        available_cpus = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(available_cpus)})

    core_count = str(cpu_cores)
    os.environ["OMP_NUM_THREADS"] = core_count
    os.environ["MKL_NUM_THREADS"] = core_count
    os.environ["OPENBLAS_NUM_THREADS"] = core_count
    os.environ["NUMEXPR_NUM_THREADS"] = core_count
    xla_flags = os.environ.get("XLA_FLAGS", "")
    thread_flags = (
        f"--xla_cpu_multi_thread_eigen=false "
        f"intra_op_parallelism_threads={cpu_cores}"
    )
    os.environ["XLA_FLAGS"] = f"{xla_flags} {thread_flags}".strip()


def write_raw_samples(path, samples_by_k):
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["num_computations", "repetition", "milliseconds"])
        for num_computations, samples in samples_by_k.items():
            for repetition, milliseconds in enumerate(samples, start=1):
                writer.writerow([num_computations, repetition, milliseconds])


def write_statistics(path, statistics):
    with path.open("w", newline="") as output_file:
        fieldnames = [
            "num_computations",
            "repetitions",
            "mean_ms",
            "variance_ms2",
            "std_ms",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(statistics)


def save_plot(path, samples_by_k, cpu_cores):
    import matplotlib.pyplot as plt

    k_values = list(samples_by_k)
    samples = [samples_by_k[k] for k in k_values]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.boxplot(
        samples,
        tick_labels=[str(k) for k in k_values],
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "tab:orange",
            "markeredgecolor": "black",
        },
        medianprops={"color": "tab:red", "linewidth": 2},
        boxprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )
    axis.set_xlabel("K — number of sampled rollouts")
    axis.set_ylabel("compute_control [ms]")
    axis.set_title(
        "Zero-order predictive sampling — "
        + ("JAX parallelism automatic" if cpu_cores == 0 else f"{cpu_cores} CPU core(s)")
    )
    axis.grid(True, which="both", axis="y", alpha=0.35)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    arguments = parse_arguments()
    configure_cpu(arguments.cpu_cores)

    # predictive_sampling imports robot_model as a sibling module.
    sampling_directory = Path(__file__).resolve().parent
    sys.path.insert(0, str(sampling_directory))

    import jax
    import jax.numpy as jnp
    import numpy as np
    from predictive_sampling import Sampling_MPC

    k_values = (10, 500, 2000, 10000)
    state = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    goal = jnp.array([4.0, 2.0, 0.0], dtype=jnp.float32)
    obstacles = np.array(
        [[1.20, 0.35, 0.28], [2.15, 1.05, 0.35], [3.05, 1.35, 0.30]],
        dtype=np.float32,
    )

    samples_by_k = {}
    statistics = []
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print("CPU cores setting:", "automatic" if arguments.cpu_cores == 0 else arguments.cpu_cores)

    for num_computations in k_values:
        controller = Sampling_MPC(
            horizon=arguments.horizon,
            num_computations=num_computations,
            interpolation="zero_order",
            obstacles=obstacles,
            sample_deltas=False,
            print_computation_time=False,
        )

        for _ in range(arguments.warmup):
            controller.compute_control(state, goal)

        samples = np.empty(arguments.repetitions, dtype=np.float64)
        for repetition in range(arguments.repetitions):
            start_ns = time.perf_counter_ns()
            controller.compute_control(state, goal)
            samples[repetition] = (time.perf_counter_ns() - start_ns) / 1e6

        samples_by_k[num_computations] = samples
        row = {
            "num_computations": num_computations,
            "repetitions": arguments.repetitions,
            "mean_ms": float(np.mean(samples)),
            "variance_ms2": float(np.var(samples, ddof=0)),
            "std_ms": float(np.std(samples, ddof=0)),
        }
        statistics.append(row)
        print(
            f"K={num_computations:5d}: mean={row['mean_ms']:.4f} ms, "
            f"variance={row['variance_ms2']:.6f} ms²"
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "auto" if arguments.cpu_cores == 0 else f"{arguments.cpu_cores}_core"
    statistics_path = arguments.output_dir / f"predictive_sampling_stats_{suffix}.csv"
    samples_path = arguments.output_dir / f"predictive_sampling_samples_{suffix}.csv"
    plot_path = arguments.output_dir / f"predictive_sampling_benchmark_{suffix}.png"
    write_statistics(statistics_path, statistics)
    write_raw_samples(samples_path, samples_by_k)
    save_plot(plot_path, samples_by_k, arguments.cpu_cores)
    print("Statistics:", statistics_path)
    print("Raw samples:", samples_path)
    print("Plot:", plot_path)


if __name__ == "__main__":
    main()
