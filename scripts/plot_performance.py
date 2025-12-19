import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "nbody_batch_summary.csv"
NBODY_RESULTS_DIR = RESULTS_DIR / "nbody"


def load_summary(csv_path: Path):
    runs = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def to_float(key):
                try:
                    return float(row[key])
                except Exception:
                    return None

            def to_int(key):
                try:
                    return int(row[key])
                except Exception:
                    return None

            row["t_end"] = to_float("t_end")
            row["dt"] = to_float("dt")
            row["eps"] = to_float("eps")
            row["particles"] = to_int("particles")
            row["total_steps"] = to_int("total_steps")
            row["steps_per_sec"] = to_float("steps_per_sec")

            runs.append(row)
    return runs


def _batch_sort_key(name: str):
    n_val = None
    dt_val = None

    parts = name.split("_")
    for p in parts:
        if p.startswith("N"):
            try:
                n_val = int(p[1:])
            except Exception:
                pass
        elif p.startswith("dt"):
            try:
                dt_val = float(p[2:])
            except Exception:
                pass

    n_sort = n_val if n_val is not None else 10**9
    dt_sort = dt_val if dt_val is not None else 10**9
    return (n_sort, dt_sort, name)


def plot_steps_per_sec_bar(runs, out_path: Path):
    base_runs = [r for r in runs if "threads" not in r["batch_name"]]

    if not base_runs:
        print("No base (non-OMP) runs for bar plot, skipping.")
        return

    by_batch = defaultdict(list)
    for r in base_runs:
        by_batch[r["batch_name"]].append(r)

    batch_names = sorted(by_batch.keys(), key=_batch_sort_key)

    cpu_vals = []
    cuda_vals = []
    for name in batch_names:
        cpu = next((r for r in by_batch[name] if r["target"] == "cpu"), None)
        cuda = next((r for r in by_batch[name] if r["target"] == "cuda"), None)
        cpu_vals.append(cpu["steps_per_sec"] if cpu and cpu["steps_per_sec"] is not None else 0.0)
        cuda_vals.append(cuda["steps_per_sec"] if cuda and cuda["steps_per_sec"] is not None else 0.0)

    x = range(len(batch_names))
    width = 0.35

    plt.figure(figsize=(max(8, len(batch_names) * 0.8), 6))
    plt.bar([i - width / 2 for i in x], cpu_vals, width=width, label="CPU")
    plt.bar([i + width / 2 for i in x], cuda_vals, width=width, label="CUDA")

    plt.xticks(list(x), batch_names, rotation=45, ha="right")
    plt.ylabel("Steps per second")
    plt.title("N-body performance: CPU vs CUDA (steps_per_sec)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved bar plot (without OMP 'threads' series): {out_path}")
    plt.close()


def plot_steps_vs_dt_for_particles(runs, particles: int, out_path: Path):
    subset = [r for r in runs if r["particles"] == particles and r["dt"] is not None]

    if not subset:
        print(f"No runs found for particles={particles}, skipping {out_path.name}")
        return

    cpu = sorted(
        (r for r in subset if r["target"] == "cpu" and r["steps_per_sec"] is not None and "threads" not in r["batch_name"]),
        key=lambda r: r["dt"],
    )
    cuda = sorted(
        (r for r in subset if r["target"] == "cuda" and r["steps_per_sec"] is not None),
        key=lambda r: r["dt"],
    )

    if not cpu and not cuda:
        print(f"No valid data (CPU/CUDA) for particles={particles}, skipping {out_path.name}")
        return

    plt.figure(figsize=(8, 5))

    if cpu:
        plt.plot(
            [r["dt"] for r in cpu],
            [r["steps_per_sec"] for r in cpu],
            marker="o",
            label=f"CPU (N={particles})",
        )

    if cuda:
        plt.plot(
            [r["dt"] for r in cuda],
            [r["steps_per_sec"] for r in cuda],
            marker="s",
            label=f"CUDA (N={particles})",
        )

    plt.xlabel("dt")
    plt.ylabel("Steps per second")
    plt.title(f"Steps per second vs dt (N={particles})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved dt plot for N={particles}: {out_path}")
    plt.close()


def parse_threads_from_batch_name(name: str):
    n_val = None
    dt_val = None
    threads_val = None

    parts = name.split("_")
    for p in parts:
        if p.startswith("N"):
            try:
                n_val = int(p[1:])
            except Exception:
                pass
        elif p.startswith("dt"):
            try:
                dt_val = float(p[2:])
            except Exception:
                pass
        elif p.startswith("threads"):
            try:
                threads_val = int(p[len("threads"):])
            except Exception:
                pass

    return n_val, dt_val, threads_val


def plot_omp_series(runs):
    omp_runs = [r for r in runs if r["target"] == "cpu" and "threads" in r["batch_name"]]

    if not omp_runs:
        print("No OMP series found (no 'threads' in batch_name), skipping OMP plots.")
        return

    groups = defaultdict(list)
    for r in omp_runs:
        n_val, dt_val, threads_val = parse_threads_from_batch_name(r["batch_name"])
        if n_val is None or dt_val is None or threads_val is None:
            continue
        r["_threads"] = threads_val
        key = (n_val, dt_val)
        groups[key].append(r)

    if not groups:
        print("No valid OMP groups parsed, skipping OMP plots.")
        return

    for (n_val, dt_val), gr in groups.items():
        gr_sorted = sorted(gr, key=lambda r: r["_threads"])

        threads = [r["_threads"] for r in gr_sorted]
        steps_per_sec = [r["steps_per_sec"] for r in gr_sorted if r["steps_per_sec"] is not None]

        if not steps_per_sec:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(threads, steps_per_sec, marker="o")
        plt.xlabel("OpenMP threads")
        plt.ylabel("Steps per second")
        plt.title(f"CPU steps_per_sec vs threads (N={n_val}, dt={dt_val})")
        plt.grid(True, linestyle="--", alpha=0.4)
        out_path_steps = NBODY_RESULTS_DIR / f"omp_steps_vs_threads_N{n_val}_dt{dt_val}.png"
        plt.tight_layout()
        plt.savefig(out_path_steps, dpi=150)
        print(f"Saved OMP steps_vs_threads plot: {out_path_steps}")
        plt.close()

        base = None
        min_threads = min(threads)
        for r in gr_sorted:
            if r["_threads"] == min_threads and r["steps_per_sec"] is not None:
                base = r["steps_per_sec"]
                break
        if base is None or base <= 0.0:
            continue

        speedups = [r["steps_per_sec"] / base if r["steps_per_sec"] is not None else 0.0
                    for r in gr_sorted]

        plt.figure(figsize=(8, 5))
        plt.plot(threads, speedups, marker="o")
        plt.xlabel("OpenMP threads")
        plt.ylabel(f"Speedup (relative to threads = {min_threads})")
        plt.title(f"CPU speedup vs threads (N={n_val}, dt={dt_val})")
        plt.grid(True, linestyle="--", alpha=0.4)
        out_path_speedup = NBODY_RESULTS_DIR / f"omp_speedup_vs_threads_N{n_val}_dt{dt_val}.png"
        plt.tight_layout()
        plt.savefig(out_path_speedup, dpi=150)
        print(f"Saved OMP speedup_vs_threads plot: {out_path_speedup}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot N-body performance from nbody_batch_summary.csv")
    parser.add_argument(
        "--summary",
        type=str,
        default=str(SUMMARY_CSV),
        help="Path to nbody_batch_summary.csv (default: results/nbody_batch_summary.csv)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show plots interactively, only save to files",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"ERROR: summary CSV not found: {summary_path}")
        return 1

    runs = load_summary(summary_path)
    if not runs:
        print(f"ERROR: no runs found in {summary_path}")
        return 1

    NBODY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bar_path = NBODY_RESULTS_DIR / "cpu_vs_cuda_steps_per_sec.png"
    plot_steps_per_sec_bar(runs, bar_path)

    particles_values = sorted({r["particles"] for r in runs if r["particles"] is not None})
    for p in particles_values:
        out_path = NBODY_RESULTS_DIR / f"steps_vs_dt_N{p}.png"
        plot_steps_vs_dt_for_particles(runs, particles=p, out_path=out_path)

    plot_omp_series(runs)

    if not args.no_show:
        print("Plots saved to:", NBODY_RESULTS_DIR)
        print("Open PNG files from this folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())