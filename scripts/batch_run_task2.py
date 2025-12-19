import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SCRIPTS_DIR = REPO_ROOT / "scripts"
RESULTS_DIR = REPO_ROOT / "results"

CONFIG_PATH = DATA_DIR / "config_n_body.json"
RUN_SCRIPT = SCRIPTS_DIR / "run_once.ps1"

BATCH_CONFIGS = [
    {"name": "N100_dt10",    "tend": 10000.0, "dt": 10.0,   "input": "random_N100.txt"},
    {"name": "N100_dt5",     "tend": 10000.0, "dt": 5.0,    "input": "random_N100.txt"},
    {"name": "N100_dt2",     "tend": 10000.0, "dt": 2.0,    "input": "random_N100.txt"},

    {"name": "N500_dt10",    "tend": 10000.0, "dt": 10.0,   "input": "random_N500.txt"},
    {"name": "N500_dt5",     "tend": 10000.0, "dt": 5.0,    "input": "random_N500.txt"},
    {"name": "N500_dt2",     "tend": 10000.0, "dt": 2.0,    "input": "random_N500.txt"},

    {"name": "N2000_dt10",   "tend": 10000.0, "dt": 10.0,   "input": "random_N2000.txt"},
    {"name": "N2000_dt5",    "tend": 10000.0, "dt": 5.0,    "input": "random_N2000.txt"},
]

BATCH_CONFIGS_OMP = [
    {"name": "N2000_dt5_threads1",  "tend": 10000.0, "dt": 5.0, "input": "random_N2000.txt", "threads": 1},
    {"name": "N2000_dt5_threads2",  "tend": 10000.0, "dt": 5.0, "input": "random_N2000.txt", "threads": 2},
    {"name": "N2000_dt5_threads4",  "tend": 10000.0, "dt": 5.0, "input": "random_N2000.txt", "threads": 4},
    {"name": "N2000_dt5_threads8",  "tend": 10000.0, "dt": 5.0, "input": "random_N2000.txt", "threads": 8},
    {"name": "N2000_dt5_threads16", "tend": 10000.0, "dt": 5.0, "input": "random_N2000.txt", "threads": 16},
]

TARGETS = ["cpu", "cuda"]


def read_json_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_powershell_script(script: Path, args: list[str], env: dict | None = None) -> int:
    cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)] + args
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    return proc.returncode


def collect_metrics(target: str) -> dict | None:
    if target == "cpu":
        mpath = RESULTS_DIR / "nbody_cpu" / "metrics_cpu.txt"
    else:
        mpath = RESULTS_DIR / "nbody_cuda" / "metrics_cuda.txt"

    if not mpath.exists():
        print(f"[WARN] Metrics file not found for target={target}: {mpath}")
        return None

    result = {}
    with mpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def main() -> int:
    print("Repo root:", REPO_ROOT)
    print("Using config file:", CONFIG_PATH)
    print("Using run script:", RUN_SCRIPT)

    if not RUN_SCRIPT.exists():
        print(f"ERROR: run_once.ps1 not found at {RUN_SCRIPT}")
        return 1

    original_config = None
    try:
        original_config = read_json_config(CONFIG_PATH)
    except FileNotFoundError:
        print(f"[WARN] Original config not found, will create {CONFIG_PATH} from scratch for each run.")

    summary_csv = RESULTS_DIR / "nbody_batch_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with summary_csv.open("w", encoding="utf-8") as outf:
        header_cols = [
            "batch_name",
            "target",
            "input_file",
            "t_end",
            "dt",
            "eps",
            "particles",
            "threads_or_device",
            "total_steps",
            "total_time_s_or_ms",
            "min_step",
            "avg_step",
            "steps_per_sec",
            "trajectories_path",
        ]
        outf.write(",".join(header_cols) + "\n")

        for cfg in BATCH_CONFIGS:
            batch_name = cfg["name"]
            tend = cfg["tend"]
            dt = cfg["dt"]
            input_rel = cfg["input"]

            print("\n======================================")
            print(f"Batch config: {batch_name}")
            print(f"  tend = {tend}, dt = {dt}, input = {input_rel}")
            print("======================================\n")

            new_config = {
                "tend": tend,
                "dt": dt,
                "input": input_rel,
            }

            if original_config:
                for extra_key in ("n", "seed"):
                    if extra_key in original_config and extra_key not in new_config:
                        new_config[extra_key] = original_config[extra_key]

            write_json_config(CONFIG_PATH, new_config)
            time.sleep(0.1)

            for target in TARGETS:
                print(f"--- Running target: {target} for batch: {batch_name} ---")

                if target == "cpu":
                    args = ["cpu", "-NoPlot", "-NoTrajectories"]
                else:
                    args = ["cuda", "-NoPlot", "-NoTrajectories"]

                env = None
                rc = run_powershell_script(RUN_SCRIPT, args, env=env)
                if rc != 0:
                    print(f"[ERROR] run_once.ps1 failed for target={target}, batch={batch_name}, exit code={rc}")
                    continue

                metrics = collect_metrics(target)
                if metrics is None:
                    continue

                if target == "cpu":
                    particles = metrics.get("Particles", "")
                    eps = metrics.get("eps", "")
                    t_end_str = metrics.get("t_end", "")
                    dt_str = metrics.get("dt", "")
                    threads = metrics.get("OpenMP threads", "")
                    total_steps = metrics.get("Total steps", "")
                    total_time_s = metrics.get("Total compute time (sum of per-step)", "").split()[0]
                    min_step_s = metrics.get("Min step time", "").split()[0]
                    avg_step_s = metrics.get("Avg step time", "").split()[0]
                    steps_per_sec = metrics.get("Steps per second (avg)", "").split()[0]
                    traj = metrics.get("Output trajectories", "")
                    input_file = metrics.get("Input file", input_rel)

                    row = [
                        batch_name,
                        target,
                        input_file,
                        t_end_str,
                        dt_str,
                        eps,
                        particles,
                        threads,
                        total_steps,
                        total_time_s,
                        min_step_s,
                        avg_step_s,
                        steps_per_sec,
                        traj,
                    ]
                else:
                    particles = metrics.get("Particles", "")
                    eps = metrics.get("eps", "")
                    t_end_str = metrics.get("t_end", "")
                    dt_str = metrics.get("dt", "")
                    device_name = metrics.get("CUDA device", "")
                    total_steps = metrics.get("Total steps", "")
                    total_ms = metrics.get("Total kernel time (reset+forces+step)", "").split()[0]
                    min_ms = metrics.get("Min step time", "").split()[0]
                    avg_ms = metrics.get("Avg step time", "").split()[0]
                    steps_per_sec = metrics.get("Steps per second (avg)", "").split()[0]
                    traj = metrics.get("Output trajectories", "")
                    input_file = metrics.get("Input file", input_rel)

                    row = [
                        batch_name,
                        target,
                        input_file,
                        t_end_str,
                        dt_str,
                        eps,
                        particles,
                        device_name,
                        total_steps,
                        total_ms,
                        min_ms,
                        avg_ms,
                        steps_per_sec,
                        traj,
                    ]

                outf.write(",".join(str(x) for x in row) + "\n")
                outf.flush()
                print(f"[OK] Finished target={target} for batch={batch_name}")

        for cfg in BATCH_CONFIGS_OMP:
            batch_name = cfg["name"]
            tend = cfg["tend"]
            dt = cfg["dt"]
            input_rel = cfg["input"]
            threads_req = cfg["threads"]

            print("\n======================================")
            print(f"OMP batch config: {batch_name}")
            print(f"  tend = {tend}, dt = {dt}, input = {input_rel}, threads = {threads_req}")
            print("======================================\n")

            new_config = {
                "tend": tend,
                "dt": dt,
                "input": input_rel,
            }

            if original_config:
                for extra_key in ("n", "seed"):
                    if extra_key in original_config and extra_key not in new_config:
                        new_config[extra_key] = original_config[extra_key]

            write_json_config(CONFIG_PATH, new_config)
            time.sleep(0.1)

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(threads_req)

            args = ["cpu", "-NoPlot", "-NoTrajectories"]
            print(f"--- Running CPU (OMP) target for batch: {batch_name} with OMP_NUM_THREADS={threads_req} ---")
            rc = run_powershell_script(RUN_SCRIPT, args, env=env)
            if rc != 0:
                print(f"[ERROR] run_once.ps1 failed for OMP batch={batch_name}, exit code={rc}")
                continue

            metrics = collect_metrics("cpu")
            if metrics is None:
                continue

            particles = metrics.get("Particles", "")
            eps = metrics.get("eps", "")
            t_end_str = metrics.get("t_end", "")
            dt_str = metrics.get("dt", "")
            threads_actual = metrics.get("OpenMP threads", "")
            total_steps = metrics.get("Total steps", "")
            total_time_s = metrics.get("Total compute time (sum of per-step)", "").split()[0]
            min_step_s = metrics.get("Min step time", "").split()[0]
            avg_step_s = metrics.get("Avg step time", "").split()[0]
            steps_per_sec = metrics.get("Steps per second (avg)", "").split()[0]
            traj = metrics.get("Output trajectories", "")
            input_file = metrics.get("Input file", input_rel)

            row = [
                batch_name,
                "cpu",
                input_file,
                t_end_str,
                dt_str,
                eps,
                particles,
                threads_actual,
                total_steps,
                total_time_s,
                min_step_s,
                avg_step_s,
                steps_per_sec,
                traj,
            ]

            outf.write(",".join(str(x) for x in row) + "\n")
            outf.flush()
            print(f"[OK] Finished OMP batch={batch_name} (requested threads={threads_req}, actual={threads_actual})")

    if original_config is not None:
        write_json_config(CONFIG_PATH, original_config)
        print(f"\nOriginal config restored to {CONFIG_PATH}")

    print(f"\nBatch summary written to: {summary_csv}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())