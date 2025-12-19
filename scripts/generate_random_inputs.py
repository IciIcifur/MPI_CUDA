import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

DATASETS = [
    {"filename": "random_N100.txt", "n": 100, "seed": 123},
    {"filename": "random_N500.txt", "n": 500, "seed": 456},
    {"filename": "random_N2000.txt", "n": 2000, "seed": 789},
]

def generate_dataset(path: Path, n: int, seed: int) -> None:
    random.seed(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    mass_min, mass_max = 1.0e22, 1.0e26     # kg
    pos_min, pos_max = -1.0e11, 1.0e11      # m
    vel_min, vel_max = -3.0e4, 3.0e4        # m/s

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for _ in range(n):
            m = random.uniform(mass_min, mass_max)
            x = random.uniform(pos_min, pos_max)
            y = random.uniform(pos_min, pos_max)
            vx = random.uniform(vel_min, vel_max)
            vy = random.uniform(vel_min, vel_max)
            f.write(f"{m:.6e} {x:.6e} {y:.6e} {vx:.6e} {vy:.6e}\n")

    print(f"Generated {n} particles in {path}")


def main() -> int:
    print("Repo root:", REPO_ROOT)
    print("Data dir:", DATA_DIR)

    for ds in DATASETS:
        filename = ds["filename"]
        n = ds["n"]
        seed = ds["seed"]
        out_path = DATA_DIR / filename
        generate_dataset(out_path, n, seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())