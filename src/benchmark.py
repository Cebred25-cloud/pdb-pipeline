import time
import random
import pandas as pd
import dask.bag as db
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from fetch import download_pdb
from parser import parse_pdb

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIZES      = [100, 250, 500]
NUM_WORKERS = 4

# ── Load IDs ──────────────────────────────────────────────────────────────────
def load_ids(path: str = "pdb_ids.txt") -> list[str]:
    ids = Path(path).read_text().strip().splitlines()
    random.seed(42)
    random.shuffle(ids)
    return ids

# ── Single unit of work ───────────────────────────────────────────────────────
def process_one(pdb_id: str) -> list[dict]:
    try:
        pdb_path = download_pdb(pdb_id)
        if pdb_path is None:
            return []
        residues = parse_pdb(pdb_path)
        return residues if residues else []
    except Exception as e:
        logger.error(f"{pdb_id} failed: {e}")
        return []

# ── Runners ───────────────────────────────────────────────────────────────────
def run_serial(pdb_ids: list[str]) -> float:
    start = time.perf_counter()
    for pdb_id in pdb_ids:
        process_one(pdb_id)
    return time.perf_counter() - start

def run_dask(pdb_ids: list[str], num_workers: int = NUM_WORKERS) -> float:
    start = time.perf_counter()
    bag = db.from_sequence(pdb_ids, npartitions=num_workers)
    bag.map(process_one).flatten().compute(
        scheduler="threads", num_workers=num_workers
    )
    return time.perf_counter() - start

# ── Benchmark ─────────────────────────────────────────────────────────────────
def run_benchmark(all_ids: list[str]) -> pd.DataFrame:
    results = []

    for n in SIZES:
        ids = all_ids[:n]
        logger.info(f"--- Benchmarking n={n} ---")

        logger.info(f"  Serial  n={n}...")
        t_serial = run_serial(ids)
        logger.success(f"  Serial  n={n}: {t_serial:.2f}s")

        logger.info(f"  Dask    n={n} ({NUM_WORKERS} workers)...")
        t_dask = run_dask(ids)
        logger.success(f"  Dask    n={n}: {t_dask:.2f}s")

        speedup = t_serial / t_dask
        logger.success(f"  Speedup n={n}: {speedup:.2f}x")

        results.append({
            "n":        n,
            "serial":   round(t_serial, 3),
            "dask":     round(t_dask, 3),
            "speedup":  round(speedup, 2),
        })

    return pd.DataFrame(results)

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_results(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0F1117")

    for ax in axes:
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#8B8FA8")
        ax.xaxis.label.set_color("#C4C6D4")
        ax.yaxis.label.set_color("#C4C6D4")
        ax.title.set_color("#E8E9F0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#23263A")

    # ── left: runtime comparison
    ax1 = axes[0]
    ax1.plot(df["n"], df["serial"], "o-", color="#7C6AF7",
             linewidth=2, markersize=7, label="Serial")
    ax1.plot(df["n"], df["dask"],   "o-", color="#3ECFCF",
             linewidth=2, markersize=7, label=f"Dask ({NUM_WORKERS} workers)")
    ax1.set_xlabel("Number of PDB structures")
    ax1.set_ylabel("Wall time (seconds)")
    ax1.set_title("Runtime: Serial vs Dask")
    ax1.legend(facecolor="#1A1D27", labelcolor="#C4C6D4", edgecolor="#23263A")
    ax1.set_xticks(df["n"])
    ax1.grid(True, color="#23263A", linewidth=0.7)

    # ── right: speedup curve
    ax2 = axes[1]
    ax2.plot(df["n"], df["speedup"], "o-", color="#F7846A",
             linewidth=2, markersize=7, label="Actual speedup")
    ax2.axhline(y=NUM_WORKERS, color="#8B8FA8", linestyle="--",
                linewidth=1.2, label=f"Ideal ({NUM_WORKERS}x)")
    ax2.set_xlabel("Number of PDB structures")
    ax2.set_ylabel("Speedup (T_serial / T_dask)")
    ax2.set_title("Speedup vs Dataset Size")
    ax2.legend(facecolor="#1A1D27", labelcolor="#C4C6D4", edgecolor="#23263A")
    ax2.set_xticks(df["n"])
    ax2.grid(True, color="#23263A", linewidth=0.7)

    plt.tight_layout()
    out = OUTPUT_DIR / "benchmark_speedup.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.success(f"Plot saved to {out}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_ids = load_ids("pdb_ids.txt")
    logger.info(f"Loaded {len(all_ids)} PDB IDs")

    df = run_benchmark(all_ids)

    print("\n── Benchmark Results ──────────────────")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)
    plot_results(df)
