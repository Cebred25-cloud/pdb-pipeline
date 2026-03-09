import time
import pandas as pd
import dask.bag as db
from pathlib import Path
from loguru import logger
from fetch import download_pdb
from parser import parse_pdb

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDB_IDS = [
    "1TIM", "1UBQ", "2HHB", "1BNA", "1CRN",
    "1HTM", "3NIR", "1MBO", "1GFL", "2LYZ"
]

# ── Single unit of work ───────────────────────────────────────────────────────
def process_one(pdb_id: str) -> list[dict]:
    """
    Download and parse a single PDB structure.
    Returns a list of residue dicts, or empty list on failure.
    This is the function Dask calls in parallel.
    """
    pdb_path = download_pdb(pdb_id)
    if pdb_path is None:
        return []

    residues = parse_pdb(pdb_path)
    if residues is None:
        return []

    return residues

# ── Serial runner ─────────────────────────────────────────────────────────────
def run_serial(pdb_ids: list[str]) -> tuple[pd.DataFrame, float]:
    all_residues = []
    start = time.perf_counter()

    for pdb_id in pdb_ids:
        all_residues.extend(process_one(pdb_id))

    elapsed = time.perf_counter() - start
    df = pd.DataFrame(all_residues)
    df.to_csv(OUTPUT_DIR / "residues_serial.csv", index=False)

    logger.info(f"Serial   — {len(df)} residues in {elapsed:.2f}s")
    return df, elapsed

# ── Dask runner ───────────────────────────────────────────────────────────────
def run_dask(pdb_ids: list[str], num_workers: int = 4) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()

    # 1. Create a lazy Dask Bag from the list of IDs
    bag = db.from_sequence(pdb_ids, npartitions=num_workers)

    # 2. Map process_one over every ID in parallel — nothing runs yet
    results = bag.map(process_one)

    # 3. Flatten list-of-lists into a flat list, then compute (runs everything)
    all_residues = results.flatten().compute(scheduler="threads",
                                             num_workers=num_workers)

    elapsed = time.perf_counter() - start
    df = pd.DataFrame(all_residues)
    df.to_csv(OUTPUT_DIR / "residues_dask.csv", index=False)

    logger.info(f"Dask     — {len(df)} residues in {elapsed:.2f}s ({num_workers} workers)")
    return df, elapsed

# ── Compare ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Running serial pipeline...")
    df_serial, t_serial = run_serial(PDB_IDS)

    logger.info("Running Dask pipeline...")
    df_dask, t_dask = run_dask(PDB_IDS, num_workers=4)

    speedup = t_serial / t_dask
    logger.success(f"Speedup: {speedup:.2f}x  ({t_serial:.2f}s → {t_dask:.2f}s)")
    logger.info(f"Row counts match: {len(df_serial) == len(df_dask)}")