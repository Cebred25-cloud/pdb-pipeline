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

# intentionally bad IDs to test error handling
BAD_IDS = ["XXXX", "0000", "FAKE"]

# ── Single unit of work ───────────────────────────────────────────────────────
def process_one(pdb_id: str) -> list[dict]:
    """
    Download and parse a single PDB structure.
    Returns a list of residue dicts, or empty list on failure.
    Failures are logged but never crash the pipeline.
    """
    try:
        pdb_path = download_pdb(pdb_id)
        if pdb_path is None:
            return []

        residues = parse_pdb(pdb_path)
        if residues is None:
            return []

        return residues

    except Exception as e:
        # catch-all so one bad structure never kills the whole pipeline
        logger.error(f"{pdb_id} unhandled exception in process_one: {e}")
        return []

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

    bag = db.from_sequence(pdb_ids, npartitions=num_workers)
    results = bag.map(process_one)
    all_residues = results.flatten().compute(scheduler="threads",
                                             num_workers=num_workers)

    elapsed = time.perf_counter() - start
    df = pd.DataFrame(all_residues)
    df.to_csv(OUTPUT_DIR / "residues_dask.csv", index=False)
    logger.info(f"Dask     — {len(df)} residues in {elapsed:.2f}s ({num_workers} workers)")
    return df, elapsed

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # test error handling with bad IDs mixed in
    test_ids = PDB_IDS + BAD_IDS
    logger.info(f"Processing {len(test_ids)} IDs ({len(BAD_IDS)} intentionally bad)...")

    df, elapsed = run_dask(test_ids, num_workers=4)

    good = len(df["pdb_id"].unique()) if len(df) > 0 else 0
    logger.success(f"Completed — {good} structures succeeded, {len(BAD_IDS)} failed gracefully")
    logger.info(f"Total residues: {len(df)}")