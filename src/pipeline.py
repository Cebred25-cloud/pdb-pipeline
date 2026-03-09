import time
import pandas as pd
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

# ── Main ──────────────────────────────────────────────────────────────────────
def run_serial(pdb_ids: list[str]) -> pd.DataFrame:
    all_residues = []
    failed = []

    start = time.perf_counter()

    for pdb_id in pdb_ids:
        pdb_path = download_pdb(pdb_id)
        if pdb_path is None:
            failed.append(pdb_id)
            continue

        residues = parse_pdb(pdb_path)
        if residues is None:
            failed.append(pdb_id)
            continue

        all_residues.extend(residues)

    elapsed = time.perf_counter() - start

    df = pd.DataFrame(all_residues)
    out_path = OUTPUT_DIR / "residues_serial.csv"
    df.to_csv(out_path, index=False)

    logger.info(f"Done — {len(pdb_ids) - len(failed)}/{len(pdb_ids)} structures processed")
    logger.info(f"Total residues: {len(df)}")
    logger.info(f"Serial runtime: {elapsed:.2f}s")
    logger.info(f"Saved to {out_path}")

    if failed:
        logger.warning(f"Failed IDs: {failed}")

    return df

if __name__ == "__main__":
    df = run_serial(PDB_IDS)
    print(df.head(10))