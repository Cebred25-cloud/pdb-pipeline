import requests
import time
from pathlib import Path
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# ── Downloader ────────────────────────────────────────────────────────────────
def download_pdb(pdb_id: str) -> Path | None:
    """Download a single .pdb file from RCSB. Returns path or None on failure."""
    pdb_id = pdb_id.upper().strip()
    dest = RAW_DIR / f"{pdb_id}.pdb"

    if dest.exists():
        logger.info(f"{pdb_id} already cached, skipping download")
        return dest

    url = RCSB_URL.format(pdb_id=pdb_id)
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        dest.write_text(response.text)
        logger.success(f"Downloaded {pdb_id}")
        return dest
    except Exception as e:
        logger.warning(f"Failed to download {pdb_id}: {e}")
        return None