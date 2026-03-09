from pathlib import Path
from Bio.PDB import PDBParser
from loguru import logger

# ── Parser ────────────────────────────────────────────────────────────────────
def parse_pdb(pdb_path: Path) -> list[dict] | None:
    """
    Parse a .pdb file and extract per-residue features.
    Returns a list of dicts (one per residue) or None on failure.
    """
    parser = PDBParser(QUIET=True)
    pdb_id = pdb_path.stem

    try:
        structure = parser.get_structure(pdb_id, str(pdb_path))
    except Exception as e:
        logger.warning(f"Could not parse {pdb_id}: {e}")
        return None

    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # skip water molecules and other heteroatoms
                if residue.get_id()[0] != " ":
                    continue

                # get CA atom coords if available
                ca_x, ca_y, ca_z, b_factor = None, None, None, None
                if "CA" in residue:
                    ca = residue["CA"]
                    ca_x, ca_y, ca_z = ca.get_vector().get_array()
                    b_factor = ca.get_bfactor()

                residues.append({
                    "pdb_id":    pdb_id,
                    "chain":     chain.get_id(),
                    "res_name":  residue.get_resname(),
                    "res_seq":   residue.get_id()[1],
                    "ca_x":      ca_x,
                    "ca_y":      ca_y,
                    "ca_z":      ca_z,
                    "b_factor":  b_factor,
                })

    if not residues:
        logger.warning(f"{pdb_id} yielded no residues")
        return None

    logger.success(f"Parsed {pdb_id} — {len(residues)} residues")
    return residues