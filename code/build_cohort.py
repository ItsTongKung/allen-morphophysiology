"""Build the analysis cohorts from locally supplied Allen Cell Types Database metadata.

Reads the metadata tables described in ``docs/DATA.md`` (electrophysiology
features, neuron reconstructions, specimens with donor joins, and the Age
lookup), assigns species / broad class / cortical layer, applies the
pre-specified electrophysiology quality-control rule, and assembles the human
donor-level table from public release fields only.

Outputs (under ``<results-dir>/cohort/``):
    ephys_cells.csv       electrophysiology table with cohort annotations
    reconstructions.csv   reconstruction table with cohort annotations
    human_donors.csv      donor-level metadata (public release fields only)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Primary electrophysiological features (released per-cell fields) used across
# the study.
EPHYS_FEATURES = [
    "input_resistance_mohm", "tau", "ri", "sag", "adaptation", "avg_isi",
    "f_i_curve_slope", "latency", "upstroke_downstroke_ratio_long_square",
    "threshold_v_long_square", "threshold_i_long_square", "vrest",
    "fast_trough_v_long_square", "peak_v_long_square", "vm_for_sag",
]

# Cre-line keys detected in specimen names, in matching-priority order.  The
# first matching key determines the broad class of mouse cells.
CRE_KEYS = ["Pvalb", "Sst", "Vip", "Htr3a", "Ndnf", "Cux2", "Emx1", "Nr5a1",
            "Rorb", "Scnn1a", "Rbp4", "Tlx3", "Tle4", "Slc17a7", "Ntsri1",
            "Glt25d2"]

# Cortical-area grouping of the human structures present in the release
# (mouse cells are all primary visual cortex).
HUMAN_AREA_GROUPS = {
    "middle temporal gyrus": "MTG",
    "middle frontal gyrus": "frontal-other",
    "frontal lobe": "frontal-other",
    "superior frontal gyrus": "frontal-other",
    "inferior frontal gyrus": "frontal-other",
    "inferior temporal gyrus": "temporal-other",
    "temporal lobe": "temporal-other",
    "planum polare": "temporal-other",
    "angular gyrus": "angular/parietal",
}


def assign_species(organism_id, specimen_name):
    """Species from the donor organism id (1 = human, 2 = mouse) with a
    specimen-name fallback (names beginning 'H<digit>' are human)."""
    if organism_id == 1:
        return "human"
    if organism_id == 2:
        return "mouse"
    return "human" if re.match(r"^H\d", str(specimen_name)) else "mouse"


def cre_line_of(specimen_name):
    name = str(specimen_name)
    for key in CRE_KEYS:
        if key in name:
            return key + "-Cre"
    return None


def broad_class_of(species, specimen_name):
    """Broad classes: Pvalb, Sst, Vip-like (Vip/Htr3a/Ndnf), excitatory
    (remaining cre lines); human cells form a single stratum."""
    if species == "human":
        return "human"
    cre = cre_line_of(specimen_name)
    if cre is None:
        return "wild-type/other"
    if cre.startswith("Pvalb"):
        return "Pvalb"
    if cre.startswith("Sst"):
        return "Sst"
    if cre.startswith(("Vip", "Htr3a", "Ndnf")):
        return "Vip-like"
    return "excitatory"


def layer_of(structure_acronym):
    """Cortical layer parsed from mouse visual-cortex structure acronyms
    (VISp1-VISp6a/b); human structures carry no layer assignment."""
    m = re.match(r"VISp(\d)([ab]?)", str(structure_acronym))
    return int(m.group(1)) if m else None


def area_group_of(species, structure_name):
    if species == "mouse":
        return "V1"
    return HUMAN_AREA_GROUPS.get(str(structure_name), "other")


def qc_mask(ephys):
    """Pre-specified electrophysiology QC rule: finite primary features with
    input resistance > 0, membrane time constant > 0, resting potential
    between -90 and -40 mV, and a non-negative seal."""
    ok_rin = pd.to_numeric(ephys["input_resistance_mohm"], errors="coerce") > 0
    ok_tau = pd.to_numeric(ephys["tau"], errors="coerce") > 0
    ok_vrest = pd.to_numeric(ephys["vrest"], errors="coerce").between(-90, -40)
    seal = pd.to_numeric(ephys["seal_gohm"], errors="coerce").fillna(-1)
    return ok_rin & ok_tau & ok_vrest & (seal >= 0)


def annotate(df):
    """Add species, cre line, broad class, layer, and area group."""
    df = df.copy()
    df["species"] = [assign_species(o, n) for o, n in
                     zip(df["donor_organism_id"], df["specimen_name"])]
    df["cre_line"] = df["specimen_name"].map(cre_line_of)
    df["broad_class"] = [broad_class_of(sp, nm) for sp, nm in
                         zip(df["species"], df["specimen_name"])]
    df["layer"] = df["structure_acronym"].map(layer_of)
    df["area_group"] = [area_group_of(sp, sn) for sp, sn in
                        zip(df["species"], df["structure_name"])]
    return df


def build_donor_table(specimens, ages):
    """Donor-level table from public release fields (sex, age via the Age
    lookup, surgical etiology from the condition description, tissue source)."""
    donors = (specimens.dropna(subset=["donor_id"])
              .drop_duplicates("donor_id")
              .rename(columns={"donor_sex": "sex",
                               "donor_condition": "condition",
                               "donor_tissue_source": "tissue_source",
                               "donor_full_genotype": "full_genotype"}))
    donors["species"] = np.where(donors["donor_organism_id"] == 1, "human", "mouse")
    if ages is not None:
        donors = donors.merge(ages, on="age_id", how="left")
        donors["age_years"] = donors["age_days"] / 365.25
    else:
        donors["age_years"] = np.nan
    condition = donors["condition"].astype(str)
    donors["etiology"] = np.where(condition.str.contains("epilepsy", case=False, na=False),
                                  "epilepsy",
                                  np.where(condition.str.contains("tumor", case=False, na=False),
                                           "tumor", "other"))
    columns = ["donor_id", "donor_label", "species", "sex", "age_years",
               "condition", "etiology", "tissue_source", "full_genotype"]
    return donors[columns].sort_values("donor_id").reset_index(drop=True)


# Specimen-level columns joined onto the feature and reconstruction tables.
SPECIMEN_COLUMNS = ["specimen_id", "specimen_name", "structure_acronym",
                    "structure_name", "hemisphere", "donor_id", "donor_label",
                    "donor_organism_id", "donor_sex", "donor_condition",
                    "donor_tissue_source", "donor_full_genotype", "age_id"]


def load_metadata(data_dir):
    """Load the four metadata tables; the Age lookup is optional (human donor
    ages require it)."""
    ephys = pd.read_csv(data_dir / "metadata" / "ephys_features.csv")
    recon = pd.read_csv(data_dir / "metadata" / "neuron_reconstructions.csv")
    specimens = pd.read_csv(data_dir / "metadata" / "specimens.csv")
    age_path = data_dir / "metadata" / "age_lookup.csv"
    ages = pd.read_csv(age_path) if age_path.exists() else None
    return ephys, recon, specimens, ages


def main(data_dir: Path, results_dir: Path) -> None:
    ephys, recon, specimens, ages = load_metadata(data_dir)

    spec = specimens[SPECIMEN_COLUMNS].drop_duplicates("specimen_id")
    ephys = ephys.rename(columns={"id": "ephys_feature_id"})
    ephys = ephys.drop(columns=[c for c in SPECIMEN_COLUMNS
                                if c in ephys.columns and c != "specimen_id"])
    ephys = ephys.merge(spec, on="specimen_id", how="left")
    ephys = annotate(ephys)
    ephys["qc_ephys_ok"] = qc_mask(ephys)

    recon = recon.drop(columns=[c for c in SPECIMEN_COLUMNS
                                if c in recon.columns and c != "specimen_id"])
    recon = recon.merge(spec, on="specimen_id", how="left")
    recon = annotate(recon)

    donors = build_donor_table(specimens, ages)

    out = results_dir / "cohort"
    out.mkdir(parents=True, exist_ok=True)
    ephys.to_csv(out / "ephys_cells.csv", index=False)
    recon.to_csv(out / "reconstructions.csv", index=False)
    donors.to_csv(out / "human_donors.csv", index=False)

    print(f"electrophysiology rows: {len(ephys)} "
          f"(mouse {int((ephys.species == 'mouse').sum())} / "
          f"human {int((ephys.species == 'human').sum())})")
    print(f"QC passed: {int(ephys.qc_ephys_ok.sum())} of {len(ephys)}")
    print(f"reconstruction rows: {len(recon)} "
          f"({recon.specimen_id.nunique()} unique specimens)")
    human = donors[donors.species == "human"]
    print(f"human donors: {len(human)}; etiology: "
          f"{human.etiology.value_counts().to_dict()}")
    axon = recon[recon.neuron_reconstruction_type != "dendrite-only"]
    print(f"axon-bearing reconstructions: {len(axon)} rows "
          f"({axon.specimen_id.nunique()} unique specimens; "
          f"mouse {int((axon.species == 'mouse').sum())}, "
          f"human {int((axon.species == 'human').sum())})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                        help="directory holding metadata/ and swc/ inputs")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                        help="directory for analysis outputs")
    args = parser.parse_args()
    main(args.data_dir, args.results_dir)
