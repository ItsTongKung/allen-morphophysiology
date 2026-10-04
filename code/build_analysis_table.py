"""Join re-derived morphology with electrophysiology into the analysis table.

Merges the per-tracing morphometric table with the per-specimen
electrophysiology table and donor metadata, producing the single
analysis-ready dataframe used by the statistics, sensitivity, and prediction
modules.  The table keeps one row per released tracing: analyses that select
axon-bearing reconstructions ('full' tracings, i.e. anything other than
'dendrite-only') form the primary coupling samples, and per-test complete-case
handling is applied downstream by each analysis.

No plotting and no analysis logic live here.

Output: ``<results-dir>/analysis_table.csv``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Column groups carried into the analysis table.
ID_COLUMNS = ["specimen_id", "recon_id", "donor_id", "species", "broad_class",
              "cre_line", "layer", "structure_acronym", "structure_name",
              "area_group", "reconstruction_type", "parsed", "n_nodes",
              "qc_ephys_ok"]
EPHYS_COLUMNS = [
    "input_resistance_mohm", "tau", "ri", "sag", "adaptation", "avg_isi",
    "f_i_curve_slope", "latency", "upstroke_downstroke_ratio_long_square",
    "threshold_v_long_square", "threshold_i_long_square", "vrest",
    "fast_trough_v_long_square", "peak_v_long_square", "vm_for_sag",
]
MORPHOLOGY_COLUMNS = [
    "dend_len", "axon_len", "dend_nbranch", "dend_ntips", "dend_max_extent",
    "dend_maxpath", "axon_max_extent", "axon_maxpath", "dend_meandiam",
    "sholl_peak", "apical_len", "basal_len", "apical_extent", "basal_extent",
    "apical_tip_trunc_frac", "dend_tip_trunc_frac", "apical_severed",
]
# Released reconstruction-summary fields used by the redundancy diagnostics.
RELEASED_SUMMARY_COLUMNS = [
    "total_length", "number_branches", "number_nodes", "max_euclidean_distance",
    "average_bifurcation_angle_remote", "average_contraction",
    "average_diameter", "average_parent_daughter_ratio", "max_branch_order",
    "number_stems", "total_surface", "total_volume", "overall_depth",
    "overall_height", "hausdorff_dimension",
]
DONOR_COLUMNS = ["sex", "age_years", "etiology", "tissue_source"]


def main(data_dir: Path, results_dir: Path) -> None:
    ephys = pd.read_csv(results_dir / "cohort" / "ephys_cells.csv")
    recon = pd.read_csv(results_dir / "cohort" / "reconstructions.csv")
    donors = pd.read_csv(results_dir / "cohort" / "human_donors.csv")
    metrics = pd.read_csv(results_dir / "morphology" / "swc_morphometrics.csv")

    # Per-specimen electrophysiology; cohort labels already ride on the
    # morphometric table (species, broad class, layer, area group, donor).
    ephys = ephys[["specimen_id"] + EPHYS_COLUMNS + ["qc_ephys_ok"]]
    # Cre line and released summary fields, taken from each specimen's
    # canonical released row (the non-superseded tracing where a specimen
    # carries two).
    released_all = recon[["id", "specimen_id", "superseded", "cre_line",
                          "structure_acronym", "structure_name"]
                         + RELEASED_SUMMARY_COLUMNS]
    canonical = released_all[~released_all["superseded"].fillna(False).astype(bool)]
    released = canonical.drop_duplicates("specimen_id").drop(
        columns=["id", "superseded"])
    table = metrics.merge(ephys, on="specimen_id", how="left")
    table = table.merge(released, on="specimen_id", how="left")
    table = table.merge(donors[["donor_id"] + DONOR_COLUMNS],
                        on="donor_id", how="left")

    columns = (ID_COLUMNS + EPHYS_COLUMNS + MORPHOLOGY_COLUMNS
               + RELEASED_SUMMARY_COLUMNS + DONOR_COLUMNS)
    # Row order follows the released reconstruction table; downstream
    # resampling procedures iterate donors in order of first appearance, so
    # the release order is preserved deliberately (no re-sorting).
    table = table[columns]
    table.to_csv(results_dir / "analysis_table.csv", index=False)

    parsed = table[table["parsed"]]
    axon = parsed[parsed["reconstruction_type"] != "dendrite-only"]
    for species in ("mouse", "human"):
        sub = axon[axon["species"] == species]
        print(f"{species}: {len(sub)} axon-bearing tracings, "
              f"{sub['donor_id'].nunique()} donors")
    print(f"analysis table written: {results_dir / 'analysis_table.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                        help="directory holding metadata/ and swc/ inputs")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                        help="directory for analysis outputs")
    args = parser.parse_args()
    main(args.data_dir, args.results_dir)
