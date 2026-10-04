"""Re-derive arborization morphometrics from the released SWC reconstructions.

Parses every released SWC file with explicit node-id to row-index parent
resolution, recenters coordinates on the soma centroid, and computes, per
released tracing:

- axonal and dendritic (basal + apical combined) compartment metrics: total
  length, branch points, tips, maximum radial extent, maximum path distance,
  and mean branch diameter;
- the apical/basal compartment split from the SWC structural-type codes
  (type 3 = basal dendrite, type 4 = apical dendrite), a dimension absent
  from the released summary tables;
- reconstruction-completeness proxies (tip-truncation fractions and a
  severed-apical-trunk flag);
- 3-D Sholl profiles on 10-600 um spheres in 10 um bins, both over all
  parent-child segments (the per-cell feature used downstream) and over
  dendritic segments only (the group-profile view).

For the 33 specimens with two released tracings, compartment metrics and
completeness proxies are taken from the tracing with the denser apical
reconstruction (most apical tips, then basal tips) so that all rows of a
specimen carry one consistent compartment description.

Outputs (under ``<results-dir>/morphology/``):
    swc_morphometrics.csv   one row per released tracing
    validation_vs_released.csv  parsing-consistency check against the
                            released summary tables (Spearman rho)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]

SHOLL_MIN, SHOLL_MAX, SHOLL_STEP = 10, 600, 10
SHOLL_RADII = np.arange(SHOLL_MIN, SHOLL_MAX + 1e-9, SHOLL_STEP, dtype=float)

# Columns describing a specimen's compartments rather than an individual
# tracing; for doubly-traced specimens these take the denser tracing's values.
SPECIMEN_LEVEL_COLUMNS = [
    "apical_len", "basal_len", "apical_extent", "basal_extent",
    "n_apical_tips", "n_basal_tips", "apical_tip_trunc_frac",
    "dend_tip_trunc_frac", "apical_severed",
]


def parse_swc(path):
    """Parse an SWC file into (ids, types, coordinates, radii, parent rows).

    Parent-child links are resolved through node ids (not row offsets), so
    arbitrarily ordered files parse correctly.  Root nodes get parent row -1.
    """
    ids, typ, xyz, rad, pids = [], [], [], [], []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            ids.append(int(float(parts[0])))
            typ.append(int(float(parts[1])))
            xyz.append([float(parts[2]), float(parts[3]), float(parts[4])])
            rad.append(float(parts[5]))
            pids.append(int(float(parts[6])))
    ids = np.asarray(ids)
    typ = np.asarray(typ, dtype=int)
    xyz = np.asarray(xyz, dtype=float)
    rad = np.asarray(rad, dtype=float)
    pids = np.asarray(pids, dtype=int)
    id_to_row = {i: k for k, i in enumerate(ids)}
    parent_row = np.asarray([id_to_row.get(p, -1) for p in pids], dtype=int)
    return ids, typ, xyz, rad, parent_row


def soma_center(typ, xyz):
    """Soma centroid (falls back to the centroid of all nodes)."""
    soma = xyz[typ == 1]
    return soma.mean(axis=0) if len(soma) else xyz.mean(axis=0)


def compartment_metrics(typ, xyz, rad, parent_row, center):
    """Axon (type 2) and combined dendritic (types 3/4) compartment metrics.

    Length sums the parent-child segment norms of the compartment's nodes
    (parents of any type); branch points are nodes with >= 2 same-compartment
    children; tips are nodes with none.  Path distances accumulate parent-child
    segment norms over nodes ordered by radial distance from the soma.
    """
    out = {}
    for name, allowed in (("axon", (2,)), ("dend", (3, 4))):
        m = np.isin(typ, allowed)
        if not m.any():
            for suffix in ("len", "nbranch", "ntips", "max_extent",
                           "maxpath", "meandiam"):
                out[f"{name}_{suffix}"] = np.nan
            continue
        idx = np.flatnonzero(m)
        has_parent = parent_row[idx] >= 0
        if has_parent.any():
            seg = xyz[idx[has_parent]] - xyz[parent_row[idx[has_parent]]]
            length = float(np.linalg.norm(seg, axis=1).sum())
        else:
            length = 0.0
        child_count = np.zeros(len(typ), dtype=int)
        np.add.at(child_count, parent_row[idx[has_parent]], 1)
        out[f"{name}_len"] = length
        out[f"{name}_nbranch"] = int(np.sum(child_count[idx] >= 2))
        out[f"{name}_ntips"] = int(np.sum(child_count[idx] == 0))
        out[f"{name}_max_extent"] = float(
            np.max(np.linalg.norm(xyz[idx] - center, axis=1)))
        depth = np.zeros(len(typ), dtype=float)
        order = np.argsort(np.linalg.norm(xyz - center, axis=1))
        for k in order:
            p = parent_row[k]
            if p >= 0:
                depth[k] = depth[p] + np.linalg.norm(xyz[k] - xyz[p])
        out[f"{name}_maxpath"] = float(np.max(depth[idx]))
        out[f"{name}_meandiam"] = float(np.mean(rad[idx]))
    return out


def connected_tips(typ, parent_row):
    """Boolean mask of connected nodes that are not the parent of any node."""
    ok = parent_row >= 0
    is_parent_of = np.zeros(len(typ), dtype=bool)
    is_parent_of[parent_row[ok]] = True
    return ok & ~is_parent_of


def apical_basal_metrics(typ, xyz, parent_row):
    """Apical (type 4) and basal (type 3) compartment metrics.

    Compartment lengths sum segments whose parent and child both belong to
    the compartment (the soma-to-primary-dendrite stub is excluded), so
    apical_len + basal_len approximates dend_len up to that stub.  An apical
    length of zero denotes a cell without a reconstructed apical dendrite.
    """
    radial = np.linalg.norm(xyz, axis=1)  # caller passes soma-centered coords
    tips = connected_tips(typ, parent_row)
    apical, basal = typ == 4, typ == 3
    out = {}
    for name, mask in (("apical", apical), ("basal", basal)):
        has_parent = parent_row[mask] >= 0
        same_type = has_parent & (typ[parent_row[mask]] == typ[mask])
        length = float(np.linalg.norm(
            xyz[mask][same_type] - xyz[parent_row[mask]][same_type],
            axis=1).sum()) if same_type.any() else 0.0
        out[f"{name}_len"] = length
        out[f"{name}_extent"] = float(np.max(radial[mask])) if mask.any() else np.nan
        n_tips = int(np.sum(mask & tips))
        out[f"n_{name}_tips"] = n_tips
    apical_radial = radial[apical]
    if apical.any():
        ap_max = float(apical_radial.max())
        tip_radial = radial[apical & tips]
        out["apical_tip_trunc_frac"] = (float(np.mean(tip_radial > 0.9 * ap_max))
                                        if len(tip_radial) else np.nan)
    else:
        out["apical_tip_trunc_frac"] = np.nan
    dend = apical | basal
    dend_radial = radial[dend]
    if dend.any():
        dend_max = float(dend_radial.max())
        tip_radial = radial[dend & tips]
        out["dend_tip_trunc_frac"] = (float(np.mean(tip_radial > 0.9 * dend_max))
                                      if len(tip_radial) else np.nan)
    else:
        out["dend_tip_trunc_frac"] = np.nan
    out["apical_severed"] = int(out["n_apical_tips"] < 2 and apical.any())
    return out


def sholl_profile(xyz, parent_row, center):
    """Sholl crossing counts over all parent-child segments."""
    has_parent = parent_row >= 0
    if has_parent.sum() < 1 or len(xyz) < 2:
        return np.zeros(len(SHOLL_RADII), dtype=int)
    parents = xyz[parent_row[has_parent]]
    children = xyz[has_parent]
    d_parent = np.linalg.norm(parents - center, axis=1)[:, None] - SHOLL_RADII[None, :]
    d_child = np.linalg.norm(children - center, axis=1)[:, None] - SHOLL_RADII[None, :]
    crossings = np.sign(d_parent) * np.sign(d_child) < 0
    return crossings.sum(axis=0).astype(int)


def sholl_profile_dendritic(typ, xyz, parent_row, center):
    """Sholl crossing counts over dendritic segments only (types 3/4, with
    soma-rooted primary dendrites allowed)."""
    has_parent = parent_row >= 0
    seg = (has_parent & np.isin(typ, (3, 4))
           & np.isin(typ[parent_row], (1, 3, 4)))
    if seg.sum() < 1 or len(xyz) < 2:
        return np.zeros(len(SHOLL_RADII), dtype=int)
    parents = xyz[parent_row[seg]]
    children = xyz[seg]
    d_parent = np.linalg.norm(parents - center, axis=1)[:, None] - SHOLL_RADII[None, :]
    d_child = np.linalg.norm(children - center, axis=1)[:, None] - SHOLL_RADII[None, :]
    crossings = np.sign(d_parent) * np.sign(d_child) < 0
    return crossings.sum(axis=0).astype(int)


def swc_path(swc_dir: Path, specimen_id, recon_id) -> Path:
    return swc_dir / f"specimen_{int(specimen_id)}_recon_{int(recon_id)}.swc"


def derive_tracing_metrics(swc_dir: Path, row) -> dict:
    """Compute all per-tracing metrics for one reconstruction-table row."""
    sid, rid = int(row["specimen_id"]), int(row["id"])
    base = dict(specimen_id=sid, recon_id=rid, parsed=False,
                reconstruction_type=row.get("neuron_reconstruction_type"))
    path = swc_path(swc_dir, sid, rid)
    if not path.exists():
        return base
    try:
        ids, typ, xyz, rad, parent_row = parse_swc(path)
        center = soma_center(typ, xyz)
        coords = xyz - center
        f = dict(base, parsed=True, n_nodes=len(ids))
        f.update(compartment_metrics(typ, xyz, rad, parent_row, center))
        f.update(apical_basal_metrics(typ, coords, parent_row))
        counts = sholl_profile(xyz, parent_row, center)
        for radius, n in zip(SHOLL_RADII, counts):
            f[f"sholl_all_{int(radius)}"] = int(n)
        f["sholl_peak"] = float(counts.max())
        f["sholl_peak_radius"] = float(SHOLL_RADII[int(np.argmax(counts))])
        f["sholl_auc"] = float(np.trapezoid(counts.astype(float), SHOLL_RADII))
        dend_counts = sholl_profile_dendritic(typ, xyz, parent_row, center)
        for radius, n in zip(SHOLL_RADII, dend_counts):
            f[f"sholl_dend_{int(radius)}"] = int(n)
        return f
    except Exception as err:  # record and continue; flagged via parsed=False
        base["error"] = str(err)[:120]
        return base


def consolidate_specimen_compartments(metrics: pd.DataFrame) -> pd.DataFrame:
    """Broadcast compartment metrics across doubly-traced specimens.

    For specimens with more than one parsed tracing, the specimen-level
    compartment columns take the values of the tracing with the most apical
    tips (ties broken by basal tips, then reconstruction id) so that every
    row of a specimen carries one consistent compartment description.
    """
    metrics = metrics.copy()
    parsed_flag = metrics["parsed"].fillna(False).astype(bool)
    duplicated = metrics["specimen_id"].duplicated(keep=False) & parsed_flag
    if not duplicated.any():
        return metrics
    multi = metrics[duplicated]
    for sid, group in multi.groupby("specimen_id"):
        if len(group) < 2:
            continue
        keep = group.sort_values(["n_apical_tips", "n_basal_tips", "recon_id"],
                                 ascending=[False, False, True]).index[0]
        idx = metrics.index[metrics["specimen_id"] == sid]
        for col in SPECIMEN_LEVEL_COLUMNS:
            metrics.loc[idx, col] = metrics.at[keep, col]
    return metrics


def validate_against_released(metrics: pd.DataFrame, recon: pd.DataFrame) -> pd.DataFrame:
    """Spearman agreement between re-derived and released summary metrics.

    This is a parsing-consistency check demonstrating that the parser
    reproduces the release's computation; it is not an independent
    biological validation.
    """
    released = recon[["specimen_id", "id", "total_length", "number_branches",
                      "number_nodes", "max_euclidean_distance"]].copy()
    released = released.rename(columns={"id": "recon_id"})
    merged = metrics.merge(released, on=["specimen_id", "recon_id"], how="inner")
    rows = []
    for derived, released_field in (("dend_len", "total_length"),
                                    ("dend_nbranch", "number_branches"),
                                    ("n_nodes", "number_nodes"),
                                    ("dend_max_extent", "max_euclidean_distance")):
        pair = merged[[derived, released_field]].apply(pd.to_numeric,
                                                       errors="coerce").dropna()
        if len(pair) < 3:
            continue
        rho, p = spearmanr(pair[derived], pair[released_field])
        rows.append(dict(derived_metric=derived, released_field=released_field,
                         n=len(pair), spearman_rho=rho, p=p))
    return pd.DataFrame(rows)


def main(data_dir: Path, results_dir: Path) -> None:
    recon = pd.read_csv(results_dir / "cohort" / "reconstructions.csv")
    swc_dir = data_dir / "swc"

    records = [derive_tracing_metrics(swc_dir, row) for _, row in recon.iterrows()]
    metrics = pd.DataFrame(records)
    metrics = consolidate_specimen_compartments(metrics)
    metrics = metrics.merge(
        recon[["specimen_id", "id", "species", "broad_class", "layer",
               "area_group", "donor_id"]]
        .rename(columns={"id": "recon_id"}),
        on=["specimen_id", "recon_id"], how="left")

    out = results_dir / "morphology"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "swc_morphometrics.csv", index=False)

    parsed = metrics[metrics["parsed"].fillna(False).astype(bool)]
    validation = validate_against_released(parsed, recon)
    validation.to_csv(out / "validation_vs_released.csv", index=False)

    print(f"tracings processed: {len(metrics)} (parsed OK: {len(parsed)})")
    print("parsing-consistency validation against released summaries:")
    print(validation.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                        help="directory holding metadata/ and swc/ inputs")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                        help="directory for analysis outputs")
    args = parser.parse_args()
    main(args.data_dir, args.results_dir)
