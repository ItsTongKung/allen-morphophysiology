"""Sensitivity and robustness analyses supporting the paper's interpretation.

Contents, in execution order:

1. Dendrite-only inclusion: the cell-level coupling family recomputed with
   all reconstruction types (primary analyses use axon-bearing tracings).
2. Reconstruction-completeness screens: the headline coupling recomputed
   after excluding cells in the top decile of the apical (T1) or dendritic
   (T2) tip-truncation fraction, and after keeping only cells without a
   severed apical trunk (T3).
3. Common-support coupling: both species restricted to the shared
   dendritic-length interval.
4. Range-matched coupling: 2,000 no-replacement matched human samples
   (uniform-stratified and closeness-weighted variants) with same-size
   placebo mouse subsamples.
5. Morphometric redundancy and effective dimensionality of the
   morphometric feature block.
6. First canonical correlation between the morphometric and physiological
   blocks on complete cases (descriptive).

Resampling primitives are imported from ``statistics``; output files are
written under ``<results-dir>/sensitivity/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from statistics import (BOOTSTRAP_SEED, EPHYS_FEATURES, HEADLINE_MORPHO,
                        MATCH_SEED, MIN_PAIR_N, MORPHO_FEATS, axon_bearing,
                        bh_fdr, build_strata, complete_pairs,
                        common_support_bounds, draw_no_replacement,
                        load_analysis_table, parsed_cells, spearman_rho)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Released reconstruction-summary fields included in the redundancy screen.
RELEASED_SUMMARY_FEATURES = [
    "average_bifurcation_angle_remote", "average_contraction",
    "average_diameter", "average_parent_daughter_ratio", "max_branch_order",
    "number_stems", "total_surface", "total_volume", "overall_depth",
    "overall_height",
]


def coupling_incl_dendrite_only(table, out_dir: Path) -> None:
    """Cell-level coupling family with all reconstruction types included
    (sensitivity counterpart of the primary axon-bearing families)."""
    families = []
    for species in ("mouse", "human"):
        sub = parsed_cells(table)
        sub = sub[sub["species"] == species]
        rows = []
        for mf in MORPHO_FEATS:
            for ef in EPHYS_FEATURES:
                x, y, _ = complete_pairs(sub, mf, ef)
                if len(x) < MIN_PAIR_N:
                    rows.append(dict(species=species, morpho=mf, ephys=ef,
                                     n=len(x), rho=np.nan, p=np.nan))
                    continue
                rho, p = spearmanr(x, y)
                rows.append(dict(species=species, morpho=mf, ephys=ef,
                                 n=len(x), rho=float(rho), p=float(p)))
        family = pd.DataFrame(rows)
        family["q"] = bh_fdr(family["p"].values)
        families.append(family)
    out = pd.concat(families, ignore_index=True)
    out.to_csv(out_dir / "coupling_incl_dendrite_only.csv", index=False)
    print("coupling_incl_dendrite_only: "
          f"{int((out['q'] < 0.05).sum())} of {len(out)} pairs at q < 0.05")


def completeness_screens(table, out_dir: Path) -> None:
    """Headline coupling under reconstruction-completeness screens."""
    rows = []
    for species in ("mouse", "human"):
        sub = axon_bearing(table, species)
        variants = {
            "all": sub,
            "T1_apical_trunc_low_decile":
                sub[sub["apical_tip_trunc_frac"]
                    <= sub["apical_tip_trunc_frac"].quantile(0.9)],
            "T3_apical_not_severed": sub[sub["apical_severed"] == 0],
            "T2_dend_trunc_low_decile":
                sub[sub["dend_tip_trunc_frac"]
                    <= sub["dend_tip_trunc_frac"].quantile(0.9)],
        }
        for name, variant in variants.items():
            v = variant.dropna(subset=["donor_id"])
            x, y, _ = complete_pairs(v, HEADLINE_MORPHO, "input_resistance_mohm")
            rows.append(dict(species=species, screen=name, n=len(x),
                             n_donors=v["donor_id"].nunique(),
                             rho_dend_len_vs_rin=spearman_rho(x, y)))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "completeness_screens.csv", index=False)
    print("completeness_screens:\n" + out.round(3).to_string(index=False))


def common_support_coupling(table, out_dir: Path) -> None:
    """Headline coupling with both species restricted to the shared
    common-support interval of dendritic length."""
    lo, hi = common_support_bounds(table)
    rows = [dict(item="support_lower_um", value=lo),
            dict(item="support_upper_um", value=hi)]
    for species in ("mouse", "human"):
        sub = axon_bearing(table, species)
        dend = pd.to_numeric(sub[HEADLINE_MORPHO], errors="coerce")
        restricted = sub[(dend >= lo) & (dend <= hi)]
        x, y, positions = complete_pairs(restricted, HEADLINE_MORPHO,
                                         "input_resistance_mohm")
        rows.append(dict(item=f"{species}_n_common", value=len(x)))
        rows.append(dict(item=f"{species}_donor_n_common",
                         value=int(restricted["donor_id"].nunique())))
        rows.append(dict(item=f"{species}_rho_common",
                         value=spearman_rho(x, y)))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "common_support_coupling.csv", index=False)
    print("common_support_coupling:\n" + out.round(4).to_string(index=False))


def range_matched_coupling(table, out_dir: Path) -> None:
    """Distribution-matching design: 2,000 no-replacement matched human
    samples per variant, with same-size placebo mouse subsamples drawn from
    the same rng stream after the human draw; the matched-human headline
    coupling (and sag / f-I slope / latency companions) is summarized across
    repetitions."""
    mouse = axon_bearing(table, "mouse")
    human = axon_bearing(table, "human")
    mdend = pd.to_numeric(mouse[HEADLINE_MORPHO], errors="coerce").values.astype(float)
    hdend = pd.to_numeric(human[HEADLINE_MORPHO], errors="coerce").values.astype(float)
    mouse, human = mouse[np.isfinite(mdend)], human[np.isfinite(hdend)]
    mdend, hdend = mdend[np.isfinite(mdend)], hdend[np.isfinite(hdend)]
    outcomes = {ef: pd.to_numeric(human[ef], errors="coerce").values.astype(float)
                for ef in ("input_resistance_mohm", "sag", "f_i_curve_slope",
                           "latency")}
    rin_m = pd.to_numeric(mouse["input_resistance_mohm"],
                          errors="coerce").values.astype(float)

    centers, counts, width = build_strata(mdend)
    reps = []
    for variant, weighted, seed in (("closeness_weighted", True, BOOTSTRAP_SEED),
                                    ("uniform_stratified", False, MATCH_SEED)):
        rng = np.random.default_rng(seed)
        for rep in range(2000):
            hidx = draw_no_replacement(rng, hdend, centers, counts, width,
                                       weighted)
            msel = rng.choice(len(mdend), size=len(hidx), replace=False)
            row = dict(variant=variant, rep=rep, n_matched=len(hidx),
                       rho_mouse_placebo=spearman_rho(mdend[msel], rin_m[msel]))
            for ef, values in outcomes.items():
                row[f"rho_human_matched_{ef}"] = spearman_rho(hdend[hidx],
                                                              values[hidx])
            reps.append(row)
    per = pd.DataFrame(reps)
    per.to_csv(out_dir / "range_matched_coupling_reps.csv", index=False)

    rows = []
    for variant, group in per.groupby("variant"):
        for column in [c for c in group.columns if c.startswith("rho_")]:
            rows.append(dict(variant=variant, metric=column,
                             median=float(group[column].median()),
                             p2_5=float(group[column].quantile(0.025)),
                             p97_5=float(group[column].quantile(0.975))))
        rows.append(dict(variant=variant, metric="mean_n_matched",
                         median=round(float(group["n_matched"].mean()), 1),
                         p2_5=np.nan, p97_5=np.nan))
    pd.DataFrame(rows).to_csv(out_dir / "range_matched_coupling_summary.csv",
                              index=False)
    unif = per[per["variant"] == "uniform_stratified"]
    print(f"range_matched_coupling: uniform-stratified matched-human Rin "
          f"median {unif['rho_human_matched_input_resistance_mohm'].median():.3f}, "
          f"placebo mouse median "
          f"{unif['rho_mouse_placebo'].median():.3f}, "
          f"mean matched n {unif['n_matched'].mean():.1f}")


def morpho_redundancy(table, out_dir: Path) -> None:
    """Spearman redundancy structure of the morphometric block (primary
    re-derived features plus released summary fields) and the effective
    dimensionality (principal components to 95% variance)."""
    pairs, dims = [], []
    for species in ("mouse", "human"):
        sub = parsed_cells(table)
        sub = sub[sub["species"] == species]
        cols = [c for c in MORPHO_FEATS + RELEASED_SUMMARY_FEATURES
                if pd.to_numeric(sub[c], errors="coerce").notna().sum() >= 30]
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                x = pd.to_numeric(sub[a], errors="coerce")
                y = pd.to_numeric(sub[b], errors="coerce")
                ok = x.notna() & y.notna()
                if ok.sum() < 20:
                    continue
                rho, p = spearmanr(x[ok], y[ok])
                pairs.append(dict(species=species, feat_a=a, feat_b=b,
                                  rho=float(rho), p=float(p), n=int(ok.sum())))
        block = sub[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(block) >= 20:
            corr = np.corrcoef(block.values.T)
            eig = np.clip(np.sort(np.linalg.eigvalsh(corr))[::-1], 1e-12, None)
            cumulative = np.cumsum(eig / eig.sum())
            k95 = int(np.searchsorted(cumulative, 0.95) + 1)
            for j in range(len(eig)):
                dims.append(dict(species=species, pc=j + 1,
                                 explained_frac=float(eig[j] / eig.sum()),
                                 cumulative_frac=float(cumulative[j]),
                                 components_to_95pct=k95))
            print(f"morpho_redundancy: {species} n_complete={len(block)}, "
                  f"components to 95% variance = {k95}")
    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(out_dir / "morpho_redundancy.csv", index=False)
    pd.DataFrame(dims).to_csv(out_dir / "morpho_effective_dim.csv", index=False)
    n_hi = int((pairs_df["rho"].abs() > 0.9).sum())
    print(f"morpho_redundancy: {n_hi} of {len(pairs_df)} pairs with |rho| > 0.9")


def _cca_r1(X, Y):
    """First canonical correlation via eigendecomposition whitening
    (collinearity-safe); returns r1 and the within-block eigenvalue spectra."""
    Xc = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    Yc = (Y - Y.mean(axis=0)) / (Y.std(axis=0, ddof=1) + 1e-12)
    n = len(Xc)
    Cxx = Xc.T @ Xc / (n - 1)
    Cyy = Yc.T @ Yc / (n - 1)
    Cxy = Xc.T @ Yc / (n - 1)
    ex, Ux = np.linalg.eigh(Cxx)
    ey, Uy = np.linalg.eigh(Cyy)
    ex = np.clip(ex, 1e-6, None)
    ey = np.clip(ey, 1e-6, None)
    Wx = Ux @ np.diag(ex ** -0.5) @ Ux.T
    Wy = Uy @ np.diag(ey ** -0.5) @ Uy.T
    r1 = np.linalg.svd(Wx @ Cxy @ Wy, compute_uv=False)[0]
    return float(min(max(r1, 0.0), 1.0)), ex, ey


def canonical_correlation(table, out_dir: Path) -> None:
    """First canonical correlation between the morphometric and
    physiological blocks on complete cases (descriptive structure; upward
    bias is expected under correlated blocks)."""
    rows = []
    for species in ("mouse", "human"):
        sub = axon_bearing(table, species)
        block = sub[MORPHO_FEATS + EPHYS_FEATURES].apply(
            pd.to_numeric, errors="coerce").dropna()
        r1, morpho_eig, ephys_eig = _cca_r1(block[MORPHO_FEATS].values,
                                            block[EPHYS_FEATURES].values)
        rows.append(dict(species=species, n_complete=len(block), r1=r1,
                         morpho_eig_min=float(morpho_eig.min()),
                         morpho_eig_max=float(morpho_eig.max()),
                         morpho_condition=float(morpho_eig.max()
                                                / max(morpho_eig.min(), 1e-12)),
                         ephys_eig_min=float(ephys_eig.min()),
                         ephys_eig_max=float(ephys_eig.max()),
                         ephys_condition=float(ephys_eig.max()
                                               / max(ephys_eig.min(), 1e-12))))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "cca_complete_cases.csv", index=False)
    print("cca_complete_cases:\n" + out.round(4).to_string(index=False))


def main(data_dir: Path, results_dir: Path) -> None:
    del data_dir  # analyses read the prebuilt analysis table only
    table = load_analysis_table(results_dir)
    out_dir = results_dir / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    coupling_incl_dendrite_only(table, out_dir)
    completeness_screens(table, out_dir)
    common_support_coupling(table, out_dir)
    range_matched_coupling(table, out_dir)
    morpho_redundancy(table, out_dir)
    canonical_correlation(table, out_dir)
    print(f"sensitivity outputs written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                        help="directory holding metadata/ and swc/ inputs")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                        help="directory for analysis outputs")
    args = parser.parse_args()
    main(args.data_dir, args.results_dir)
