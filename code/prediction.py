"""Morphology-to-physiology prediction benchmark with donor-grouped evaluation.

Primary benchmark: ridge regression (alpha = 1) predicting six canonical
electrophysiological features from morphometric features, standardized within
each training fold (no leakage), evaluated by leave-one-donor-out
cross-validation; performance is the Spearman correlation between predicted
and observed values over all held-out cells.  Models: size-only baseline
(total dendritic length), a three-feature subset, and the full ten-feature
morphometric block.  The size-beyond-size delta (full minus size-only) is
computed over 100 shared random donor partitions (identical folds for both
models), summarized by median and 2.5-97.5 percentile interval.

Comparability stress tests reuse the identical protocol (same estimator, same
alpha, no re-tuning, no model selection):

- compartment prediction (total dendritic, apical-only, basal-only,
  apical+basal, full 10-feature models);
- common-support prediction (both species restricted to the shared
  dendritic-length interval);
- range-matched prediction (LODO evaluation inside each of 2,000 no-replacement
  range-matched human samples);
- composition sensitivity (mouse excitatory-compatible subsets against the
  within-species samples; subsets below the minimum-inference rule are
  reported with an explicit status flag and no forced CV estimates).

All benchmarks describe the released samples; they are not tuned, not
model-selected, and not calibrated population statements.

Outputs are written under ``<results-dir>/prediction/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from statistics import (BOOTSTRAP_SEED, DENDRITIC_FEATS, MATCH_SEED,
                        MORPHO_FEATS, axon_bearing, build_strata,
                        common_support_bounds, load_analysis_table,
                        match_human_only)

REPO_ROOT = Path(__file__).resolve().parents[1]

OUTCOMES = ["input_resistance_mohm", "f_i_curve_slope", "latency",
            "adaptation", "sag", "tau"]
MODELS = {
    "B2_size_only": ["dend_len"],
    "B3_small_subset": ["dend_len", "dend_max_extent", "dend_nbranch"],
    "M_full_10": MORPHO_FEATS,
}
N_PARTITIONS = 100
PARTITION_SEED_BASE = 999   # paired partitions: seed 999 + repetition


def cv_score(df, xcols, ycol, fold_of_donor):
    """One cross-validation pass; returns the Spearman correlation between
    predicted and observed values over all held-out cells (NaN when fewer
    than 10 complete cases or a degenerate target)."""
    d = df[["donor_id"] + list(xcols) + [ycol]].copy()
    for c in list(xcols) + [ycol]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(axis=1, how="all")
    xcols = [c for c in d.columns if c not in ("donor_id", ycol)]
    d = d.dropna()
    if len(d) < 10:
        return np.nan, 0
    X = d[xcols].values.astype(float)
    y = d[ycol].values.astype(float)
    donors = d["donor_id"].values
    preds = np.full(len(y), np.nan)
    for fold in sorted(set(fold_of_donor.values())):
        test = np.fromiter((fold_of_donor[u] == fold for u in donors),
                           dtype=bool, count=len(donors))
        train = ~test
        if train.sum() < 10 or test.sum() < 1:
            continue
        scaler = StandardScaler().fit(X[train])
        model = Ridge(alpha=1.0).fit(scaler.transform(X[train]), y[train])
        preds[test] = model.predict(scaler.transform(X[test]))
    ok = np.isfinite(preds)
    if ok.sum() < 10 or np.ptp(y[ok]) == 0:
        return np.nan, int(ok.sum())
    ra = rankdata(preds[ok])
    rb = rankdata(y[ok])
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return np.nan, int(ok.sum())
    return float((ra * rb).sum() / denom), int(ok.sum())


def lodo_score(df, xcols, ycol):
    """Leave-one-donor-out cross-validation: each donor is its own fold."""
    uniq = np.unique(df["donor_id"].values)
    fold_of_donor = {u: i for i, u in enumerate(uniq)}
    return cv_score(df, xcols, ycol, fold_of_donor)


def paired_partition_deltas(df, alt_xcols, base_xcols, ycol,
                            n_partitions=N_PARTITIONS):
    """Full-minus-base prediction delta over shared random 5-fold donor
    partitions (identical folds for both models)."""
    donors = np.unique(df["donor_id"].values)
    diffs = []
    for rep in range(n_partitions):
        rng = np.random.default_rng(PARTITION_SEED_BASE + rep)
        fold_of_donor = {u: int(rng.integers(0, 5)) for u in donors}
        r_alt, _ = cv_score(df, alt_xcols, ycol, fold_of_donor)
        r_base, _ = cv_score(df, base_xcols, ycol, fold_of_donor)
        if np.isfinite(r_alt) and np.isfinite(r_base):
            diffs.append(r_alt - r_base)
    if not diffs:
        return np.nan, np.nan, np.nan, 0
    return (float(np.median(diffs)), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)), len(diffs))


def prediction_scope(table, species):
    """Primary prediction scope: parsed axon-bearing tracings with a known
    donor, per species."""
    sub = axon_bearing(table, species)
    return sub.dropna(subset=["donor_id"])


def benchmark(table, out_dir: Path) -> None:
    """Primary LODO ridge benchmark with repeated-partition spread and
    size-beyond-size paired deltas."""
    rows = []
    for species in ("mouse", "human"):
        sub = prediction_scope(table, species)
        for ycol in OUTCOMES:
            entry = dict(species=species, outcome=ycol,
                         n_donors_scope=int(sub["donor_id"].nunique()))
            for model, xcols in MODELS.items():
                rho, n_held = lodo_score(sub, xcols, ycol)
                entry[f"lodo_rho_{model}"] = rho
                entry[f"n_held_{model}"] = n_held
            donors = np.unique(sub["donor_id"].values)
            spread = []
            for rep in range(N_PARTITIONS):
                rng = np.random.default_rng(BOOTSTRAP_SEED + rep)
                fold_of_donor = {u: int(rng.integers(0, 5)) for u in donors}
                r, _ = cv_score(sub, MODELS["M_full_10"], ycol, fold_of_donor)
                if np.isfinite(r):
                    spread.append(r)
            entry["repeated_cv_median"] = (float(np.median(spread))
                                           if spread else np.nan)
            entry["repeated_cv_lo"] = (float(np.percentile(spread, 2.5))
                                       if spread else np.nan)
            entry["repeated_cv_hi"] = (float(np.percentile(spread, 97.5))
                                       if spread else np.nan)
            (entry["sbs_delta_median"], entry["sbs_delta_lo"],
             entry["sbs_delta_hi"], entry["sbs_reps"]) = paired_partition_deltas(
                sub, MODELS["M_full_10"], MODELS["B2_size_only"], ycol)
            rows.append(entry)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "prediction_benchmark.csv", index=False)
    rin = out[out["outcome"] == "input_resistance_mohm"]
    print("benchmark (input resistance):")
    print(rin[["species", "lodo_rho_B2_size_only", "lodo_rho_M_full_10",
               "sbs_delta_median", "sbs_delta_lo", "sbs_delta_hi"]].round(3).to_string(index=False))


def compartment_prediction(table, out_dir: Path) -> None:
    """Compartment prediction decomposition for input resistance; an
    apical length of zero denotes a cell without an apical dendrite."""
    models = {
        "M1_dend_len": ["dend_len"],
        "M2_apical_len": ["apical_len"],
        "M3_basal_len": ["basal_len"],
        "M4_apical_plus_basal": ["apical_len", "basal_len"],
        "M5_full_10": MORPHO_FEATS,
    }
    table = table.copy()
    table["apical_len"] = pd.to_numeric(table["apical_len"], errors="coerce").fillna(0.0)
    table["basal_len"] = pd.to_numeric(table["basal_len"], errors="coerce").fillna(0.0)
    parsed = table[table["parsed"].fillna(False).astype(bool)]
    scopes = {
        "human_v3_scope": parsed[(parsed["species"] == "human")
                                 & (parsed["reconstruction_type"] != "dendrite-only")],
        "mouse_v3_scope": parsed[(parsed["species"] == "mouse")
                                 & (parsed["reconstruction_type"] != "dendrite-only")],
        "mouse_pooled_apical_bearing": parsed[(parsed["species"] == "mouse")
                                              & (parsed["apical_len"] > 0)],
    }
    ycol = "input_resistance_mohm"
    rows = []
    for scope, sub in scopes.items():
        sub = sub.dropna(subset=["donor_id"])
        base = dict(scope=scope, n_cells_scope=len(sub),
                    n_donors_scope=int(sub["donor_id"].nunique()),
                    n_apical_bearing=int((sub["apical_len"] > 0).sum()))
        for model, xcols in models.items():
            rho, n_held = lodo_score(sub, xcols, ycol)
            rows.append(dict(**base, model=model, lodo_rho=rho, n_held=n_held))
        for alt in list(models)[1:]:
            median, lo, hi, reps = paired_partition_deltas(
                sub, models[alt], models["M1_dend_len"], ycol)
            rows.append(dict(**base, model=f"delta_{alt}_vs_M1",
                             lodo_rho=median, n_held=reps,
                             delta_lo=lo, delta_hi=hi))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "compartment_prediction.csv", index=False)
    print("compartment_prediction:\n"
          + out[~out["model"].str.startswith("delta_")]
          .pivot_table(index="scope", columns="model", values="lodo_rho")
          .round(3).to_string())


def common_support_prediction(table, out_dir: Path) -> None:
    """The identical prediction protocol re-run inside the shared
    common-support restriction of dendritic length."""
    lo, hi = common_support_bounds(table)
    rows = []
    for species in ("mouse", "human"):
        sub = prediction_scope(table, species)
        dend = pd.to_numeric(sub["dend_len"], errors="coerce")
        sub = sub[np.isfinite(dend) & (dend >= lo) & (dend <= hi)]
        for ycol in OUTCOMES:
            r_size, n_size = lodo_score(sub, MODELS["B2_size_only"], ycol)
            r_full, n_full = lodo_score(sub, MODELS["M_full_10"], ycol)
            median, d_lo, d_hi, reps = paired_partition_deltas(
                sub, MODELS["M_full_10"], MODELS["B2_size_only"], ycol)
            rows.append(dict(species=species, outcome=ycol,
                             support_lo_um=lo, support_hi_um=hi,
                             n_cells_scope=len(sub),
                             n_donors_scope=int(sub["donor_id"].nunique()),
                             lodo_rho_size_only=r_size, n_held_size=n_size,
                             lodo_rho_full=r_full, n_held_full=n_full,
                             sbs_delta_median=median, sbs_delta_lo=d_lo,
                             sbs_delta_hi=d_hi, sbs_reps=reps))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "prediction_common_support.csv", index=False)
    rin = out[out["outcome"] == "input_resistance_mohm"]
    print("common_support_prediction (input resistance):")
    print(rin[["species", "n_cells_scope", "lodo_rho_size_only",
               "lodo_rho_full", "sbs_delta_median"]].round(3).to_string(index=False))


def range_matched_prediction(table, out_dir: Path) -> None:
    """Size-only vs full-morphology LODO prediction of input resistance
    inside each of 2,000 no-replacement range-matched human samples."""
    ycol = "input_resistance_mohm"
    mouse = axon_bearing(table, "mouse")
    human = axon_bearing(table, "human")
    mdend = pd.to_numeric(mouse["dend_len"], errors="coerce").values.astype(float)
    hdend = pd.to_numeric(human["dend_len"], errors="coerce").values.astype(float)
    mouse, human = mouse[np.isfinite(mdend)], human[np.isfinite(hdend)]
    mdend, hdend = mdend[np.isfinite(mdend)], hdend[np.isfinite(hdend)]
    centers, counts, width = build_strata(mdend)

    rows = []
    rng = np.random.default_rng(MATCH_SEED)
    for rep in range(2000):
        hidx = match_human_only(rng, hdend, centers, counts, width)
        sub = human.iloc[hidx]
        r_size, n_size = lodo_score(sub, MODELS["B2_size_only"], ycol)
        r_full, n_full = lodo_score(sub, MODELS["M_full_10"], ycol)
        rows.append(dict(rep=rep, n_matched=len(hidx),
                         n_donors=int(sub["donor_id"].nunique()),
                         rho_size=r_size, rho_full=r_full,
                         delta=(r_full - r_size)
                         if np.isfinite(r_size) and np.isfinite(r_full) else np.nan,
                         n_held_size=n_size, n_held_full=n_full))
    per = pd.DataFrame(rows)
    per.to_csv(out_dir / "prediction_range_matched_reps.csv", index=False)

    deltas = per["delta"].dropna()
    summary = pd.DataFrame([dict(
        n_reps=2000,
        mean_n_matched=round(float(per["n_matched"].mean()), 1),
        median_n_donors=int(per["n_donors"].median()),
        min_n_donors=int(per["n_donors"].min()),
        max_n_donors=int(per["n_donors"].max()),
        mean_rho_size=round(float(per["rho_size"].mean()), 4),
        mean_rho_full=round(float(per["rho_full"].mean()), 4),
        delta_median=float(deltas.median()),
        delta_p2_5=float(deltas.quantile(0.025)),
        delta_p97_5=float(deltas.quantile(0.975)),
        delta_reps=len(deltas))])
    summary.to_csv(out_dir / "prediction_range_matched_summary.csv", index=False)
    print(f"range_matched_prediction: paired delta median "
          f"{summary['delta_median'].iloc[0]:.4f} "
          f"[{summary['delta_p2_5'].iloc[0]:.4f}, "
          f"{summary['delta_p97_5'].iloc[0]:.4f}], "
          f"mean matched n = {summary['mean_n_matched'].iloc[0]}")


def composition_prediction(table, out_dir: Path) -> None:
    """Composition sensitivity: the mouse excitatory-compatible subset
    (all reconstruction types) against the within-species samples, using
    size-only, dendritic-feature (valid across reconstruction types), and
    full-10 models.  Subsets below the minimum-inference rule are flagged
    inadequate rather than forced."""
    ycol = "input_resistance_mohm"
    parsed = table[table["parsed"].fillna(False).astype(bool)]
    subsets = {
        "C1_mouse_excitatory_all_reconstruction_types":
            parsed[(parsed["species"] == "mouse")
                   & (parsed["broad_class"] == "excitatory")],
        "C2_mouse_v3_scope":
            parsed[(parsed["species"] == "mouse")
                   & (parsed["reconstruction_type"] != "dendrite-only")],
        "C2ref_human_v3_scope":
            parsed[(parsed["species"] == "human")
                   & (parsed["reconstruction_type"] != "dendrite-only")],
        "C3_mouse_excitatory_v3_scope":
            parsed[(parsed["species"] == "mouse")
                   & (parsed["broad_class"] == "excitatory")
                   & (parsed["reconstruction_type"] != "dendrite-only")],
    }
    models = {"size_only_dend_len": ["dend_len"],
              "dendritic_7": DENDRITIC_FEATS,
              "full_10": MORPHO_FEATS}
    rows = []
    for subset, sub in subsets.items():
        sub = sub.dropna(subset=["donor_id"])
        adequate = len(sub) >= 20 and sub["donor_id"].nunique() >= 10
        status = "evaluated" if adequate else "inadequate_small_n_descriptive"
        for model, xcols in models.items():
            rho, n_held = lodo_score(sub, xcols, ycol)
            rows.append(dict(subset=subset, model=model, status=status,
                             n_cells_scope=len(sub),
                             n_donors_scope=int(sub["donor_id"].nunique()),
                             lodo_rho=rho, n_held=n_held))
        if adequate:
            for alt in ("full_10", "dendritic_7"):
                median, d_lo, d_hi, reps = paired_partition_deltas(
                    sub, models[alt], models["size_only_dend_len"], ycol)
                rows.append(dict(subset=subset,
                                 model=f"delta_{alt}_vs_size", status=status,
                                 n_cells_scope=len(sub),
                                 n_donors_scope=int(sub["donor_id"].nunique()),
                                 lodo_rho=median, n_held=reps,
                                 delta_lo=d_lo, delta_hi=d_hi))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "prediction_composition.csv", index=False)
    print("composition_prediction:\n" + out.round(4).to_string(index=False))


def main(data_dir: Path, results_dir: Path) -> None:
    del data_dir  # analyses read the prebuilt analysis table only
    table = load_analysis_table(results_dir)
    out_dir = results_dir / "prediction"
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark(table, out_dir)
    compartment_prediction(table, out_dir)
    common_support_prediction(table, out_dir)
    range_matched_prediction(table, out_dir)
    composition_prediction(table, out_dir)
    print(f"prediction outputs written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                        help="directory holding metadata/ and swc/ inputs")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                        help="directory for analysis outputs")
    args = parser.parse_args()
    main(args.data_dir, args.results_dir)
