"""Primary statistical analyses for the morphology-physiology coupling study.

Contents, in execution order:

1. Cell-level Spearman coupling matrices (10 morphometric x 15 physiological
   feature pairs per species) with Benjamini-Hochberg FDR control within each
   species' family of 150 pairs.
2. Donor-aware inference for the same families: delete-one-donor jackknife
   p-values, donor-cluster bootstrap confidence intervals (cells travel with
   their donor), and the minimum-inference rule (p-values and CIs are reported
   only for strata with >= 10 donors and >= 20 cells; smaller strata are
   labelled DESCRIPTIVE).
3. Leave-one-donor-out influence screening for the headline pairs.
4. Donor-level aggregation (per-donor means as the observational unit).
5. Human middle-temporal-gyrus-only coupling family (cortical-area control).
6. Composition-restricted species contrasts (mouse excitatory cells against
   human samples) with donor-cluster bootstrap of the Fisher-z difference.
7. Human donor-stratum coupling (etiology, sex, age, cortical area).
8. Nested restriction ladder for the headline contrast (dendritic length vs
   input resistance): L0 human all -> L1 human MTG-only -> L2 MTG within the
   shared common support; M0 mouse all -> M1 mouse excitatory.
9. Direct donor-aware residual human-mouse contrasts at the shared
   common-support restriction, and the 2,000-sample matched-sample contrast.
10. Class signatures of intrinsic physiology (mouse) and laminar gradients.

This module also hosts the resampling primitives reused by ``sensitivity.py``
and ``prediction.py``: the donor-aware jackknife and cluster bootstrap,
BH-FDR, the common-support bounds, and the no-replacement
distribution-matching engine.

Deterministic randomness: bootstrap seed 42, matching seed 43, paired
prediction partitions seed 999 + repetition.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, norm, spearmanr, t as tdist
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]

EPHYS_FEATURES = [
    "input_resistance_mohm", "tau", "ri", "sag", "adaptation", "avg_isi",
    "f_i_curve_slope", "latency", "upstroke_downstroke_ratio_long_square",
    "threshold_v_long_square", "threshold_i_long_square", "vrest",
    "fast_trough_v_long_square", "peak_v_long_square", "vm_for_sag",
]
MORPHO_FEATS = ["dend_len", "axon_len", "dend_nbranch", "dend_ntips",
                "dend_max_extent", "dend_maxpath", "axon_max_extent",
                "axon_maxpath", "dend_meandiam", "sholl_peak"]
DENDRITIC_FEATS = ["dend_len", "dend_nbranch", "dend_ntips",
                   "dend_max_extent", "dend_maxpath", "dend_meandiam",
                   "sholl_peak"]
HEADLINE_OUTCOMES = ["input_resistance_mohm", "f_i_curve_slope", "latency",
                     "sag"]
HEADLINE_MORPHO = "dend_len"

MIN_CELLS = 20          # minimum-inference rule
MIN_DONORS = 10
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
MATCH_SEED = 43
MIN_PAIR_N = 8          # minimum complete cases for a point estimate


# --------------------------------------------------------------------------
# Resampling and inference primitives (shared with sensitivity / prediction)
# --------------------------------------------------------------------------

def spearman_rho(x, y):
    """Spearman rank correlation via the rank transform.  Inputs must be
    finite and of equal length; NaN inputs yield NaN (all analysis paths
    select complete pairs before calling)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return np.nan
    return float((rx * ry).sum() / denom)


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values; NaN inputs pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    po = p[ok]
    order = np.argsort(po)
    q = po[order] * ok.sum() / np.arange(1, ok.sum() + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    filled = np.full(ok.sum(), np.nan)
    filled[order] = q
    out[ok] = filled
    return out


def complete_pairs(df, morpho_feat, ephys_feat):
    """Complete-case pairs of two columns; returns values and row positions."""
    x = pd.to_numeric(df[morpho_feat], errors="coerce")
    y = pd.to_numeric(df[ephys_feat], errors="coerce")
    ok = (x.notna() & y.notna()).values
    positions = np.arange(len(df))[ok]
    return x.values[ok].astype(float), y.values[ok].astype(float), positions


def is_inferential(n_cells, n_donors):
    return (n_donors >= MIN_DONORS) and (n_cells >= MIN_CELLS)


def jackknife_p(x, y, donor_labels, rho_obs):
    """Delete-one-donor jackknife p-value for a Spearman correlation.

    Pseudo-values are computed by deleting each donor in turn; the standard
    error is sqrt((G-1)/G) * sum((theta_g - theta_bar)^2) and the p-value is a
    two-sided t-test with G-1 degrees of freedom.  Donors whose removal leaves
    fewer than 8 complete pairs contribute NaN pseudo-values.
    """
    uniq = pd.unique(donor_labels)
    if len(uniq) < 2 or not np.isfinite(rho_obs):
        return np.nan, np.nan
    pseudo = np.empty(len(uniq))
    for g, donor in enumerate(uniq):
        keep = donor_labels != donor
        pseudo[g] = spearman_rho(x[keep], y[keep]) if keep.sum() >= MIN_PAIR_N else np.nan
    finite = np.isfinite(pseudo)
    if finite.sum() < 2:
        return np.nan, np.nan
    g = finite.sum()
    theta_bar = np.nanmean(pseudo)
    se = np.sqrt((g - 1) / g * np.nansum((pseudo - theta_bar) ** 2))
    if not np.isfinite(se) or se == 0:
        return np.nan, np.nan
    t = rho_obs / se
    return float(2 * (1 - tdist.cdf(abs(t), df=g - 1))), float(se)


def donor_cluster_bootstrap_draws(x, y, donor_labels, n_boot=N_BOOTSTRAP,
                                  seed=BOOTSTRAP_SEED):
    """Donor-cluster bootstrap draws of the Spearman correlation (donors are
    resampled with replacement; cells travel with their donor)."""
    uniq = pd.unique(donor_labels)
    rows_by_donor = {d: np.where(donor_labels == d)[0] for d in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_donor[d] for d in picked])
        draws[b] = spearman_rho(x[idx], y[idx])
    return draws


def cluster_boot_ci(x, y, donor_labels, rho_obs, n_boot=N_BOOTSTRAP,
                    seed=BOOTSTRAP_SEED):
    """Percentile 95% CI from the donor-cluster bootstrap (NaN unless at
    least 100 draws are finite)."""
    if not np.isfinite(rho_obs) or len(pd.unique(donor_labels)) < 2:
        return np.nan, np.nan
    draws = donor_cluster_bootstrap_draws(x, y, donor_labels, n_boot, seed)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 100:
        return np.nan, np.nan
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def fisher_z(rho):
    return np.arctanh(np.clip(rho, -0.999, 0.999))


def fisher_z_diff_point(rho1, n1, rho2, n2):
    """Point Fisher-z difference between two independent correlations with
    its classical standard error and two-sided p-value."""
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_stat = (fisher_z(rho1) - fisher_z(rho2)) / se
    p = 2 * (1 - norm.cdf(abs(z_stat)))
    return float(z_stat), float(se), float(p)


def donor_boot_fisher_z_diff(x1, y1, d1, x2, y2, d2, n_boot=N_BOOTSTRAP,
                             seed=BOOTSTRAP_SEED):
    """Median and 95% percentile interval of the Fisher-z difference between
    two correlations, from independent donor-cluster bootstraps of each group
    (seeds ``seed`` and ``seed + 1``)."""
    b1 = donor_cluster_bootstrap_draws(x1, y1, d1, n_boot, seed)
    b2 = donor_cluster_bootstrap_draws(x2, y2, d2, n_boot, seed + 1)
    dz = fisher_z(b1) - fisher_z(b2)
    dz = dz[np.isfinite(dz)]
    if len(dz) == 0:
        return np.nan, np.nan, np.nan
    return (float(np.median(dz)), float(np.percentile(dz, 2.5)),
            float(np.percentile(dz, 97.5)))


# ------------------------------ data handling ------------------------------

def load_analysis_table(results_dir: Path) -> pd.DataFrame:
    return pd.read_csv(results_dir / "analysis_table.csv")


def parsed_cells(table):
    return table[table["parsed"].fillna(False).astype(bool)]


def axon_bearing(table, species=None):
    """Primary coupling scope: parsed, axon-bearing reconstructions
    (anything other than a dendrite-only tracing)."""
    sub = parsed_cells(table)
    sub = sub[sub["reconstruction_type"] != "dendrite-only"]
    if species is not None:
        sub = sub[sub["species"] == species]
    return sub


def common_support_bounds(table):
    """Shared common-support interval: the mouse P0.5-P99.5 percentile range
    of total dendritic length (axon-bearing scope)."""
    mouse = axon_bearing(table, "mouse")
    dend = pd.to_numeric(mouse[HEADLINE_MORPHO], errors="coerce").values
    lo, hi = np.quantile(dend[np.isfinite(dend)], [0.005, 0.995])
    return float(lo), float(hi)


# ----------------------- distribution-matching engine ----------------------

def build_strata(mouse_dend_len, n_strata=20):
    """Quantile strata of the mouse dendritic-length distribution: stratum
    medians (draw centres), stratum cell counts, and the window half-width."""
    edges = np.quantile(mouse_dend_len, np.linspace(0, 1, n_strata + 1))
    centers, counts = [], []
    for s in range(n_strata):
        lo = edges[s]
        hi = edges[s + 1] if s < n_strata - 1 else edges[-1] + 1e-9
        sel = (mouse_dend_len >= lo) & (mouse_dend_len < hi) if s < n_strata - 1 \
            else (mouse_dend_len >= lo) & (mouse_dend_len <= hi)
        counts.append(int(sel.sum()))
        centers.append(float(np.median(mouse_dend_len[sel])))
    width = (edges[-1] - edges[0]) / n_strata
    return centers, counts, width


def draw_no_replacement(rng, target_values, centers, counts, width,
                        weighted=False):
    """One no-replacement matched sample from ``target_values``.

    For each stratum, candidates lie within the window (centre +/- width) and
    are still available; the fallback set is any remaining cell.  Each stratum
    draws k = min(stratum count, candidates) cells without replacement, either
    uniformly or with probability proportional to exp(-distance/tau)
    (closeness-weighted variant, tau = max(width, 1) / 2).
    """
    available = np.ones(len(target_values), dtype=bool)
    picks = []
    tau = max(width, 1.0) / 2.0
    for s, center in enumerate(centers):
        window = (target_values >= center - width) & \
                 (target_values <= center + width) & available
        candidates = np.where(window)[0]
        if len(candidates) == 0:
            candidates = np.where(available)[0]
        k = min(counts[s], len(candidates))
        if len(candidates) == 0 or k == 0:
            continue
        if weighted:
            distance = np.abs(target_values[candidates] - center)
            w = np.exp(-distance / tau)
            w = w / w.sum()
            chosen = rng.choice(candidates, size=k, replace=False, p=w)
        else:
            chosen = rng.choice(candidates, size=k, replace=False)
        picks.extend(chosen.tolist())
        available[chosen] = False
    return np.array(picks, dtype=int)


def match_human_only(rng, human_dend_len, centers, counts, width,
                     weighted=False):
    """One matched-human sample (no placebo draw consumed)."""
    return draw_no_replacement(rng, human_dend_len, centers, counts, width,
                               weighted)


def match_human_and_placebo(rng, human_dend_len, mouse_dend_len, centers,
                            counts, width, weighted=False):
    """One matched-human sample followed by the same-size placebo mouse
    subsample, drawn from the shared rng stream in that order."""
    human_idx = draw_no_replacement(rng, human_dend_len, centers, counts,
                                    width, weighted)
    mouse_idx = rng.choice(len(mouse_dend_len), size=len(human_idx),
                           replace=False)
    return human_idx, mouse_idx


# ------------------------- primary coupling families -----------------------

def _cell_level_family(sub, label):
    """Cell-level Spearman family (10 x 15 pairs) with BH-FDR."""
    rows = []
    for mf in MORPHO_FEATS:
        for ef in EPHYS_FEATURES:
            x, y, _ = complete_pairs(sub, mf, ef)
            if len(x) < MIN_PAIR_N:
                rows.append(dict(species=label, morpho=mf, ephys=ef, n=len(x),
                                 rho=np.nan, p=np.nan))
                continue
            rho, p = spearmanr(x, y)
            rows.append(dict(species=label, morpho=mf, ephys=ef, n=len(x),
                             rho=float(rho), p=float(p)))
    family = pd.DataFrame(rows)
    family["q"] = bh_fdr(family["p"].values)
    return family


def _donor_aware_family(sub, label):
    """Donor-aware inference for the 10 x 15 pairs of one scope."""
    donors_arr = sub["donor_id"].values
    rows = []
    for mf in MORPHO_FEATS:
        for ef in EPHYS_FEATURES:
            x, y, positions = complete_pairs(sub, mf, ef)
            n = len(x)
            if n < MIN_PAIR_N:
                rows.append(dict(species=label, morpho=mf, ephys=ef, n=n,
                                 n_donors=0, rho=np.nan, jack_p=np.nan,
                                 jack_se=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                                 cell_p=np.nan, inferential=False))
                continue
            dsub = donors_arr[positions]
            n_donors = len(pd.unique(dsub))
            rho = spearman_rho(x, y)
            inferential = is_inferential(n, n_donors)
            if inferential:
                jack_p, jack_se = jackknife_p(x, y, dsub, rho)
                ci_lo, ci_hi = cluster_boot_ci(x, y, dsub, rho)
                _, cell_p = spearmanr(x, y)
            else:
                jack_p = jack_se = ci_lo = ci_hi = cell_p = np.nan
            rows.append(dict(species=label, morpho=mf, ephys=ef, n=n,
                             n_donors=n_donors, rho=rho, jack_p=jack_p,
                             jack_se=jack_se, ci_lo=ci_lo, ci_hi=ci_hi,
                             cell_p=cell_p, inferential=inferential))
    family = pd.DataFrame(rows)
    inferential = family["inferential"].values
    q_donor = bh_fdr(family["jack_p"].values)
    q_cell = bh_fdr(family["cell_p"].values)
    family["q_donor"] = np.where(inferential, q_donor, np.nan)
    family["q_cell"] = np.where(inferential, q_cell, np.nan)
    return family


def coupling_spearman(table, out_dir: Path) -> None:
    """Cell-level Spearman coupling matrices for the primary (axon-bearing)
    samples, BH-FDR within each species' family of 150 pairs."""
    families = []
    for species in ("mouse", "human"):
        families.append(_cell_level_family(axon_bearing(table, species), species))
    out = pd.concat(families, ignore_index=True)
    out.to_csv(out_dir / "coupling_spearman.csv", index=False)
    sig = int((out["q"] < 0.05).sum())
    print(f"coupling_spearman: {len(out)} pairs, {sig} significant at q < 0.05")


def donor_aware_coupling(table, out_dir: Path) -> None:
    """Donor-aware coupling inference and the cell-level vs donor-aware
    significance survival comparison."""
    families = []
    for species in ("mouse", "human"):
        families.append(_donor_aware_family(axon_bearing(table, species), species))
    out = pd.concat(families, ignore_index=True)
    out.to_csv(out_dir / "donor_aware_coupling.csv", index=False)

    survival = []
    for species in ("mouse", "human"):
        m = (out["species"] == species) & out["inferential"]
        cell_sig = out.loc[m, "q_cell"] < 0.05
        donor_sig = out.loc[m, "q_donor"] < 0.05
        survival.append(dict(
            species=species,
            inferential_pairs=int(m.sum()),
            cell_level_bh_sig=int(cell_sig.sum()),
            donor_aware_sig=int(donor_sig.sum()),
            both=int((cell_sig & donor_sig).sum()),
            lost=int((cell_sig & ~donor_sig).sum()),
            gained=int((~cell_sig & donor_sig).sum())))
    pd.DataFrame(survival).to_csv(out_dir / "donor_aware_survival.csv", index=False)
    print("donor_aware_coupling: survival " +
          ", ".join(f"{r['species']}: cell-level {r['cell_level_bh_sig']}, "
                    f"donor-aware {r['donor_aware_sig']}" for r in survival))


def lodo_influence(table, out_dir: Path) -> None:
    """Leave-one-donor-out influence on the headline pairs (total dendritic
    length vs input resistance and f-I slope)."""
    rows = []
    for species in ("mouse", "human"):
        sub = axon_bearing(table, species)
        donors_arr = sub["donor_id"].values
        for ef in ("input_resistance_mohm", "f_i_curve_slope"):
            x, y, positions = complete_pairs(sub, HEADLINE_MORPHO, ef)
            dsub = donors_arr[positions]
            rho_all = spearman_rho(x, y)
            for donor in pd.unique(dsub):
                keep = dsub != donor
                rho_excl = spearman_rho(x[keep], y[keep])
                rows.append(dict(species=species, ephys=ef, donor_id=donor,
                                 donor_cells=int((dsub == donor).sum()),
                                 rho_all=rho_all, rho_excl=rho_excl,
                                 delta_rho=rho_excl - rho_all,
                                 influence_flag=abs(rho_excl - rho_all) > 0.05))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "lodo_influence.csv", index=False)
    print(f"lodo_influence: {int(out['influence_flag'].sum())} flags of {len(out)} "
          "donor exclusions (|delta rho| > 0.05)")


def donor_means_coupling(table, out_dir: Path) -> None:
    """Donor-level aggregation: per-donor means as the observational unit."""
    rows = []
    for species in ("mouse", "human"):
        sub = axon_bearing(table, species)
        features = [HEADLINE_MORPHO] + HEADLINE_OUTCOMES
        agg = sub.groupby("donor_id")[features].mean().dropna(how="all")
        for ef in HEADLINE_OUTCOMES:
            pair = agg[[HEADLINE_MORPHO, ef]].dropna()
            if len(pair) < MIN_DONORS:
                continue
            rho, p = spearmanr(pair[HEADLINE_MORPHO], pair[ef])
            rows.append(dict(species=species, ephys=ef,
                             n_donors=len(pair), rho=float(rho), p=float(p)))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "donor_level_coupling.csv", index=False)
    print("donor_level_coupling:\n" + out.to_string(index=False))


def mtg_coupling(table, out_dir: Path) -> None:
    """Donor-aware coupling family for the human middle-temporal-gyrus
    sample (cortical-area control)."""
    human = axon_bearing(table, "human")
    mtg = human[human["area_group"] == "MTG"]
    out = _donor_aware_family(mtg, "human_MTG")
    out.to_csv(out_dir / "human_mtg_coupling.csv", index=False)
    headline = out[(out.morpho == HEADLINE_MORPHO)
                   & (out.ephys == "input_resistance_mohm")].iloc[0]
    print(f"human_mtg_coupling: {len(mtg)} cells / {mtg['donor_id'].nunique()} "
          f"donors; headline rho = {headline['rho']:.3f} "
          f"[{headline['ci_lo']:.3f}, {headline['ci_hi']:.3f}]")


def composition_restricted_contrast(table, out_dir: Path) -> None:
    """Species contrasts with the mouse sample restricted to its excitatory
    cells (all reconstruction types) against human all-MTG samples."""
    mouse_exc = parsed_cells(table)
    mouse_exc = mouse_exc[(mouse_exc["species"] == "mouse")
                          & (mouse_exc["broad_class"] == "excitatory")]
    human_all = axon_bearing(table, "human")
    human_mtg = human_all[human_all["area_group"] == "MTG"]
    rows = []
    for ef in EPHYS_FEATURES:
        entry = dict(ephys=ef)
        xm, ym, pm = complete_pairs(mouse_exc, HEADLINE_MORPHO, ef)
        for label, hsub in (("human_all", human_all), ("human_MTG", human_mtg)):
            xh, yh, ph = complete_pairs(hsub, HEADLINE_MORPHO, ef)
            if len(xm) < MIN_PAIR_N or len(xh) < MIN_PAIR_N:
                continue
            rho_m, rho_h = spearman_rho(xm, ym), spearman_rho(xh, yh)
            z_stat, _, p = fisher_z_diff_point(rho_m, len(xm), rho_h, len(xh))
            z_med, z_lo, z_hi = donor_boot_fisher_z_diff(
                xm, ym, mouse_exc["donor_id"].values[pm],
                xh, yh, hsub["donor_id"].values[ph])
            entry.update({
                f"rho_mouse_{label}": rho_m, f"rho_{label}": rho_h,
                f"n_mouse_{label}": len(xm), f"n_{label}": len(xh),
                f"n_donors_mouse_{label}": len(pd.unique(mouse_exc["donor_id"].values[pm])),
                f"n_donors_{label}": len(pd.unique(hsub["donor_id"].values[ph])),
                f"fisher_z_{label}": z_stat, f"p_{label}": p,
                f"zdiff_boot_median_{label}": z_med,
                f"zdiff_boot_lo_{label}": z_lo, f"zdiff_boot_hi_{label}": z_hi})
        if entry:
            rows.append(entry)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "composition_restricted_contrast.csv", index=False)
    print(f"composition_restricted_contrast: mouse excitatory n = {len(mouse_exc)} "
          f"({mouse_exc['donor_id'].nunique()} donors)")


def donor_strata(table, donor_table, out_dir: Path) -> None:
    """Headline coupling across human donor strata (etiology, sex, age,
    cortical area); strata below the minimum-inference cell count are
    omitted, strata below the donor count are labelled DESCRIPTIVE.  The age
    median is taken over all human donors in the release (not only those
    with reconstructed cells)."""
    human = axon_bearing(table, "human").dropna(subset=["donor_id"])
    human = human.dropna(subset=["input_resistance_mohm", HEADLINE_MORPHO])
    age_median = (donor_table[donor_table["species"] == "human"]["age_years"]
                  .median())
    sex_norm = human["sex"].astype(str).str.upper().str.slice(0, 1)

    def strata_frames():
        yield "human_all", human
        yield "epilepsy", human[human["etiology"] == "epilepsy"]
        yield "tumor", human[human["etiology"] == "tumor"]
        yield "male", human[sex_norm == "M"]
        yield "female", human[sex_norm == "F"]
        yield "age_ge_median", human[human["age_years"] >= age_median]
        yield "age_lt_median", human[human["age_years"] < age_median]
        yield "age_ge_40", human[human["age_years"] >= 40]
        yield "age_lt_40", human[human["age_years"] < 40]
        for area in ("MTG", "frontal-other", "temporal-other",
                     "angular/parietal"):
            yield f"area_{area}", human[human["area_group"] == area]

    rows = []
    for name, sub in strata_frames():
        n = len(sub)
        if n < MIN_CELLS:
            continue
        x = pd.to_numeric(sub[HEADLINE_MORPHO], errors="coerce").values
        y = pd.to_numeric(sub["input_resistance_mohm"], errors="coerce").values
        ok = np.isfinite(x) & np.isfinite(y)
        x, y, dsub = x[ok], y[ok], sub["donor_id"].values[ok]
        n_donors = len(pd.unique(dsub))
        rho = spearman_rho(x, y)
        inferential = is_inferential(int(ok.sum()), n_donors)
        jack_p = jackknife_p(x, y, dsub, rho)[0] if inferential else np.nan
        rows.append(dict(stratum=name, n_cells=int(ok.sum()), n_donors=n_donors,
                         rho=rho, jack_p=jack_p,
                         status="inferential" if inferential else "DESCRIPTIVE"))
    mouse = axon_bearing(table, "mouse").dropna(subset=["donor_id"])
    mouse = mouse.dropna(subset=["input_resistance_mohm", HEADLINE_MORPHO])
    rows.append(dict(stratum="mouse_reference", n_cells=len(mouse),
                     n_donors=mouse["donor_id"].nunique(),
                     rho=spearman_rho(mouse[HEADLINE_MORPHO].values,
                                      mouse["input_resistance_mohm"].values),
                     jack_p=np.nan, status="reference"))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "donor_strata.csv", index=False)
    print("donor_strata:\n" + out.round(3).to_string(index=False))


# --------------------- restriction ladder and contrasts --------------------

def _headline_scope(table, species=None, area_group=None, broad_class=None,
                    support=None):
    sub = axon_bearing(table, species)
    if area_group is not None:
        sub = sub[sub["area_group"] == area_group]
    if broad_class is not None:
        sub = sub[sub["broad_class"] == broad_class]
    sub = sub.dropna(subset=[HEADLINE_MORPHO, "input_resistance_mohm",
                             "donor_id"])
    sub = sub[np.isfinite(pd.to_numeric(sub[HEADLINE_MORPHO], errors="coerce"))]
    if support is not None:
        lo, hi = support
        dend = pd.to_numeric(sub[HEADLINE_MORPHO], errors="coerce")
        sub = sub[(dend >= lo) & (dend <= hi)]
    return sub


def _headline_rung(table, name, **filters):
    """Point estimate for one ladder rung; returns the summary row and the
    rung's rows (for the bootstrap of the transition delta)."""
    sub = _headline_scope(table, **filters)
    x = sub[HEADLINE_MORPHO].values.astype(float)
    y = sub["input_resistance_mohm"].values.astype(float)
    donors = sub["donor_id"].values
    row = dict(rung=name, n_cells=len(sub), n_donors=len(pd.unique(donors)),
               rho=spearman_rho(x, y))
    return row, sub


def restriction_ladder(table, out_dir: Path) -> None:
    """Nested restriction ladder for the headline contrast; each rung is
    compared with the previous one by the Fisher-z difference with a
    donor-cluster bootstrap interval (deltas are per-transition and
    non-additive by construction)."""
    support = common_support_bounds(table)
    rungs = [
        _headline_rung(table, "L0_human_all", species="human"),
        _headline_rung(table, "L1_human_MTG", species="human",
                       area_group="MTG"),
        _headline_rung(table, "L2_human_MTG_common_support", species="human",
                       area_group="MTG", support=support),
        _headline_rung(table, "M0_mouse_all", species="mouse"),
        _headline_rung(table, "M1_mouse_excitatory", species="mouse",
                       broad_class="excitatory"),
    ]
    rows = []
    for i, (rung, sub) in enumerate(rungs):
        if i == 0:
            delta = lo = hi = np.nan
        else:
            prev = rungs[i - 1][1]
            delta, lo, hi = donor_boot_fisher_z_diff(
                prev[HEADLINE_MORPHO].values.astype(float),
                prev["input_resistance_mohm"].values.astype(float),
                prev["donor_id"].values,
                sub[HEADLINE_MORPHO].values.astype(float),
                sub["input_resistance_mohm"].values.astype(float),
                sub["donor_id"].values)
        rows.append(dict(rung=rung["rung"], n_cells=rung["n_cells"],
                         n_donors=rung["n_donors"], rho=rung["rho"],
                         delta_from_prev_z=delta, delta_lo=lo, delta_hi=hi))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "restriction_ladder.csv", index=False)
    print("restriction_ladder:\n" + out.round(4).to_string(index=False))


def direct_residual_contrast(table, out_dir: Path) -> None:
    """Direct donor-aware residual human-mouse contrast at the shared
    common-support restriction (both species restricted to the same interval;
    the Fisher-z difference is computed from paired bootstrap indices)."""
    support = common_support_bounds(table)
    mouse = _headline_scope(table, species="mouse", support=support)
    xm = mouse[HEADLINE_MORPHO].values.astype(float)
    ym = mouse["input_resistance_mohm"].values.astype(float)
    dm = mouse["donor_id"].values
    rho_m = spearman_rho(xm, ym)
    bm = donor_cluster_bootstrap_draws(xm, ym, dm, seed=BOOTSTRAP_SEED + 1)
    zm = fisher_z(bm[np.isfinite(bm)])

    variants = {
        "D1_human_MTG_common_support": _headline_scope(
            table, species="human", area_group="MTG", support=support),
        "D2_human_all_common_support": _headline_scope(
            table, species="human", support=support),
    }
    rows = []
    for name, hs in variants.items():
        xh = hs[HEADLINE_MORPHO].values.astype(float)
        yh = hs["input_resistance_mohm"].values.astype(float)
        dh = hs["donor_id"].values
        rho_h = spearman_rho(xh, yh)
        bh = donor_cluster_bootstrap_draws(xh, yh, dh, seed=BOOTSTRAP_SEED)
        zh = fisher_z(bh[np.isfinite(bh)])
        n = min(len(zh), len(zm))
        delta = zh[:n] - zm[:n]
        rows.append(dict(
            variant=name, human_n=len(hs),
            human_donors=int(hs["donor_id"].nunique()),
            human_rho=rho_h,
            human_ci_lo=float(np.nanpercentile(bh, 2.5)),
            human_ci_hi=float(np.nanpercentile(bh, 97.5)),
            mouse_n=len(mouse), mouse_donors=int(mouse["donor_id"].nunique()),
            mouse_rho=rho_m,
            mouse_ci_lo=float(np.nanpercentile(bm, 2.5)),
            mouse_ci_hi=float(np.nanpercentile(bm, 97.5)),
            support_lo_um=support[0], support_hi_um=support[1],
            delta_z_median=float(np.median(delta)),
            delta_z_lo=float(np.percentile(delta, 2.5)),
            delta_z_hi=float(np.percentile(delta, 97.5)),
            raw_rho_diff=rho_h - rho_m, boot_draws=n))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "direct_residual_contrast.csv", index=False)
    print("direct_residual_contrast:\n" +
          out[["variant", "human_n", "human_rho", "mouse_n", "mouse_rho",
               "delta_z_median", "delta_z_lo", "delta_z_hi"]].round(4).to_string(index=False))


def matched_sample_contrast(table, out_dir: Path) -> None:
    """Matched-sample contrast: 2,000 no-replacement range-matched human
    samples against same-size placebo mouse samples, with donor-cluster
    bootstrap uncertainty (100 draws) inside each repetition."""
    mouse = axon_bearing(table, "mouse")
    human = axon_bearing(table, "human")
    mdend = pd.to_numeric(mouse[HEADLINE_MORPHO], errors="coerce").values.astype(float)
    hdend = pd.to_numeric(human[HEADLINE_MORPHO], errors="coerce").values.astype(float)
    mouse, human = mouse[np.isfinite(mdend)], human[np.isfinite(hdend)]
    mdend, hdend = mdend[np.isfinite(mdend)], hdend[np.isfinite(hdend)]
    rin_m = pd.to_numeric(mouse["input_resistance_mohm"],
                          errors="coerce").values.astype(float)
    rin_h = pd.to_numeric(human["input_resistance_mohm"],
                          errors="coerce").values.astype(float)
    mdon, hdon = mouse["donor_id"].values, human["donor_id"].values

    centers, counts, width = build_strata(mdend)
    rows = []
    rng = np.random.default_rng(MATCH_SEED)
    for rep in range(2000):
        hidx, msel = match_human_and_placebo(rng, hdend, mdend, centers,
                                             counts, width)
        rho_h = spearman_rho(hdend[hidx], rin_h[hidx])
        rho_m = spearman_rho(mdend[msel], rin_m[msel])
        hd, md = hdon[hidx], mdon[msel]
        h_by = {d: np.where(hd == d)[0] for d in pd.unique(hd)}
        m_by = {d: np.where(md == d)[0] for d in pd.unique(md)}
        brng = np.random.default_rng(1_000_000 + rep)
        zb_h = np.empty(100)
        zb_m = np.empty(100)
        for b in range(100):
            hsel = brng.choice(list(h_by), size=len(h_by), replace=True)
            mix = np.concatenate([h_by[d] for d in hsel])
            mselb = brng.choice(list(m_by), size=len(m_by), replace=True)
            mix_m = np.concatenate([m_by[d] for d in mselb])
            zb_h[b] = fisher_z(spearman_rho(hdend[hidx][mix], rin_h[hidx][mix]))
            zb_m[b] = fisher_z(spearman_rho(mdend[msel][mix_m], rin_m[msel][mix_m]))
        zdiff_boot = zb_h - zb_m
        rows.append(dict(rep=rep, n_matched=len(hidx),
                         n_donors=len(h_by), rho_human_matched=rho_h,
                         rho_mouse_placebo=rho_m,
                         zdiff_design=fisher_z(rho_h) - fisher_z(rho_m),
                         zdiff_boot_median=float(np.median(zdiff_boot)),
                         zdiff_boot_lo=float(np.percentile(zdiff_boot, 2.5)),
                         zdiff_boot_hi=float(np.percentile(zdiff_boot, 97.5))))
    per = pd.DataFrame(rows)
    per.to_csv(out_dir / "matched_contrast_reps.csv", index=False)

    design = per["zdiff_design"].dropna()
    summary = pd.DataFrame([dict(
        n_reps=2000, inner_boot=100,
        mean_n_matched=round(float(per["n_matched"].mean()), 1),
        median_n_donors=int(per["n_donors"].median()),
        design_median_zdiff=float(design.median()),
        design_p2_5=float(design.quantile(0.025)),
        design_p97_5=float(design.quantile(0.975)),
        donoraware_median_zdiff=float(np.nanmedian(per["zdiff_boot_median"])),
        donoraware_p2_5=float(np.nanpercentile(per["zdiff_boot_lo"], 2.5)),
        donoraware_p97_5=float(np.nanpercentile(per["zdiff_boot_hi"], 97.5)))])
    summary.to_csv(out_dir / "matched_contrast_summary.csv", index=False)
    print(f"matched_sample_contrast: design median z-diff "
          f"{summary['design_median_zdiff'].iloc[0]:.3f} "
          f"[{summary['design_p2_5'].iloc[0]:.3f}, "
          f"{summary['design_p97_5'].iloc[0]:.3f}], "
          f"mean matched n = {summary['mean_n_matched'].iloc[0]}")


# --------------------- class signatures and laminar gradients --------------

def _epsilon_squared(H, n, k):
    return (H - k + 1) / (n - k)


def _rank_biserial(x, y):
    """Cliff's delta between two samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.nan
    greater = np.greater.outer(x, y).sum()
    less = np.less.outer(x, y).sum()
    return (greater - less) / (x.size * y.size)


def class_signatures(ephys, out_dir: Path) -> None:
    """Class signatures of intrinsic physiology in the QC-passed mouse
    electrophysiology sample (the full released ephys cohort, independent of
    reconstruction): Kruskal-Wallis across broad classes with epsilon-squared
    effect sizes, layer gradients, pairwise rank-biserial effects, and
    medians."""
    mouse = ephys[(ephys["species"] == "mouse")
                  & ephys["qc_ephys_ok"].fillna(False).astype(bool)]
    rows, pairwise, medians = [], [], []
    canon = ["input_resistance_mohm", "adaptation",
             "upstroke_downstroke_ratio_long_square", "tau", "avg_isi", "sag"]
    for feat in EPHYS_FEATURES:
        d = mouse.dropna(subset=[feat])
        groups = [g[feat].values for _, g in d.groupby("broad_class") if len(g) >= 5]
        labels = [k for k, g in d.groupby("broad_class") if len(g) >= 5]
        if len(groups) >= 3:
            H, p = kruskal(*groups)
            n, k = sum(len(g) for g in groups), len(groups)
            eps2 = _epsilon_squared(H, n, k)
        else:
            H, p, eps2, n, k = np.nan, np.nan, np.nan, len(d), len(groups)
        d_layer = d.dropna(subset=["layer"])
        if len(d_layer) > 30 and d_layer["layer"].nunique() >= 3:
            rho_layer, p_layer = spearmanr(d_layer["layer"], d_layer[feat])
        else:
            rho_layer, p_layer = np.nan, np.nan
        rows.append(dict(feature=feat, n_total=len(d), n_groups=k, H=H,
                         p_kruskal=p, epsilon_sq=eps2,
                         layer_spearman=rho_layer, p_layer=p_layer))
        if feat in canon:
            for a, b in itertools.combinations(sorted(d["broad_class"].unique()), 2):
                xa = d.loc[d["broad_class"] == a, feat]
                xb = d.loc[d["broad_class"] == b, feat]
                if len(xa) >= 5 and len(xb) >= 5:
                    pairwise.append(dict(feature=feat, class_a=a, n_a=len(xa),
                                         median_a=float(xa.median()),
                                         class_b=b, n_b=len(xb),
                                         median_b=float(xb.median()),
                                         rank_biserial=_rank_biserial(xa.values,
                                                                      xb.values)))
        medians.append(dict(feature=feat, **d.groupby("broad_class")[feat]
                            .median().to_dict()))
    fam = pd.DataFrame(rows)
    fam["q_kruskal"] = bh_fdr(fam["p_kruskal"].values)
    fam["q_layer"] = bh_fdr(fam["p_layer"].values)
    fam.to_csv(out_dir / "class_signatures.csv", index=False)
    pd.DataFrame(pairwise).to_csv(out_dir / "class_pairwise_effects.csv", index=False)
    pd.DataFrame(medians).to_csv(out_dir / "class_medians.csv", index=False)
    print(f"class_signatures: {int((fam['q_kruskal'] < 0.05).sum())} of "
          f"{len(fam)} features separate the mouse classes at q < 0.05")


def laminar_gradients(table, out_dir: Path) -> None:
    """Laminar gradients of dendritic morphology within the mouse sample,
    computed within class as well as across all reconstructed cells (layer
    and class are entangled in the released pool)."""
    mouse = parsed_cells(table)
    mouse = mouse[mouse["species"] == "mouse"]
    scopes = {
        "mouse_all_reconstructed": mouse,
        "mouse_excitatory_all_reconstruction_types":
            mouse[mouse["broad_class"] == "excitatory"],
        "mouse_pvalb_axon_bearing":
            mouse[(mouse["broad_class"] == "Pvalb")
                  & (mouse["reconstruction_type"] != "dendrite-only")],
    }
    features = ["dend_len", "dend_max_extent", "dend_nbranch", "sholl_peak"]
    rows = []
    for scope, sub in scopes.items():
        d = sub.dropna(subset=["layer"])
        for mf in features:
            x = pd.to_numeric(d[mf], errors="coerce")
            ok = x.notna()
            if ok.sum() > 15 and d["layer"].nunique() >= 3:
                rho, p = spearmanr(d.loc[ok, "layer"], x[ok])
                rows.append(dict(scope=scope, morpho=mf, n=int(ok.sum()),
                                 rho=float(rho), p=float(p)))
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["p"].values)
    out.to_csv(out_dir / "laminar_gradients.csv", index=False)
    print(f"laminar_gradients: {len(out)} scope x feature rows")


def main(data_dir: Path, results_dir: Path) -> None:
    del data_dir  # analyses read the prebuilt analysis table only
    table = load_analysis_table(results_dir)
    out_dir = results_dir / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    coupling_spearman(table, out_dir)
    donor_aware_coupling(table, out_dir)
    lodo_influence(table, out_dir)
    donor_means_coupling(table, out_dir)
    mtg_coupling(table, out_dir)
    composition_restricted_contrast(table, out_dir)
    donors = pd.read_csv(results_dir / "cohort" / "human_donors.csv")
    donor_strata(table, donors, out_dir)
    restriction_ladder(table, out_dir)
    direct_residual_contrast(table, out_dir)
    matched_sample_contrast(table, out_dir)
    ephys = pd.read_csv(results_dir / "cohort" / "ephys_cells.csv")
    class_signatures(ephys, out_dir)
    laminar_gradients(table, out_dir)
    print(f"statistics outputs written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                        help="directory holding metadata/ and swc/ inputs")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                        help="directory for analysis outputs")
    args = parser.parse_args()
    main(args.data_dir, args.results_dir)
