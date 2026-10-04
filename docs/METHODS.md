# Methods

A concise technical companion to the analysis code. The accompanying
manuscript is the definitive description; this document explains what each
module computes and the conventions used.

## Cohort construction (`build_cohort.py`)

- **Species**: donor organism id (1 = human, 2 = mouse), with a specimen-name
  fallback (names beginning `H<digit>` classified as human). The fallback was
  verified to introduce no misassignment in the analysis snapshot.
- **Broad classes (mouse)**: cre-line keys detected in the specimen name —
  Pvalb; Sst; Vip-like (Vip, Htr3a, Ndnf); excitatory (Cux2, Emx1, Nr5a1,
  Rorb, Scnn1a, Rbp4, Tlx3, Tle4, Slc17a7, Ntsri1, Glt25d2); everything else
  wild-type/other. Human cells form a single stratum dominated by middle
  temporal gyrus.
- **Cortical layer**: parsed from mouse visual-cortex structure acronyms
  (`VISp1`–`VISp6a/b`). Human structures carry no layer assignment.
- **Cortical area (human)**: structure-name grouping into MTG, frontal-other,
  temporal-other, angular/parietal, other; mouse cells are V1.
- **Electrophysiology QC (pre-specified)**: finite primary features with
  input resistance > 0, membrane time constant > 0, resting potential between
  -90 and -40 mV, and non-negative seal. One mouse cell is excluded.
- **Human donor table**: public release fields only — sex, age in years (Age
  lookup days / 365.25), surgical etiology from the condition description
  (epilepsy / tumor / other), tissue source.

Primary coupling samples: parsed axon-bearing reconstructions ("full"
tracings, i.e. anything other than `dendrite-only`) — 236 mouse cells
(163 donors) and 162 human cells (35 donors). The released reconstruction
table has 701 rows for 668 unique specimens; 33 specimens carry two tracings,
and one released tracing lacks a type annotation and is retained by the
axon-bearing filter.

## Morphometric re-derivation (`morphology.py`)

SWC files are parsed with explicit node-id → row-index parent resolution and
coordinates are recentered on the soma centroid. Compartments follow the SWC
structural-type codes: 1 = soma, 2 = axon, 3 = basal dendrite, 4 = apical
dendrite.

Per compartment (axon; dendrites combined):

- **length**: sum of parent-child segment Euclidean norms over the
  compartment's nodes;
- **branch points**: nodes with ≥ 2 same-compartment children;
- **tips**: nodes with no same-compartment child;
- **max radial extent**: largest distance from the soma centroid;
- **max path distance**: largest accumulated parent-child path length from
  the soma (accumulated over nodes ordered by radial distance);
- **mean branch diameter**: mean node radius.

**Apical/basal split**: compartment lengths sum segments whose parent and
child both carry the compartment's type code, so `apical_len + basal_len`
approximates `dend_len` up to the soma-to-primary-dendrite stub. An apical
length of zero denotes a cell without a reconstructed apical dendrite.

**Completeness proxies**: T1 = fraction of apical tips within the outermost
10% of apical radial extent; T2 = the same over all dendritic tips; T3 =
severed-apical-trunk flag (fewer than two apical tips). No truncation flags
exist in the release, so these are conservative screens, not certificates.

**3-D Sholl profiles**: counts of parent-child segments crossing spheres of
radius 10–600 µm in 10 µm steps, computed both over all parent-child segments
(the `sholl_*` feature block used downstream) and over dendritic segments
only (`sholl_dend_*`, the group-profile view). Summaries: peak count, peak
radius, area under the profile.

**Doubly-traced specimens**: compartment metrics and completeness proxies are
broadcast per specimen from the tracing with the denser apical reconstruction
(most apical tips, then basal tips), so all rows of a specimen carry one
consistent compartment description.

**Parsing-consistency validation**: re-derived dendritic length, branch
count, node count, and max radial extent are compared with the released
summaries by Spearman rank correlation (rho = 0.9999, 0.995, 1.000, 1.000;
n = 701). This demonstrates that the parser reproduces the release's
computation; it is not an independent biological validation.

## Analysis table (`build_analysis_table.py`)

Joins the per-tracing morphometrics with per-specimen electrophysiology,
released summary fields, and donor metadata into a single analysis table (one
row per released tracing).  Cre line and released summary fields are taken
from each specimen's canonical released row (the non-superseded tracing where
a specimen carries two).  All downstream analyses apply their own
deterministic scope filters and complete-case handling; there is no
imputation.

## Primary statistics (`statistics.py`)

**Estimand**: Spearman rank correlation per morphometric × physiological
feature pair per species (10 × 15 = 150 pairs per species; listwise deletion
per test).

**Donor-aware inference** (the donor is the sampling unit):

- *Delete-one-donor jackknife*: pseudo-values over donors (deletions leaving
  fewer than 8 complete pairs contribute NaN); standard error
  `sqrt((G-1)/G) · Σ(θ_g − θ̄)²`; two-sided t-test with G−1 degrees of
  freedom — the primary p-value.
- *Donor-cluster bootstrap*: 2,000 resamples of donors with replacement
  (seed 42); cells travel with their donor; percentile 95% CIs.
- *Multiple testing*: Benjamini-Hochberg FDR within each species' family of
  150 jackknife p-values (cell-level BH p-values retained alongside).
- *Minimum-inference rule*: p-values and CIs reported only for strata with
  ≥ 10 donors and ≥ 20 cells; smaller strata are labelled DESCRIPTIVE.

**Headline-pair diagnostics**: leave-one-donor-out influence screening (a
flag at |Δρ| > 0.05) and donor-level aggregation (per-donor means as the
observational unit).

**Restriction designs for the species contrast** (headline pair: total
dendritic length vs input resistance):

- *Nested ladder*: L0 human all → L1 human MTG-only → L2 MTG within the
  shared common support; M0 mouse all → M1 mouse excitatory. Each rung is
  compared with the previous one by the Fisher-z difference with a
  donor-cluster bootstrap 95% interval (2,000 draws; seeds 42/43). Deltas are
  per-transition and non-additive by construction; the ladder supports only
  the claim that restriction changes the human estimate.
- *Direct residual contrasts*: both species restricted to the same
  common-support interval (mouse P0.5–P99.5 of dendritic length, intersected
  with human support); the Fisher-z difference is taken over paired
  bootstrap indices with donors resampled independently within species.
- *Matched-sample contrast*: 2,000 no-replacement range-matched human
  samples (20 quantile strata of the mouse dendritic-length distribution;
  seed 43) against same-size placebo mouse subsamples; donor-aware
  uncertainty from a 100-draw donor-cluster bootstrap inside each repetition.

**Class signatures and laminar gradients** (mouse): Kruskal-Wallis across
broad classes per physiological feature with epsilon-squared effect sizes and
BH-FDR; pairwise rank-biserial (Cliff's delta) effect sizes; laminar
gradients by Spearman correlation with layer order, computed within class
because layer and class are entangled in the released pool.

## Sensitivity analyses (`sensitivity.py`)

- Dendrite-only inclusion (all reconstruction types) for the coupling family.
- Reconstruction-completeness screens (T1/T2 top-decile exclusions, T3 keep)
  for the headline coupling.
- Common-support coupling estimates for both species.
- Range-matched coupling under both matching variants (closeness-weighted,
  uniform-stratified) with placebo mouse subsamples.
- Morphometric redundancy (within-block Spearman structure including released
  summary fields) and effective dimensionality (principal components to 95%
  variance).
- First canonical correlation between the morphometric and physiological
  blocks on complete cases (descriptive).

## Prediction benchmark (`prediction.py`)

Ridge regression (alpha = 1) predicting six canonical electrophysiological
features (input resistance, f-I curve slope, latency, adaptation, sag, tau)
from morphometric features, standardized within each training fold, evaluated
by leave-one-donor-out cross-validation; performance is the Spearman
correlation between predicted and observed values over all held-out cells.
Models: size-only (total dendritic length), a three-feature subset, and the
full ten-feature block. The size-beyond-size delta (full minus size-only) is
computed over 100 shared random donor partitions (seed 999 + repetition;
identical folds for both models), summarized by median and 2.5–97.5
percentile interval.

Comparability stress tests reuse the identical protocol (no re-tuning, no
model selection): compartment prediction (total/apical/basal/apical+basal/
full-10 models), common-support prediction, range-matched prediction
(2,000 matched human samples with donor-level folds inside each sample), and
composition sensitivity (mouse excitatory-compatible subsets; subsets below
the minimum-inference rule are flagged inadequate rather than forced).

All prediction benchmarks describe the released samples; they are not
calibrated population statements.

## Software and determinism

Python 3.12 with NumPy, SciPy, pandas, and scikit-learn only. All resampling
uses fixed seeds: bootstrap 2,000 draws @ seed 42; range matching @ seed 43;
paired prediction partitions @ seed 999 + repetition (100 partitions).
Resampling primitives are shared by the analysis modules so that every
analysis uses the identical estimator implementations.
