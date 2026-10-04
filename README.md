# Coupling between dendritic arborization and intrinsic membrane physiology in the released mouse and human reconstructions of the Allen Cell Types Database

Reproducible Python code for a secondary analysis of publicly released mouse
and human cortical neuron morphology and intrinsic electrophysiology from the
Allen Cell Types Database.

## Overview

The Allen Cell Types Database released standardized whole-cell recordings
together with three-dimensional neuron reconstructions for 2,336 mouse and
human cortical specimens, making a whole-release, cross-modal analysis
possible without new tissue. This project re-derives arborization
morphometrics for all 701 released SWC reconstructions — adding axon-level
geometry, per-cell 3-D Sholl profiles, and the apical/basal compartment split
that the released summary tables omit — and links morphology to intrinsic
physiology cell-by-cell. All inference treats the donor, not the cell, as the
sampling unit (delete-one-donor jackknife, donor-cluster bootstrap). The
analyses quantify morphology-physiology coupling in both species, decompose
how much of the species contrast reflects cortical area, sample composition,
and morphological range, and stress-test what morphology predicts under
comparable sampling distributions.

## Main analyses

- Cohort construction from the released metadata tables (species, class,
  layer, area, electrophysiology QC, human donor metadata from public
  release fields)
- SWC morphometric re-derivation with parsing-consistency validation against
  the released summaries
- 3-D Sholl analysis and an apical/basal compartment split from SWC type codes
- Morphology-physiology coupling matrices with donor-aware inference and
  BH-FDR control (150 feature pairs per species)
- Class signatures, laminar gradients, and cortical-area controls
- Restriction ladder, direct residual contrasts, and matched-sample designs
  for the species coupling contrast
- Reconstruction-completeness, redundancy, range-matching, and
  common-support sensitivity analyses
- Donor-grouped (leave-one-donor-out) ridge prediction benchmark with
  compartment, common-support, range-matched, and composition stress tests

## Repository structure

```text
allen-morphophysiology/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── code/
│   ├── build_cohort.py          # metadata tables -> cohort annotations, QC, donor table
│   ├── morphology.py            # SWC parsing, morphometrics, Sholl, completeness proxies
│   ├── build_analysis_table.py  # morphology x electrophysiology x donor join
│   ├── statistics.py            # donor-aware coupling, contrasts, ladder, class/laminar analyses
│   ├── sensitivity.py           # completeness, range-matching, redundancy, CCA
│   └── prediction.py            # donor-grouped ridge benchmark and stress tests
└── docs/
    ├── DATA.md                  # how to obtain and arrange the source data
    └── METHODS.md               # concise methods companion
```

## Data

Source data are **not included**. The pipeline reads the public Allen Cell
Types Database metadata tables and SWC reconstruction files that you retrieve
yourself from the Allen Brain Map API; `docs/DATA.md` documents the exact
components, API query patterns, expected column names, and the local `data/`
layout the scripts expect. The snapshot used in the manuscript was accessed
2026-09-17.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.12 was used for the published analyses; the code needs only NumPy,
SciPy, pandas, and scikit-learn.

## Reproducing the analyses

Run the modules in order from the repository root (each also accepts
`--data-dir` / `--results-dir`):

```bash
python code/build_cohort.py          # cohort tables from the metadata export
python code/morphology.py            # SWC morphometrics (all 701 reconstructions)
python code/build_analysis_table.py  # analysis-ready join
python code/statistics.py            # primary coupling, contrasts, ladder
python code/sensitivity.py           # robustness and sensitivity analyses
python code/prediction.py            # prediction benchmark and stress tests
```

`code/statistics.py`, `code/sensitivity.py`, and `code/prediction.py` share
their resampling primitives (donor-aware jackknife and cluster bootstrap,
BH-FDR, distribution matching) through `code/statistics.py`. All resampling
is seeded (bootstrap 42, matching 43, prediction partitions 999 + repetition),
so runs are deterministic. Outputs are written under `results/`.

## Citation

See `CITATION.cff`. The v1.0.0 release is archived on Zenodo:
DOI 10.5281/zenodo.23135429. Please cite both the repository archive and the
accompanying manuscript.

## License

MIT — covers the code only. The Allen Cell Types Database source data are not
redistributed here and remain governed by the Allen Institute's own terms
(see `docs/DATA.md`).

## Paper

This repository accompanies the manuscript "Coupling between dendritic
arborization and intrinsic membrane physiology in the released mouse and
human reconstructions of the Allen Cell Types Database."
