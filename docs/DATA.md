# Data

This repository contains **analysis code only**. No Allen Cell Types Database
data are included or redistributed: the source data remain distributed by the
Allen Institute for Brain Science, and users should retrieve the public source
data themselves, following the pointers below.

## Source

All data come from the [Allen Cell Types Database](https://celltypes.brain-map.org/)
(Allen Institute for Brain Science), accessed programmatically through the
public [Allen Brain Map API](https://api.brain-map.org/) without credentials,
license agreements, or restricted data.

The study snapshot is the API release state at **access date 2026-09-17**;
that release state defines the analysis cohort. Re-freezing the snapshot at a
later date may yield slightly different row counts if the release has changed.

## Required inputs

Four metadata tables and the SWC reconstruction files:

| File | Content | API source |
|---|---|---|
| `ephys_features.csv` | one row per specimen; per-cell intrinsic-physiology features | `model::EphysFeature` (all rows) |
| `neuron_reconstructions.csv` | one row per released tracing; reconstruction type, released morphometric summaries | `model::NeuronReconstruction` (all rows) |
| `specimens.csv` | one row per specimen; specimen name, structure, and donor fields | `model::Specimen` with `structure` and `donor` includes |
| `age_lookup.csv` | Age lookup table (donor ages are recorded in days) | `model::Age` (all rows) |
| `swc/*.swc` | one SWC file per released tracing | `model::WellKnownFile` links attached to each reconstruction |

The study used 2,336 electrophysiology rows (1,923 mouse / 413 human by donor
organism; 2,335 passing the pre-specified QC rule), 701 released
reconstructions (668 unique specimens; 33 specimens carry two tracings each;
398 axon-bearing rows covering 368 unique specimens), and 42 human donors.
All 701 SWC files were downloaded and parsed successfully. See Table 1 of the
manuscript for the full cohort composition.

### Column requirements

`build_cohort.py` reads the following columns (extra columns are ignored):

- `ephys_features.csv`: `specimen_id`, and the released feature fields
  `input_resistance_mohm`, `tau`, `ri`, `sag`, `adaptation`, `avg_isi`,
  `f_i_curve_slope`, `latency`, `upstroke_downstroke_ratio_long_square`,
  `threshold_v_long_square`, `threshold_i_long_square`, `vrest`,
  `fast_trough_v_long_square`, `peak_v_long_square`, `vm_for_sag`,
  `seal_gohm`.
- `neuron_reconstructions.csv`: `id` (reconstruction id), `specimen_id`,
  `neuron_reconstruction_type` (`full` / `dendrite-only`), `superseded`
  (marks the superseded tracing of doubly-traced specimens), and the released
  summary fields — `total_length`, `number_branches`, `number_nodes`,
  `max_euclidean_distance` (parsing-consistency validation) and
  `average_bifurcation_angle_remote`, `average_contraction`,
  `average_diameter`, `average_parent_daughter_ratio`, `max_branch_order`,
  `number_stems`, `total_surface`, `total_volume`, `overall_depth`,
  `overall_height`, `hausdorff_dimension` (redundancy diagnostics).
- `specimens.csv`: flattened specimen/donor fields `specimen_id`,
  `specimen_name`, `structure_acronym`, `structure_name`, `hemisphere`,
  `donor_id`, `donor_label` (donor external name), `donor_organism_id`
  (1 = human, 2 = mouse), `donor_sex`, `donor_condition` (condition
  description), `donor_tissue_source`, `donor_full_genotype`, `age_id`.
- `age_lookup.csv`: `age_id`, `age_name`, `age_days`.

Human donor metadata are limited to the public release fields (sex, age via
the Age lookup, surgical etiology from the condition description, tissue
source). No additional scraping and no restricted or participant-identifying
data are used.

### API query patterns

Metadata tables are fetched with JSON queries of the form
`https://api.brain-map.org/api/v2/data/query.json?criteria=<criteria>`:

```
model::EphysFeature,rma::options[num_rows$eqall]
model::NeuronReconstruction,rma::options[num_rows$eqall]
model::Specimen,rma::criteria,[id$in<id1>,<id2>,...],rma::include,structure,donor,rma::options[num_rows$eqall]
model::Age,rma::options[num_rows$eqall]
```

Specimen queries are batched (the API caps responses at 50 rows unless
`num_rows$eqall` is given). Each reconstruction's SWC file is resolved through
its attached `WellKnownFile` record (`attachable_id` = reconstruction id) and
downloaded from the `download_link` it provides; name the file
`specimen_<specimen_id>_recon_<reconstruction_id>.swc`.

This repository deliberately ships **no download script**; the queries above
are a few minutes of straightforward work and keep the analysis code free of
network logic.

## Expected local layout

Create the following structure under the repository root (or pass
`--data-dir` to each script):

```text
data/
├── metadata/
│   ├── ephys_features.csv
│   ├── neuron_reconstructions.csv
│   ├── specimens.csv
│   └── age_lookup.csv
└── swc/
    ├── specimen_313861608_recon_491119570.swc
    └── ...  (one file per released reconstruction)
```

Analysis outputs are written to `results/` (created on first run); both
directories are git-ignored.

## Terms and attribution

The Allen Cell Types Database is a public resource of the Allen Institute for
Brain Science; use of the data and attribution requirements are governed by
the Allen Institute's own data terms. This repository redistributes none of
the data; users of the analysis code should cite the database and the relevant
primary publications, and must not attempt to re-identify human donors. The
MIT license of this repository covers the code only.
