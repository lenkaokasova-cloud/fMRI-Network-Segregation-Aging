# Preprocessing notes

This document records the preprocessing demonstration for the dissertation project.

The aim is to preprocess a small subset of OpenNeuro ds005752 using fMRIPrep, inspect the fMRIPrep HTML reports, and extract motion quality-control information.

## Recommended first-pass subject

Use a single participant with MRI data only for the first demonstration. A good default is `sub-ON01016`, which is marked with `MRI=1` in `participants.tsv`.

The earlier draft script used `sub-ON52083`, but that participant is mentioned in the dataset `CHANGES` file because incorrect ASL data were removed in version `2.1.0` on `2025-02-18`. That does not automatically invalidate T1w/BOLD preprocessing, but it is cleaner to start with another participant for the demo.

## Minimal workflow

1. Create the environment:

```bash
conda env create -f environment.yml
conda activate fmri-aging
```

2. Download one MRI subset into this repository:

```bash
bash code/01_download_openneuro_subset.sh sub-ON01016
```

This downloads:
- top-level BIDS metadata
- `anat/`
- `func/`
- `fmap/`

3. Run fMRIPrep with Docker:

```bash
export FS_LICENSE=$HOME/license.txt
bash code/02_run_fmriprep_subset.sh sub-ON01016
```

## Expected outputs

After a successful run, inspect:

- `data/derivatives/fmriprep/sub-ON01016.html`
- `data/derivatives/fmriprep/sub-ON01016/`
- `data/derivatives/fmriprep/logs/`

Key files for a dissertation preprocessing demonstration:

- the subject HTML report
- preprocessed BOLD images in MNI and T1w space
- brain mask outputs
- confounds tables such as `*desc-confounds_timeseries.tsv`

## What to check in the HTML report

For a basic preprocessing proof-of-work, confirm that:

- the anatomical skull-stripping looks sensible
- functional-to-anatomical alignment looks reasonable
- susceptibility distortion correction did not obviously fail
- there are no fatal errors or major warnings

## Next step after preprocessing

Once one participant runs successfully, the next logical step is to extract simple QC metrics from the confounds tables, especially:

- framewise displacement
- DVARS
- number of high-motion volumes

That gives you a concrete bridge from "I can preprocess the data" to "I can assess data quality before network segregation analysis."

Run the QC summary script with:

```bash
python code/03_qc_from_confounds.py --subject sub-ON01016
```

This writes a concise table to:

- `data/processed/qc/sub-ON01016_qc_summary.tsv`
