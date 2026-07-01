# NIMH Age-Analysis Workflow

This project is now limited to one scientifically consistent workflow for OpenNeuro `ds005752`:

1. identify `MRI=1` participants
2. split them into the younger and older age bands of interest
3. keep only subjects with remote raw `anat` plus forward resting-state `func`
4. download only those candidates
5. preprocess all retained subjects with the same `fMRIPrep` settings
6. apply explicit QC thresholds
7. compute network segregation using the same atlas and confound strategy for everyone
8. test age-group effects while controlling for sex and motion

## Scientific Design

### Dataset

- OpenNeuro `ds005752`
- healthy NIMH research volunteers
- raw BIDS dataset requiring preprocessing before connectivity analysis

### Default age bands

- younger: `20-25`
- older: `50-75`

You can change these with script arguments, but the repository defaults are now tuned to this comparison.

### Raw data requirements

A participant is considered a raw-data candidate only if remote metadata show:

- structural MRI (`anat`)
- forward resting-state BOLD (`task-rest_dir-forward`)

This is stricter than simply checking `MRI=1`, and it avoids downloading subjects who cannot enter the current connectivity pipeline.

### Younger preprocessing-selection rule

The younger pool is larger than the number of subjects needed for preprocessing, so the repository now uses a fixed pre-processing selection rule before download.

Default younger-selection targets:

- primary younger preprocessing set: `25` subjects
- backup younger set: `10` subjects
- age targets: `20:1`, `21:4`, `22:7`, `23:6`, `24:4`, `25:3`
- sex targets for the primary set: `17 female`, `8 male`
- release targets for the primary set: `17 release_1`, `8 release_2`

Selection principles:

- keep already downloaded or already preprocessed younger subjects when they fit the balancing targets
- preserve spread across the full `20-25` age band instead of taking mostly ages `22-23`
- keep the younger set closer to the expected older sample structure rather than forcing an artificial `50/50` sex split
- choose backups in advance so QC replacements are not decided post hoc

This selection step is only for deciding whom to preprocess first. Final inclusion still depends on the same automated and manual QC rules for both age groups.

### Preprocessing requirement

All final analysis subjects should be preprocessed in the same way with `fMRIPrep`. Do not mix preprocessing pipelines across subjects.

### QC defaults

The clean-sample builder now defaults to:

- minimum forward-run volumes: `180`
- minimum retained volumes after censoring: `150`
- maximum mean FD: `0.25 mm`
- maximum percent of volumes with `FD > 0.2 mm`: `25%`
- required manual `fMRIPrep` HTML report review: `pass`

These defaults are more dissertation-defensible for an aging resting-state study because they combine subject-level motion exclusion, within-subject frame censoring, and visual review of preprocessing quality. Mean FD is still carried forward as an analysis covariate.

### Connectivity analysis choices

- atlas: Schaefer 200 parcels
- network labels: Yeo 7
- filtering: `0.008-0.09 Hz`
- confounds: motion parameters, motion derivatives, white matter, CSF, first six `aCompCor` components
- frame censoring: remove nonsteady-state volumes and volumes with `FD > 0.5 mm`
- run used: forward resting-state run only

This means the final analysis no longer relies on confound regression alone. It now combines nuisance regression with frame censoring, which is easier to justify in the motion-control literature.

### Main statistical models

The final analysis script runs:

1. a global segregation model
   `global_segregation ~ age_group + sex + mean_fd`

2. a higher-order vs sensory-motor interaction model
   `segregation ~ age_group * network_type + sex + mean_fd`

3. a network-specific model
   `segregation ~ age_group * network + sex + mean_fd`

It also writes a continuous-age sensitivity model for global segregation, but the primary workflow is the younger-vs-older comparison because the repo is now organized around explicit age bands.

## Step-by-Step Use

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate fmri-aging
```

### 2. Filter to `MRI=1`

```bash
python code/01_filter_mri_participants.py
```

Output:

- `data/processed/screening/ds005752_mri_participants.tsv`

### 3. Split into younger and older age groups

```bash
python code/02_split_age_groups.py
```

Default outputs:

- `data/processed/screening/ds005752_mri_participants_age_20_25.tsv`
- `data/processed/screening/ds005752_mri_participants_age_50_75.tsv`

### 4. Screen remote raw-file availability

```bash
python code/03_screen_remote_anat_forward_rest.py
```

This adds remote metadata columns and then creates the download-ready candidate lists.

Key outputs:

- `data/processed/screening/ds005752_mri_participants_age_20_25_remote_anat_forward.tsv`
- `data/processed/screening/ds005752_mri_participants_age_50_75_remote_anat_forward.tsv`

These are the tables to use when deciding who to download.

### 5. Choose the younger subjects to preprocess first

```bash
python code/03a_select_younger_preprocessing_subset.py
```

Key outputs:

- `data/processed/screening/ds005752_mri_participants_age_20_25_preprocessing_primary.tsv`
- `data/processed/screening/ds005752_mri_participants_age_20_25_preprocessing_backup.tsv`
- `data/processed/screening/ds005752_mri_participants_age_20_25_preprocessing_plan.tsv`

Use the primary TSV as the younger download/preprocessing list. The backup TSV should only be used if a primary younger subject later fails QC or cannot be processed.

### 6. Download the selected raw subjects

```bash
bash code/04_download_openneuro_subjects.sh
```

If the younger primary-selection TSV exists, the download script uses it by default for the younger group and still uses the full older `*_remote_anat_forward.tsv` table for the older group. You can also pass one or more TSV files or individual subject IDs explicitly.

### 7. Run `fMRIPrep` and QC

```bash
export FS_LICENSE=$HOME/license.txt
bash code/05_run_fmriprep_subjects.sh
```

This script:

- runs `fMRIPrep` on the selected subjects
- skips existing reports unless `FMRIPREP_FORCE=1`
- runs `code/06_qc_from_confounds.py` after each subject
- refreshes `data/processed/qc/manual_fmriprep_report_review.tsv`

If the younger primary-selection TSV exists, this script also uses it by default for the younger group.

QC outputs:

- `data/processed/qc/sub-ON*_qc_summary.tsv`
- `data/processed/qc/manual_fmriprep_report_review.tsv`

Open each subject's `fMRIPrep` HTML report and update the manual review TSV. Mark `manual_qc_status` as `pass` only when the structural mask, BOLD-to-T1w alignment, T1w-to-MNI alignment, and forward-run BOLD mask all look acceptable.

### 8. Build the final clean sample

```bash
python code/07_build_clean_sample.py
```

Key outputs:

- `data/processed/screening/ds005752_clean_age_sample.tsv`
- `data/processed/screening/ds005752_clean_age_sample_decisions.tsv`

The clean sample file contains only included subjects. The decisions file shows every candidate and the reason for inclusion or exclusion.

Subjects now need all of the following to enter the final sample:

- downloaded raw data
- required forward-run `fMRIPrep` outputs
- a confounds-based QC summary
- a manual `fMRIPrep` report-review pass
- at least `180` acquired forward-run volumes
- at least `150` retained volumes after censoring
- mean FD below `0.25 mm`
- fewer than `25%` volumes above `FD > 0.2 mm`

### 9. Compute connectivity and segregation

```bash
python code/08_run_connectivity_analysis.py
```

Key outputs:

- `data/processed/connectivity/metrics/subject_global_segregation.tsv`
- `data/processed/connectivity/metrics/subject_network_segregation.tsv`
- `data/processed/connectivity/metrics/atlas_labels.tsv`

### 10. Run the final age analysis

```bash
python code/09_run_age_group_analysis.py
```

Key outputs:

- `data/processed/analysis/sample_summary.tsv`
- `data/processed/analysis/model_coefficients.tsv`
- `data/processed/analysis/analysis_summary.md`
- `data/processed/analysis/figures/`

## Practical Recommendation

Do not preprocess all `MRI=1` participants immediately.

Use the workflow in this order:

1. screen remote availability
2. download only candidates with `anat + forward rest`
3. preprocess those candidates
4. exclude poor-quality data using QC
5. run the age-group analysis on the clean final sample

That keeps the workflow efficient, reproducible, and easy to justify in the dissertation methods section.

## Why This Is Defensible

- `fMRIPrep` gives you one standardized preprocessing pipeline for every participant instead of mixing legacy outputs across subjects.
- The final sample is screened with both automated QC and manual report review rather than motion numbers alone.
- The connectivity step uses a standard cortical atlas and nuisance-regression strategy, then censors high-motion volumes before correlation estimation.
- The final models still control for mean FD so residual motion differences are explicitly accounted for statistically.

## Core References

- Esteban et al. (2019), `fMRIPrep: a robust preprocessing pipeline for functional MRI`, Nature Methods.
- Behzadi et al. (2007), `a component based noise correction method (CompCor) for BOLD and perfusion based fMRI`.
- Power et al. (2014), `Methods to detect, characterize, and remove motion artifact in resting state fMRI`.
- Schaefer et al. (2018), `Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI`.
