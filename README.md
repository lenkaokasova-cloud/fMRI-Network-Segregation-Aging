# fMRI-Network-Segregation-Aging

This repo contains the main workflow I used for my dissertation analysis of resting-state network segregation in OpenNeuro `ds005752`.

The aim of the cleanup here is simple:

- keep the main workflow easy to follow from start to finish
- keep the optional branches available without mixing them into the main path
- make it obvious which scripts are primary, which are checks, and which are extensions

## Quick Start

If someone wants the shortest reproducible main path, the core commands are:

```bash
python code/primary/01_filter_mri_participants.py
python code/primary/02_split_age_groups.py
python code/primary/03_screen_remote_anat_forward_rest.py
python code/primary/04_select_younger_preprocessing_subset.py
bash code/primary/05_download_openneuro_subjects.sh
export FS_LICENSE=$HOME/license.txt
bash code/primary/06_run_fmriprep_subjects.sh
python code/primary/09_build_clean_sample.py
python code/primary/10_check_atlas_overlay.py
python code/primary/11_run_denoising.py
python code/primary/12_run_connectivity_analysis.py
python code/primary/13_run_age_group_analysis.py
```

Main follow-up inference:

```bash
python code/followup/21_run_permutation_fdr_analysis.py
python code/followup/18_run_normative_analysis.py
```

## Main Idea

The primary branch of the project is:

1. keep only `MRI = 1` participants
2. split into younger and older age groups
3. check remote metadata for `anat` plus forward resting-state `func`
4. lock the younger preprocessing subset used in this project
5. download those subjects
6. run `fMRIPrep`
7. apply automated QC plus manual report review
8. build the final `TR = 3 s` analysis sample
9. check atlas alignment
10. denoise parcel time series
11. build connectivity and segregation metrics
12. run the main age-group analysis

## Environment

```bash
conda env create -f environment.yml
conda activate fmri-aging
```

## Repo Layout

The `code/` folder is now split by purpose.

- `code/primary/`
  Main scripts needed to go from screened participants to the primary analysis outputs.

- `code/utilities/`
  Helper scripts for TR annotation, denoised time-series checks, QC reporting, and dissertation summary tables.

- `code/followup/`
  Follow-up analyses that help interpret the main findings.

- `code/sensitivity/`
  Sensitivity branches and comparison scripts.

- `code/exploratory/`
  Exploratory extensions such as PCA anomaly scoring and classification.

- `code/archive/`
  Older or diagnostic scripts kept for provenance, not needed for the main branch.

## Main Workflow

Run these from the repo root:

```bash
python code/primary/01_filter_mri_participants.py
python code/primary/02_split_age_groups.py
python code/primary/03_screen_remote_anat_forward_rest.py
python code/primary/04_select_younger_preprocessing_subset.py
bash code/primary/05_download_openneuro_subjects.sh
export FS_LICENSE=$HOME/license.txt
bash code/primary/06_run_fmriprep_subjects.sh
python code/primary/09_build_clean_sample.py
python code/primary/10_check_atlas_overlay.py
python code/primary/11_run_denoising.py
python code/primary/12_run_connectivity_analysis.py
python code/primary/13_run_age_group_analysis.py
```

These are the main QC and inference follow-ups I would normally run after that:

```bash
python code/followup/21_run_permutation_fdr_analysis.py
python code/followup/18_run_normative_analysis.py
```

`code/primary/08_qc_from_confounds.py` and `code/primary/07_prepare_manual_qc_review.py` are normally called inside `code/primary/06_run_fmriprep_subjects.sh`, so I do not usually need to run them separately.

## Primary Sample

The locked primary sample for the dissertation is:

- `data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv`

That is the sample that should be treated as primary in the write-up.

## Main Scripts

- `code/primary/01_filter_mri_participants.py`
  Filters the OpenNeuro participant table to rows with `MRI = 1`.

- `code/primary/02_split_age_groups.py`
  Splits the MRI-eligible table into the younger and older age bands.

- `code/primary/03_screen_remote_anat_forward_rest.py`
  Checks remote OpenNeuro metadata and keeps only subjects with `anat` and a forward resting-state run.

- `code/primary/04_select_younger_preprocessing_subset.py`
  Locks the younger preprocessing subset used in this project so the workflow reproduces the same branch.

- `code/primary/05_download_openneuro_subjects.sh`
  Downloads the selected subjects from OpenNeuro.

- `code/primary/06_run_fmriprep_subjects.sh`
  Runs `fMRIPrep`, writes logs, creates confounds-based QC summaries, and updates the manual-review TSV.

- `code/primary/09_build_clean_sample.py`
  Applies the QC rules and builds the QC-pass sample.

- `code/primary/10_check_atlas_overlay.py`
  Checks whether the Schaefer/Yeo atlas sits sensibly on the normalized anatomy.

- `code/primary/11_run_denoising.py`
  Runs the parcel-level denoising workflow and saves denoised time series.

- `code/primary/12_run_connectivity_analysis.py`
  Builds the parcel connectivity matrices and the network/global segregation metrics.

- `code/primary/13_run_age_group_analysis.py`
  Runs the main age-group regression models and writes the primary analysis outputs.

## Utilities

- `code/utilities/14_annotate_remote_tr.py`
  Annotates remote candidate tables with `TR` before full download.

- `code/utilities/15_check_denoised_timeseries.py`
  Quick visual check of denoised parcel time series.

- `code/utilities/16_build_qc_reporting.py`
  Builds the extra QC reporting figures and summary tables.

- `code/utilities/17_build_dissertation_summary_outputs.py`
  Builds cleaned dissertation tables and summary figures from the analysis outputs.

## Follow-Up Analyses

- `code/followup/21_run_permutation_fdr_analysis.py`
  Adds the stricter permutation and FDR-based inferential layer.

- `code/followup/18_run_normative_analysis.py`
  Scores older participants relative to the younger reference distribution.

- `code/followup/19_run_within_between_analysis.py`
  Splits segregation into within-network and between-network components.

- `code/followup/20_run_component_network_type_analysis.py`
  Compares sensory/motor and higher-order network components.

## Sensitivity Analyses

- `code/sensitivity/22_make_balanced_tr3_sensitivity_sample.py`
  Builds the balanced `11 x 11` TR = 3 s sensitivity sample.

- `code/sensitivity/23_run_robustness_checks.py`
  Runs robustness branches such as GSR, partial correlation, or related alternatives.

- `code/sensitivity/24_compare_sensitivity_branches.py`
  Compares the main branch against the sensitivity branches.

## Exploratory Analyses

- `code/exploratory/25_run_pca_anomaly_analysis.py`
  PCA-based anomaly/deviation scoring.

- `code/exploratory/26_run_classification_analysis.py`
  Small exploratory classification branch.

## Main Output Folders

- screening tables:
  `data/processed/screening/`

- QC outputs:
  `data/processed/qc/`

- denoising outputs:
  `data/processed/denoising/`

- connectivity outputs:
  `data/processed/connectivity/`

- analysis outputs:
  `data/processed/analysis/`

## Notes

More detailed notes are in:

- [docs/preprocessing_notes.md](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/docs/preprocessing_notes.md)
- [docs/analysis_notebook_notes.md](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/docs/analysis_notebook_notes.md)
- [docs/dissertation_figure_table_notes.md](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/docs/dissertation_figure_table_notes.md)
