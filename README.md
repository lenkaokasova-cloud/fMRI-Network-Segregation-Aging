# fMRI-Network-Segregation-Aging

This repository is now focused on one dissertation workflow only:

1. identify NIMH `ds005752` participants with `MRI=1`
2. split them into the chosen younger and older age bands
3. screen remote OpenNeuro metadata for raw `anat` plus forward resting-state `func`
4. choose the fixed younger preprocessing subset
5. download only those candidates
6. preprocess them consistently with `fMRIPrep`
7. exclude poor-quality data using explicit QC thresholds plus manual report review
8. compute resting-state network segregation with nuisance regression and frame censoring
9. run the final age-group analysis controlling for sex and mean FD

## Environment

```bash
conda env create -f environment.yml
conda activate fmri-aging
```

## Main Workflow

Run the scripts in this order from the repository root:

```bash
python code/01_filter_mri_participants.py
python code/02_split_age_groups.py
python code/03_screen_remote_anat_forward_rest.py
python code/03a_select_younger_preprocessing_subset.py
bash code/04_download_openneuro_subjects.sh
export FS_LICENSE=$HOME/license.txt
bash code/05_run_fmriprep_subjects.sh
python code/07_build_clean_sample.py
python code/08_run_connectivity_analysis.py
python code/09_run_age_group_analysis.py
```

`code/06_qc_from_confounds.py` and `code/06_prepare_manual_qc_review.py` are called automatically by `code/05_run_fmriprep_subjects.sh`, but you can also run them manually if needed.

## Scripts

- `code/01_filter_mri_participants.py`: create the `MRI=1` participant table.
- `code/02_split_age_groups.py`: split that table into younger and older age-band TSVs.
- `code/03_screen_remote_anat_forward_rest.py`: query OpenNeuro metadata and keep only participants with remote `anat` plus forward resting-state `func`.
- `code/03a_select_younger_preprocessing_subset.py`: choose the fixed primary and backup younger preprocessing sets before download.
- `code/04_download_openneuro_subjects.sh`: download selected raw subjects from one or more TSV files.
- `code/05_run_fmriprep_subjects.sh`: run `fMRIPrep` and generate QC summaries for the selected subjects.
- `code/06_qc_from_confounds.py`: summarize motion and DVARS from `fMRIPrep` confounds files.
- `code/06_prepare_manual_qc_review.py`: create the manual `fMRIPrep` report-review TSV used by the clean-sample step.
- `code/07_build_clean_sample.py`: build the final clean analysis sample after QC.
- `code/08_run_connectivity_analysis.py`: compute Schaefer/Yeo connectivity and segregation metrics.
- `code/09_run_age_group_analysis.py`: run the final age-group models and write results tables and figures.

## Key Outputs

Candidate-screening outputs are written to:

- `data/processed/screening/`

The clean final sample is written to:

- `data/processed/screening/ds005752_clean_age_sample.tsv`

Manual-review outputs are written to:

- `data/processed/qc/manual_fmriprep_report_review.tsv`

Connectivity outputs are written to:

- `data/processed/connectivity/metrics/`

Final age-analysis outputs are written to:

- `data/processed/analysis/`

More detailed guidance is in [docs/preprocessing_notes.md](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/docs/preprocessing_notes.md).
