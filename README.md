# fMRI-Network-Segregation-Aging

Small dissertation project scaffold for demonstrating preprocessing of an OpenNeuro subset from `ds005752` before moving on to resting-state network segregation analyses in aging.

## First milestone

Preprocess one participant from `ds005752` with `fMRIPrep` and review the quality-control report.

```bash
conda env create -f environment.yml
conda activate fmri-aging
bash code/01_download_openneuro_subset.sh sub-ON01016
export FS_LICENSE=$HOME/license.txt
bash code/02_run_fmriprep_subset.sh sub-ON01016
python code/03_qc_from_confounds.py --subject sub-ON01016
```

More detail is in [docs/preprocessing_notes.md](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/docs/preprocessing_notes.md).
