# Preprocessing Notes

These are just working notes for the locked preprocessing branch I am actually using.

## 1. Main branch I am treating as primary

The main dissertation branch is now:

- final sample file:
  [ds005752_final_analysis_sample_tr_3s.tsv](../data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv)
- total `n = 44`
- younger `n = 33`
- older `n = 11`
- only `TR = 3 s`

Older mixed-TR branches and the old balanced branches are not the main workflow anymore. If I mention them, it should only be as sensitivity or historical branches.

## 2. Where this pipeline sits methodologically

The current preprocessing branch is basically in line with:

- `BIDS`-based dataset screening and organization (`Gorgolewski et al., 2016`)
- `fMRIPrep` for the anatomical and functional preprocessing backbone (`Esteban et al., 2019`)
- study-specific denoising done later rather than inside `fMRIPrep`
- a motion + `aCompCor` nuisance model without making global signal regression the default (`Behzadi et al., 2007`; `Muschelli et al., 2014`; `Murphy and Fox, 2017`)

## 3. Sample flow I ended up with

Current sample flow:

- `239` participants marked `MRI = 1`
- younger age band (`20-25`): `69`
- older age band (`50-75`): `29`
- remote candidates with both `anat` and forward resting-state `func`:
  - younger `58`
  - older `22`
- QC-pass preprocessed sample:
  - younger `39`
  - older `15`
- final `TR = 3 s` sample:
  - younger `33`
  - older `11`

The sample-flow figure for this is:

- [sample_flow_primary.png](../data/processed/qc/reporting/figures/sample_flow_primary.png)

## 4. What counted as raw-data eligibility

I only treated a participant as eligible for preprocessing if remote OpenNeuro metadata showed:

- structural MRI `anat`
- forward resting-state BOLD `task-rest_dir-forward`

This was stricter than just keeping everyone with `MRI = 1`.

## 5. fMRIPrep settings

Main preprocessing used:

- `fMRIPrep 25.2.5`
- container `nipreps/fmriprep:25.2.5`
- `--output-spaces MNI152NLin2009cAsym:res-2`
- `--nthreads 4`
- `--omp-nthreads 2`
- `--mem_mb 10000`
- `--fs-no-reconall`

So the final sample is a volumetric MNI-space sample at `2 mm` resolution.

## 6. What fMRIPrep actually did

For the processed sample, `fMRIPrep` did:

- BIDS validation
- anatomical conforming and reference building
- skull stripping / brain extraction
- tissue segmentation into GM, WM, and CSF
- slice-timing correction
- rigid-body motion correction
- BOLD-to-T1w co-registration
- normalization to `MNI152NLin2009cAsym`
- resampling to `res-2`
- confounds extraction
- subject HTML report generation

The main functional file later used in analysis is:

- `sub-*_ses-01_task-rest_dir-forward_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz`

## 7. What fMRIPrep did not do

In the main branch, it did not include:

- fieldmap-based distortion correction
- `FreeSurfer` surface reconstruction
- extra spatial smoothing of the `desc-preproc_bold` images
- final study-specific nuisance regression
- final study-specific temporal filtering

Those last two are handled later in my own denoising script.

## 8. Why the omitted steps are still okay for this project

### No fieldmap correction

I did not use fieldmaps because:

- they were not available consistently across the final pool
- where they existed, they were not always linked properly to the resting-state runs in a way `fMRIPrep` could use

So I thought one consistent no-fieldmap branch was better than mixing corrected and uncorrected subjects.

### No FreeSurfer

I did not need `FreeSurfer` because this project is:

- volumetric
- atlas-based
- done in MNI space

So skipping `recon-all` saved a lot of time without removing something I actually needed for the analysis.

### No extra smoothing

I did not apply additional smoothing before parcel extraction because:

- parcel averaging already smooths in a practical sense
- extra smoothing can blur signal across parcel borders
- that can reduce parcel-level specificity

That is also in line with concerns in parcel-based connectivity work (`Alakorkko et al., 2017`; `Scheinost et al., 2014`).

## 9. QC rules for getting into the clean sample

A participant only entered the clean preprocessed pool if:

- the required forward-run outputs existed
- manual `fMRIPrep` review was marked `pass`
- the run had at least `200` acquired volumes
- at least `9.0` minutes remained after censoring
- `mean_fd < 0.20 mm`
- `pct_fd_gt_0p2 < 25%`

I used retained time rather than retained volume count because the broader processed dataset originally included more than one TR, so retained minutes was the fairer comparison.

- `9.0` minutes here is a minimum inclusion rule
- it is not meant to be presented as an ideal or optimal scan length


## 10. Atlas overlay QC before denoising

Before denoising, I check atlas alignment with:

- [09_check_atlas_overlay.py](../code/primary/09_check_atlas_overlay.py)

This overlays the `Schaefer 200 / Yeo 7` atlas onto each participant's normalized `T1w` image in MNI `2 mm` space.

Outputs:

- `data/processed/qc/atlas_overlay/figures/`
- [atlas_overlay_summary.tsv](../data/processed/qc/atlas_overlay/atlas_overlay_summary.tsv)

## 11. Study-specific denoising

Main denoising is done by:

- [10_run_denoising.py](../code/primary/10_run_denoising.py)

Main library:

- `nilearn.maskers.NiftiLabelsMasker`
- cleaning backend `nilearn.signal.clean`
- `nilearn 0.13.1`

## 12. What goes into denoising

For each subject, the script loads:

- forward preprocessed BOLD image
- confounds TSV
- BOLD JSON
- confounds JSON

The subject-specific `RepetitionTime` comes from the BOLD JSON, so filtering uses the correct TR.

## 13. Main censoring rule

In the main branch I censor volumes that are:

- marked as non-steady-state by `fMRIPrep`
- or have `framewise_displacement > 0.5 mm`

The main branch does not automatically add:

- DVARS-based censoring
- adjacent-frame expansion

Those are treated as sensitivity options instead.

## 14. Nuisance regressors in the main branch

The main nuisance model includes:

- 6 rigid-body motion parameters
- first derivatives of those 6 motion parameters
- mean white-matter signal
- mean CSF signal
- first 6 `aCompCor` components

The main branch does not include global signal regression. GSR is only kept as a separate sensitivity branch.

## 15. Note to self on aCompCor

The first six `aCompCor` components come from the `fMRIPrep` confounds file. I am treating them as a fixed nuisance model, not tuning them subject by subject.

This follows the usual `CompCor` logic of modeling structured non-neural variance from nuisance tissue regions (`Behzadi et al., 2007`; `Muschelli et al., 2014`).

In the current primary sample, cumulative variance explained by the first six `aCompCor` components was:

- mean `0.2696`
- median `0.2588`
- range `0.1495` to `0.6164`

This is recorded in:

- [subject_denoising_summary.tsv](../data/processed/denoising/metrics/subject_denoising_summary.tsv)

## 16. Physiological noise note

I did not find explicit cardiac or respiratory recordings in the raw BIDS search, so I could not include model-based physiological regressors.

So physiological noise is only being handled indirectly through:

- WM signal
- CSF signal
- `aCompCor`

Physiological noise is reduced, but not fully modeled (`Birn, 2012`).

## 17. Denoising order in the main branch

With `sample_mask` and Butterworth filtering, the effective order is:

1. flag censored volumes from non-steady-state status and `FD > 0.5 mm`
2. pass the censor mask into `nilearn.signal.clean`
3. spline-interpolate flagged volumes so filtering can be applied
4. detrend
5. band-pass filter (`0.008-0.09 Hz`)
6. re-apply censoring
7. regress out confounds
8. standardize the final parcel time series

Matters because filtering and regression can interact badly if done in a sloppy order (`Lindquist et al., 2019`).

## 18. Denoising outputs

Main denoising outputs are:

- denoised parcel time series:
  `data/processed/denoising/timeseries/sub-*_forward_timeseries.npy`
- [subject_denoising_summary.tsv](../data/processed/denoising/metrics/subject_denoising_summary.tsv)
- [denoising_settings.tsv](../data/processed/denoising/metrics/denoising_settings.tsv)
- [atlas_labels.tsv](../data/processed/denoising/metrics/atlas_labels.tsv)

Important interpretation boundary:

- these denoised parcel time series are cleaned BOLD signals, not direct neural recordings
- nuisance regression and censoring reduce non-neural variance, but they do not turn the data into a pure measure of neuronal activity
- later connectivity and segregation results should therefore be described as BOLD-based functional connectivity estimates

## 19. Extra QC reporting added later

- sample-flow figure
- retained-minutes distribution
- FD/DVARS/censor traces
- QC-FC summary tables
- QC-FC distribution figure
- QC-FC distance-dependence figure
- within-network denominator-stability figure

These are generated by:

- [15_build_qc_reporting.py](../code/utilities/15_build_qc_reporting.py)

Key outputs:

- `data/processed/qc/reporting/figures/`
- `data/processed/qc/reporting/motion_traces/`
- [qcfc_summary.tsv](../data/processed/qc/reporting/qcfc_summary.tsv)
- [qcfc_edge_summary.tsv](../data/processed/qc/reporting/qcfc_edge_summary.tsv)

These QC-FC outputs are there to show the residual motion pattern

## 20. Formal sensitivity branches

The preprocessing / denoising side includes sensitivity branches:

- GSR branch
- Schaefer-100 branch
- partial-correlation branch
- adjacent-frame scrub branch

These are formal supplementary branches around the locked primary pipeline. They do not replace the main no-`GSR`, Pearson-correlation, `Schaefer 200` branch.

For the adjacent-frame branch, each flagged motion spike is expanded by:

- `1` previous volume
- `2` following volumes

That branch is stored under:

- `data/processed/sensitivity/adjacent_scrub/`

The partial-correlation branch is stored under:

- `data/processed/sensitivity/partial_correlation/`

Other robustness checks linked to the same preprocessing / denoising decisions are handled later in the analysis stage, including:

- stricter motion subset analysis
- retained-time covariate sensitivity
- leave-one-older-out influence checks

## 21. Main limitations I still need to say clearly

- fieldmap correction was not available in a consistent usable form
- the main censoring rule is transparent and defensible, but not the most conservative possible
- physiological recordings were not available
- the no-`GSR` branch is one valid choice, not the only one
- denoised BOLD time series are still indirect measures of underlying neural activity

## 22. Atlas interpretation boundary

The main atlas is cortical only.

## 23. Reproducibility files

Current reproducibility files:

- [software_versions.tsv](../data/processed/reproducibility/software_versions.tsv)
- [workflow_context.tsv](../data/processed/reproducibility/workflow_context.tsv)
- [fmri_aging_environment.yml](../data/processed/reproducibility/fmri_aging_environment.yml)

Main implementation details are also recoverable from:

- [05_run_fmriprep_subjects.sh](../code/primary/05_run_fmriprep_subjects.sh)
- [10_run_denoising.py](../code/primary/10_run_denoising.py)
- [denoising_settings.tsv](../data/processed/denoising/metrics/denoising_settings.tsv)
