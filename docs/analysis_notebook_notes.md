# Analysis Notebook Notes

These are my working notes for the analysis branch I am actually using.

## 1. Locked primary analysis state

The main sample is:

- [ds005752_final_analysis_sample_tr_3s.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv)
- younger `n = 33`
- older `n = 11`
- `TR = 3 s` only

This is the primary sample in the dissertation.

The earlier mixed-TR and old balanced branches are not the main pipeline anymore.

The sample flow can still be audited from:

- [ds005752_qc_pass_sample.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/screening/ds005752_qc_pass_sample.tsv)
- [ds005752_qc_pass_sample_decisions.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/screening/ds005752_qc_pass_sample_decisions.tsv)

## 2. Where the analysis sits in the literature

The main analysis branch fits most closely with:

- standard resting-state functional connectivity using Pearson correlation plus Fisher `z` transform (`Biswal et al., 1995`; `Fisher, 1915`)
- large-scale network segregation work in aging (`Chan et al., 2014`; `Wig, 2017`)
- permutation-based inference plus multiplicity control (`Winkler et al., 2014`; `Benjamini and Hochberg, 1995`)
- a normative-style extension where older participants are also scored relative to the younger reference distribution (`Marquand et al., 2016`; `Wolfers et al., 2018`; `Kia et al., 2022`)

So overall the analysis is pretty standard conceptually, even if the sample is modest.

## 3. Why I restricted the main sample to TR = 3 s

The main branch excludes `TR = 2 s` runs because I did not want to mix acquisition protocols in the primary comparison.

That keeps interpretation simpler because:

- TR changes temporal sampling
- TR can affect variance structure and effective degrees of freedom
- one protocol is easier to defend than mixing protocols in the main confirmatory branch

So the final `33 young / 11 older` imbalance mainly comes from keeping the main branch protocol-consistent after QC and TR filtering, not from trying to force a perfectly matched sample.

## 4. Main analysis order after denoising

After denoising, the main order is:

1. [11_run_connectivity_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/primary/11_run_connectivity_analysis.py)
2. [12_run_age_group_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/primary/12_run_age_group_analysis.py)
3. [18_run_within_between_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/18_run_within_between_analysis.py)
4. [19_run_component_network_type_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/19_run_component_network_type_analysis.py)
5. [20_run_permutation_fdr_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/20_run_permutation_fdr_analysis.py)
6. [17_run_normative_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/17_run_normative_analysis.py)
7. [22_run_robustness_checks.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/sensitivity/22_run_robustness_checks.py)
8. [23_compare_sensitivity_branches.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/sensitivity/23_compare_sensitivity_branches.py)
9. [24_run_pca_anomaly_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/exploratory/24_run_pca_anomaly_analysis.py)
10. [25_run_classification_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/exploratory/25_run_classification_analysis.py)

The order I should keep in my write-up is:

- main age-group models first
- permutation/FDR follow-up second
- within/between decomposition after that
- normative branch as strongest secondary extension
- sensitivity branches after that
- PCA and classification last, as exploratory only

## 5. Main connectivity choices

Primary connectivity branch:

- Pearson correlation
- Fisher `z` transform
- `Schaefer 200 / Yeo 7`

This is still the most conventional first-pass setup for resting-state FC.

## 6. Main segregation metric

The main network metric is:

- `segregation_prop = (within_mean_z - between_mean_z) / within_mean_z`

where:

- `within_mean_z` is mean Fisher `z` connectivity within the network
- `between_mean_z` is mean Fisher `z` connectivity from that network to the other cortical networks

Companion metrics also kept:

- `segregation_raw_diff = within_mean_z - between_mean_z`
- `segregation_prop_posonly`

I kept those because ratio-style metrics can get awkward if the denominator becomes very small.

## 7. Denominator stability check

For the primary sample:

- minimum `within_mean_z` across network rows = `0.1536`
- rows with `within_mean_z <= 0` = `0`
- rows with `|within_mean_z| < 0.05` = `0`

So the denominator for the proportional metric did not get near zero in the main sample. That makes the proportional metric usable here, although the raw-difference metric is still a useful companion.

## 8. Main age-group model setup

Main covariates:

- sex
- mean FD

For repeated network rows, the current code uses:

- cluster-robust standard errors by `subject_id`

That makes sense because:

- network rows are repeated within person
- the sample is not huge
- the models stay simple enough to report clearly

A mixed model would also be reasonable, but it is not the current main implementation.

## 9. Main primary result

For global proportional segregation:

- older-vs-younger estimate = `-0.0464`
- 95% CI = `[-0.1415, 0.0487]`
- asymptotic `p = 0.3392`

So the direction matches the idea of lower segregation in older adults, but it is not strong enough to claim a solid confirmatory age effect on its own.

That is how I should write it:

- directionally consistent
- modest
- not strong confirmatory evidence by itself

## 10. Within- vs between-network decomposition

I now split segregation into:

- within-network connectivity
- between-network connectivity

This matters because lower segregation can happen because of:

- weaker internal network coherence
- more cross-network mixing
- or both

The current pattern looks fairly modest and mixed rather than one huge clean shift.

## 11. Sensory/motor vs higher-order comparison

Grouped system comparison:

- sensory/motor = `Vis`, `SomMot`
- higher-order = `Default`, `Cont`, `DorsAttn`, `SalVentAttn`

The current grouped interaction is weak.

So the safest wording is:

- the direction is broadly compatible with the literature
- but this sample does not give strong evidence for a selective higher-order effect

## 12. Permutation and FDR step

The strictest inferential checkpoint in the main branch is:

- Freedman-Lane permutation inference
- FDR correction across the network-family tests

For the primary global proportional segregation result:

- permutation `p = 0.3032`
- `q = 0.6063`

So the main global effect does not survive the stricter permutation/FDR step.

That means the safest summary is:

- the findings are in the expected direction
- they are more descriptive / suggestive than confirmatory
- they should not be written up as a strong established group difference

## 13. Normative analysis

The normative branch treats the younger group as the reference and then scores the older group relative to it.

This is one of the strongest secondary analyses in the repo because it asks:

- which networks deviate most from younger norms
- whether some older participants fall outside the younger reference range

This is easier to interpret biologically than the classifier branch, so if space is limited this is the secondary branch I should keep.

## 14. Sensitivity branches I now have

The appraisal-driven sensitivity branches now include:

- GSR
- Schaefer-100
- partial correlation
- adjacent-frame motion expansion

### GSR branch

- branch name: `with_gsr`
- purpose: check whether the result depends heavily on including global signal regression

Current global result:

- estimate = `-0.0197`
- permutation `q = 0.1846`

### Schaefer-100 branch

- branch name: `schaefer_100`
- purpose: check whether the overall pattern depends a lot on atlas resolution

Current global result:

- estimate = `-0.0414`
- permutation `q = 0.7173`

### Partial-correlation branch

- branch name: `partial_correlation`
- purpose: check whether a less motion-sensitive FC estimator changes the result

Current global result:

- estimate = `-0.0074`
- asymptotic `p = 0.0501`
- permutation `p = 0.0241`
- permutation `q = 0.0482`

This is the most supportive sensitivity branch, but it is still a sensitivity branch, not the main prespecified result.

### Adjacent-frame scrub branch

- branch name: `adjacent_scrub`
- purpose: check whether expanding each motion spike to nearby frames changes the result
- implementation: `1` volume backward and `2` forward

Current global result:

- estimate = `-0.0544`
- permutation `q = 0.4334`

So this keeps the effect negative, but does not make it much stronger inferentially.

## 15. Overall branch pattern

Across the current branches:

- the sign of the global age effect stays negative
- the size of the effect moves around depending on the analytic choice
- partial correlation is the strongest supportive branch
- the main Pearson branch still stays the primary analysis

So overall I can say there is directional robustness, but not uniform inferential robustness.

## 16. Balanced 11x11 sensitivity sample

I also added a balanced-sample branch to check whether the main result depends too much on the `33 young / 11 older` imbalance.

Balanced sample details:

- `11` younger
- `11` older
- `TR = 3 s` only
- exact sex matching
- one-to-one pairing optimized on `mean_fd` and `retained_minutes_after_scrub`

Files:

- `data/processed/screening/ds005752_balanced_sensitivity_sample_tr_3s_11x11.tsv`
- `data/processed/screening/ds005752_balanced_sensitivity_sample_tr_3s_11x11_pairs.tsv`
- `data/processed/screening/ds005752_balanced_sensitivity_sample_tr_3s_11x11_summary.md`

Balanced global result:

- estimate = `-0.0565`
- asymptotic `p = 0.3956`
- permutation `p = 0.3635`
- permutation `q = 0.7445`

So the negative direction stays the same, but the balanced branch does not make the case stronger. It is useful as a robustness check, not as a better main analysis.

## 17. Extra QC reporting I now have

The workflow now also includes a compact QC-reporting set from:

- [15_build_qc_reporting.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/utilities/15_build_qc_reporting.py)

This includes:

- sample-flow figure
- retained-minutes distribution
- FD/DVARS/censor traces
- QC-FC summary
- QC-FC distance-dependence figure
- within-network denominator-stability figure

Important outputs:

- [qcfc_summary.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/qc/reporting/qcfc_summary.tsv)
- [sample_flow_primary.png](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/qc/reporting/figures/sample_flow_primary.png)
- [retained_minutes_distribution.png](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/qc/reporting/figures/retained_minutes_distribution.png)

The point of these is to show the residual motion story honestly, not to pretend denoising removed everything.

## 18. Main methodological weaknesses I still need to name

Even though the pipeline is coherent, I should still state clearly that:

- the sample is modest and group-imbalanced
- the main results are better treated as effect estimation plus sensitivity checking than as strong confirmatory evidence
- Pearson correlation is standard, but not the only plausible FC estimator
- the no-`GSR` branch is defensible, but not uniquely correct

These points mainly affect how strongly I should claim the findings, not whether the pipeline itself is acceptable.

## 19. How I should report the results

I should lean mainly on:

- regression estimates
- confidence intervals
- raw group means where useful
- corrected permutation results

I should rely less on:

- uncorrected p-values by themselves
- binary significant / non-significant wording

That fits the current strength of the evidence much better.

## 20. Interpretation boundaries

What I can say fairly safely:

- the direction of the age effect broadly matches published aging work on lower segregation and more network mixing
- the pattern looks modest rather than dramatic
- the evidence is stronger descriptively than confirmatorily

What I should say more carefully:

- the cortical `Limbic` network is not the same thing as direct hippocampus / amygdala / subcortical limbic anatomy
- the main branch does not justify strong mechanistic or causal claims
- the partial-correlation result is supportive, but still only a sensitivity result

## 21. Exploratory branches

The repo also contains:

- PCA-based anomaly analysis
- simple age-group classification

These are interesting, but they should stay behind:

- the main age-group models
- permutation/FDR follow-up
- normative interpretation

## 22. Best final hierarchy for the dissertation

If I need to keep the write-up focused, the cleanest order is:

1. main age-group segregation analysis
2. permutation + `FDR` follow-up
3. within- vs between-network decomposition
4. normative deviation analysis
5. targeted sensitivity branches
6. PCA and classification only briefly, if at all

That is the order that best matches how strong the actual evidence is.

## 23. Reproducibility files

Current reproducibility files:

- [software_versions.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/reproducibility/software_versions.tsv)
- [workflow_context.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/reproducibility/workflow_context.tsv)
- [fmri_aging_environment.yml](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/reproducibility/fmri_aging_environment.yml)

Analysis settings and branch outputs are also recoverable from:

- [connectivity_settings.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/connectivity/metrics/connectivity_settings.tsv)
- [20_run_permutation_fdr_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/20_run_permutation_fdr_analysis.py)
- [23_compare_sensitivity_branches.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/sensitivity/23_compare_sensitivity_branches.py)
- [analysis_summary.md](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/analysis/analysis_summary.md)

Current recorded workflow snapshot commit:

- `d10985de220978834e485821a805b72c92c70521`

## 24. Exact calculation map for the analysis code

This is the clean map of what is actually calculated in the code, what the variable names are, and where those values get written out.

### 24.1 Connectivity matrix calculation

Main script:

- [11_run_connectivity_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/primary/11_run_connectivity_analysis.py)

Main calculation steps:

- denoised parcel time series are loaded from the denoising output
- parcel-by-parcel functional connectivity is calculated with Nilearn `ConnectivityMeasure`
- main estimator = `correlation`
- sensitivity estimator = `partial_correlation`
- the correlation matrix is Fisher `z` transformed using:
  - `z = arctanh(r)`

Exact code locations:

- Fisher `z` transform: `fisher_z()` in [11_run_connectivity_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/primary/11_run_connectivity_analysis.py:102)
- network grouping from atlas labels: `network_groups()` in [11_run_connectivity_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/primary/11_run_connectivity_analysis.py:113)
- connectivity estimator setup: [11_run_connectivity_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/primary/11_run_connectivity_analysis.py:213)

Saved outputs:

- subject matrices:
  - `data/processed/connectivity/matrices/sub-*_forward_z_matrix.npy`
  - `data/processed/connectivity/matrices/sub-*_forward_z_matrix.csv`
- group mean matrix:
  - `data/processed/connectivity/matrices/group_mean_z_matrix.npy`
  - `data/processed/connectivity/matrices/group_mean_z_matrix.csv`

### 24.2 Network segregation calculation

Main network calculation function:

- `compute_network_segregation()` in [11_run_connectivity_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/primary/11_run_connectivity_analysis.py:133)

Main helper functions:

- `mean_off_diagonal()` = average of within-network off-diagonal values
- `proportional_segregation()` = main segregation formula

Exact main formula:

- `segregation_prop = (within_mean_z - between_mean_z) / within_mean_z`

Exact variable names written by the script:

- `within_mean_z`
- `between_mean_z`
- `segregation`
- `segregation_raw_diff`
- `segregation_prop`
- `within_mean_z_posonly`
- `between_mean_z_posonly`
- `segregation_prop_posonly`

What those mean:

- `within_mean_z` = mean Fisher `z` connectivity among parcels inside one network
- `between_mean_z` = mean Fisher `z` connectivity from parcels in that network to parcels outside it
- `segregation_raw_diff` = simple difference:
  - `within_mean_z - between_mean_z`
- `segregation_prop` = proportional version of the same contrast
- `*_posonly` = sensitivity version after negative values are set to `0`

Saved output:

- [subject_network_segregation.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/connectivity/metrics/subject_network_segregation.tsv)

### 24.3 Global segregation calculation

Main function:

- `compute_global_metrics()` in [11_run_connectivity_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/primary/11_run_connectivity_analysis.py:168)

How it is calculated:

- first compute a within-network mean for each network
- then compute a between-network mean for each network
- then average those values across networks

Exact global variable names:

- `global_within_mean_z`
- `global_between_mean_z`
- `global_segregation`
- `global_segregation_raw_diff`
- `global_segregation_prop`
- `global_within_mean_z_posonly`
- `global_between_mean_z_posonly`
- `global_segregation_prop_posonly`

Main formula:

- `global_segregation_prop = (global_within_mean_z - global_between_mean_z) / global_within_mean_z`

Saved output:

- [subject_global_segregation.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/connectivity/metrics/subject_global_segregation.tsv)

The script also writes the metric definitions into:

- [connectivity_settings.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/connectivity/metrics/connectivity_settings.tsv)

That file records:

- `primary_global_metric = global_segregation_prop`
- `primary_network_metric = segregation_prop`
- `legacy_global_sensitivity_metric = global_segregation_raw_diff`
- `legacy_network_sensitivity_metric = segregation_raw_diff`
- `positive_only_sensitivity_metric = segregation_prop_posonly`

### 24.4 Main age-group models

Main script:

- [12_run_age_group_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/primary/12_run_age_group_analysis.py)

This is the main primary analysis script for the dissertation branch.

Main model formulas:

- global group model:
  - `global_segregation_prop ~ C(age_group, Treatment(reference='young')) + C(sex) + mean_fd`
- continuous age model:
  - `global_segregation_prop ~ age + C(sex) + mean_fd`
- network-type model:
  - `segregation_prop ~ C(age_group, Treatment(reference='young')) * C(network_type, Treatment(reference='sensory_motor')) + C(sex) + mean_fd`
- network-specific model:
  - `segregation_prop ~ C(age_group, Treatment(reference='young')) * C(network) + C(sex) + mean_fd`

Exact code location for those:

- [12_run_age_group_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/primary/12_run_age_group_analysis.py:467)

Covariates in the main branch:

- `sex`
- `mean_fd`

Standard error choices:

- global and continuous-age models use `HC3`
- repeated network-row models use cluster-robust SE by `subject_id`

Key output tables:

- [model_coefficients.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/analysis/model_coefficients.tsv)
- [sample_summary.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/analysis/sample_summary.tsv)
- [subject_global_with_metadata.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/analysis/subject_global_with_metadata.tsv)
- [subject_network_with_metadata.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/analysis/subject_network_with_metadata.tsv)
- [subject_network_type_summary.tsv](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/data/processed/analysis/subject_network_type_summary.tsv)

Main coefficient columns in `model_coefficients.tsv`:

- `model`
- `term`
- `estimate`
- `std_error`
- `statistic`
- `p_value`
- `conf_low`
- `conf_high`
- `nobs`
- `r_squared`

### 24.5 Within- and between-network decomposition

Follow-up script:

- [18_run_within_between_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/18_run_within_between_analysis.py)

Main subject-level averages created there:

- `overall_within_mean`
- `overall_between_mean`
- `overall_segregation`

Exact code location:

- [18_run_within_between_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/18_run_within_between_analysis.py:157)

Subject-level model formula:

- `outcome ~ C(age_group, Treatment(reference='young')) + C(sex) + mean_fd`

This is fitted separately for:

- `overall_within_mean`
- `overall_between_mean`
- `overall_segregation`

Then network-specific models are also fitted separately for:

- `within_mean_z`
- `between_mean_z`
- `segregation_prop`

Important output names from this branch:

- subject-level component table
- network effect table
- network component summary

### 24.6 Higher-order vs sensory/motor interaction analysis

Follow-up script:

- [19_run_component_network_type_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/19_run_component_network_type_analysis.py)

This script first recodes networks into:

- `sensory_motor` = `Vis`, `SomMot`
- `higher_order` = `Default`, `Cont`, `DorsAttn`, `SalVentAttn`

Exact code location:

- network-type recode: [19_run_component_network_type_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/19_run_component_network_type_analysis.py:141)

The interaction model is:

- `value ~ C(age_group, Treatment(reference='young')) * C(network_type, Treatment(reference='sensory_motor')) + C(sex) + mean_fd`

Exact code location:

- [19_run_component_network_type_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/19_run_component_network_type_analysis.py:181)

It is run separately for:

- `within_mean_z`
- `between_mean_z`
- `segregation_prop`

Main interpreted term names:

- `older_vs_young_within_sensory_motor`
- `higher_order_vs_sensory_motor_within_young`
- `additional_older_effect_in_higher_order`

### 24.7 Permutation and FDR inference

Follow-up script:

- [20_run_permutation_fdr_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/20_run_permutation_fdr_analysis.py)

Main permutation setup:

- Freedman-Lane permutation test
- default `n_permutations = 10000`
- age group is the tested term
- nuisance covariates are kept in the reduced model

Exact reduced and full design matrices:

- full design:
  - intercept
  - `age_older`
  - `sex_male`
  - `mean_fd`
- reduced design:
  - intercept
  - `sex_male`
  - `mean_fd`

Exact code locations:

- BH-FDR function: [20_run_permutation_fdr_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/20_run_permutation_fdr_analysis.py:105)
- design encoding: [20_run_permutation_fdr_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/20_run_permutation_fdr_analysis.py:144)
- Freedman-Lane implementation: [20_run_permutation_fdr_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/20_run_permutation_fdr_analysis.py:166)

Important output columns:

- `estimate_older_vs_young`
- `std_error`
- `conf_low`
- `conf_high`
- `permutation_p_value`
- `fdr_q_value`
- `young_mean`
- `older_mean`
- `older_minus_young_mean`

Important caution:

- the `p` values in this branch are permutation-based
- the confidence intervals are still standard beta `+/- 1.96 * SE`
- so the confidence intervals are not themselves permutation confidence intervals

### 24.8 Normative analysis

Follow-up script:

- [17_run_normative_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/17_run_normative_analysis.py)

This branch uses the younger group as the reference distribution.

Reference table calculation:

- `reference_mean`
- `reference_sd`
- `reference_median`
- `reference_p025`
- `reference_p05`
- `reference_p95`
- `reference_p975`
- `reference_z_lower`
- `reference_z_upper`

Exact code location:

- [17_run_normative_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/17_run_normative_analysis.py:214)

Subject scoring calculation:

- `deviation_from_reference_mean = value - reference_mean`
- `z_score = (value - reference_mean) / reference_sd`
- `abs_z_score = abs(z_score)`
- `outside_reference_z95 = abs_z_score > 1.96`
- `below_reference_p05 = value < reference_p05`
- `above_reference_p95 = value > reference_p95`
- `outside_reference_empirical_95 = value < reference_p025 or value > reference_p975`

Exact code location:

- [17_run_normative_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/followup/17_run_normative_analysis.py:275)

This is done for:

- global segregation
- network-specific segregation

### 24.9 PCA anomaly analysis

Exploratory script:

- [24_run_pca_anomaly_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/exploratory/24_run_pca_anomaly_analysis.py)

Feature matrix:

- one row per subject
- columns are the 7 network `segregation_prop` values
- optional extra feature: `global_segregation_prop`

Exact code location:

- feature matrix build: [24_run_pca_anomaly_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/exploratory/24_run_pca_anomaly_analysis.py:203)

Standardization:

- done using younger-group means and SDs only

Exact code:

- [24_run_pca_anomaly_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/exploratory/24_run_pca_anomaly_analysis.py:240)

PCA fitting:

- PCA is fit on the younger standardized data only
- SVD is used internally
- enough components are kept to reach the cumulative explained variance threshold
- default threshold = `0.90`

Exact code:

- [24_run_pca_anomaly_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/exploratory/24_run_pca_anomaly_analysis.py:264)

Anomaly scores:

- `reconstruction_error = sqrt(mean(residual^2))`
- `pca_distance = sqrt(sum((PC_score^2) / retained_variance))`
- `reconstruction_error_z`
- `pca_distance_z`
- `error_outside_reference_p95`
- `distance_outside_reference_p95`
- `pca_anomaly_flag`

Exact code:

- [24_run_pca_anomaly_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/exploratory/24_run_pca_anomaly_analysis.py:294)

### 24.10 Exploratory classification analysis

Exploratory script:

- [25_run_classification_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/exploratory/25_run_classification_analysis.py)

Feature matrix:

- network features only:
  - `Vis`, `SomMot`, `DorsAttn`, `SalVentAttn`, `Limbic`, `Cont`, `Default`
- or network features plus:
  - `global_segregation_prop`

Target variable:

- `target = 1` for `older`
- `target = 0` for `young`

Exact code location:

- [25_run_classification_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/exploratory/25_run_classification_analysis.py:187)

Models used:

- logistic regression
- linear SVM

Exact settings:

- logistic regression:
  - `class_weight = "balanced"`
  - `solver = "liblinear"`
  - `max_iter = 2000`
- SVM:
  - `kernel = "linear"`
  - `class_weight = "balanced"`

Exact code location:

- [25_run_classification_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/exploratory/25_run_classification_analysis.py:219)

Cross-validation:

- repeated stratified k-fold
- default:
  - `n_splits = 5`
  - `n_repeats = 100`

Performance metrics written out:

- `accuracy`
- `balanced_accuracy`
- `sensitivity`
- `specificity`
- `roc_auc`
- confusion matrix values:
  - `tn`
  - `fp`
  - `fn`
  - `tp`

Exact code location:

- [25_run_classification_analysis.py](/Users/lenkaokasova/Documents/GitHub/Dissertation-fMRI-Aging/fMRI-Network-Segregation-Aging/code/exploratory/25_run_classification_analysis.py:267)

### 24.11 Short version of what I actually use most

If I need the shortest possible explanation of the main analysis branch, the key outcome variables are:

- primary global metric:
  - `global_segregation_prop`
- primary network metric:
  - `segregation_prop`

The main dissertation model is:

- `global_segregation_prop ~ age_group + sex + mean_fd`

with the exact coded version:

- `global_segregation_prop ~ C(age_group, Treatment(reference='young')) + C(sex) + mean_fd`

And the main supporting follow-ups are:

- permutation + `FDR`
- within vs between decomposition
- normative scoring relative to younger participants
