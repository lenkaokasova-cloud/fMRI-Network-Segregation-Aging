#!/usr/bin/env bash

set -euo pipefail

# Run fMRIPrep, confounds-based QC, and manual-review template generation
# for selected ds005752 subjects.
#
# Usage:
#   export FS_LICENSE=$HOME/license.txt
#   bash code/05_run_fmriprep_subjects.sh
#   bash code/05_run_fmriprep_subjects.sh data/processed/screening/ds005752_mri_participants_age_50_75_remote_anat_forward.tsv
#   bash code/05_run_fmriprep_subjects.sh sub-ON01016 sub-ON39099

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIDS_DIR="${REPO_ROOT}/data/raw/ds005752"
OUT_DIR="${REPO_ROOT}/data/derivatives/fmriprep"
WORK_DIR="${REPO_ROOT}/data/work/fmriprep"
QCDIR="${REPO_ROOT}/data/processed/qc"
LOG_DIR="${REPO_ROOT}/data/logs/fmriprep"
NTHREADS="${FMRIPREP_NTHREADS:-6}"
OMP_NTHREADS="${FMRIPREP_OMP_NTHREADS:-4}"
MEM_MB="${FMRIPREP_MEM_MB:-12000}"
OUTPUT_RES="${FMRIPREP_OUTPUT_RES:-2}"
USE_FREESURFER="${FMRIPREP_USE_FREESURFER:-0}"
OUTPUT_T1W="${FMRIPREP_OUTPUT_T1W:-0}"
FORCE_FMRIPREP="${FMRIPREP_FORCE:-0}"
REQUIRE_FS_LICENSE="${FMRIPREP_REQUIRE_FS_LICENSE:-1}"
CONTINUE_ON_ERROR="${FMRIPREP_CONTINUE_ON_ERROR:-1}"
BATCH_STATUS_FILE="${QCDIR}/fmriprep_batch_status.tsv"
YOUNG_SELECTED_INPUT="${REPO_ROOT}/data/processed/screening/ds005752_mri_participants_age_20_25_preprocessing_primary.tsv"
YOUNG_FALLBACK_INPUT="${REPO_ROOT}/data/processed/screening/ds005752_mri_participants_age_20_25_remote_anat_forward.tsv"
OLDER_DEFAULT_INPUT="${REPO_ROOT}/data/processed/screening/ds005752_mri_participants_age_50_75_remote_anat_forward.tsv"

if [[ -f "${YOUNG_SELECTED_INPUT}" ]]; then
  YOUNG_DEFAULT_INPUT="${YOUNG_SELECTED_INPUT}"
else
  YOUNG_DEFAULT_INPUT="${YOUNG_FALLBACK_INPUT}"
fi

DEFAULT_INPUTS=(
  "${YOUNG_DEFAULT_INPUT}"
  "${OLDER_DEFAULT_INPUT}"
)

collect_subjects_from_arg() {
  local arg="$1"

  if [[ -f "${arg}" && "${arg}" == *.tsv ]]; then
    python - "${arg}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    key = "subject_id" if "subject_id" in reader.fieldnames else "participant_id"
    for row in reader:
        subject = row.get(key, "").strip()
        if subject:
            print(subject)
PY
  elif [[ "${arg}" == sub-* ]]; then
    printf '%s\n' "${arg}"
  else
    printf 'Unrecognized input: %s\n' "${arg}" >&2
    exit 1
  fi
}

append_unique_subject() {
  local subject="$1"
  local existing

  for existing in "${SUBJECTS[@]:-}"; do
    if [[ "${existing}" == "${subject}" ]]; then
      return
    fi
  done

  SUBJECTS+=("${subject}")
}

contains_subject() {
  local needle="$1"
  local item

  for item in "${@:2}"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done

  return 1
}

find_fs_license() {
  local candidate
  for candidate in \
    "${FS_LICENSE:-}" \
    "${FREESURFER_HOME:-}/license.txt" \
    "${HOME}/license.txt" \
    "${HOME}/Downloads/license.txt" \
    "${HOME}/Desktop/license.txt" \
    "/Applications/freesurfer/license.txt"
  do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

timestamp_now() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

clean_status_detail() {
  local detail="${1:-}"
  detail="${detail//$'\t'/ }"
  detail="${detail//$'\n'/ }"
  printf '%s' "${detail}"
}

record_status() {
  local subject="$1"
  local status="$2"
  local detail
  detail="$(clean_status_detail "${3:-}")"
  printf '%s\t%s\t%s\t%s\n' "$(timestamp_now)" "${subject}" "${status}" "${detail}" >> "${BATCH_STATUS_FILE}"
}

subject_outputs_complete() {
  local subject="$1"
  local func_dir="${OUT_DIR}/${subject}/ses-01/func"
  local report_file="${OUT_DIR}/${subject}.html"
  local bold_file="${func_dir}/${subject}_ses-01_task-rest_dir-forward_space-MNI152NLin2009cAsym_res-${OUTPUT_RES}_desc-preproc_bold.nii.gz"
  local confounds_file="${func_dir}/${subject}_ses-01_task-rest_dir-forward_desc-confounds_timeseries.tsv"
  local json_file="${func_dir}/${subject}_ses-01_task-rest_dir-forward_space-MNI152NLin2009cAsym_res-${OUTPUT_RES}_desc-preproc_bold.json"

  [[ -f "${report_file}" && -f "${bold_file}" && -f "${confounds_file}" && -f "${json_file}" ]]
}

run_fmriprep_subject() {
  local subject="$1"
  local report_file="${OUT_DIR}/${subject}.html"
  local raw_subject_dir="${BIDS_DIR}/${subject}"
  local subject_log="${LOG_DIR}/${subject}_$(date '+%Y%m%d_%H%M%S').log"

  if [[ ! -d "${raw_subject_dir}" ]]; then
    printf 'Skipping %s because raw data are missing at %s\n' "${subject}" "${raw_subject_dir}"
    record_status "${subject}" "missing_raw" "${raw_subject_dir}"
    MISSING_RAW_SUBJECTS+=("${subject}")
    return 0
  fi

  if [[ "${FORCE_FMRIPREP}" != "1" ]] && subject_outputs_complete "${subject}"; then
    printf 'Skipping fMRIPrep for %s because required outputs already exist\n' "${subject}"
    record_status "${subject}" "skipped_existing" "Required report, forward BOLD, confounds, and JSON are already present."
    SKIPPED_SUBJECTS+=("${subject}")
  else
    if [[ "${FORCE_FMRIPREP}" != "1" && -f "${report_file}" ]]; then
      printf 'Found partial outputs for %s. Rerunning because the required forward outputs are incomplete.\n' "${subject}"
    fi

    cmd=(
      fmriprep-docker
      "${BIDS_DIR}"
      "${OUT_DIR}"
      participant
      --participant-label "${subject#sub-}"
      --work-dir "${WORK_DIR}"
      --output-spaces "MNI152NLin2009cAsym:res-${OUTPUT_RES}"
      --nthreads "${NTHREADS}"
      --omp-nthreads "${OMP_NTHREADS}"
      --mem_mb "${MEM_MB}"
      --stop-on-first-crash
    )

    if [[ "${OUTPUT_T1W}" == "1" ]]; then
      cmd+=( "T1w" )
    fi

    if [[ -n "${FS_LICENSE_FILE}" ]]; then
      cmd+=( --fs-license-file "${FS_LICENSE_FILE}" )
    fi

    if [[ "${USE_FREESURFER}" != "1" ]]; then
      cmd+=( --fs-no-reconall )
    fi

    printf '\nRunning fMRIPrep for %s with %s threads, %s OMP threads, %s MB, MNI res-%s, FreeSurfer=%s\n' \
      "${subject}" "${NTHREADS}" "${OMP_NTHREADS}" "${MEM_MB}" "${OUTPUT_RES}" "${USE_FREESURFER}"
    printf 'Command log: %s\n' "${subject_log}"
    if "${cmd[@]}" 2>&1 | tee "${subject_log}"; then
      printf 'fMRIPrep finished for %s. Review %s\n' "${subject}" "${report_file}"
    else
      printf 'fMRIPrep failed for %s. Review %s\n' "${subject}" "${subject_log}" >&2
      record_status "${subject}" "failed" "${subject_log}"
      FAILED_SUBJECTS+=("${subject}")
      return 1
    fi
  fi

  if ! subject_outputs_complete "${subject}"; then
    printf 'Required forward outputs are still incomplete for %s after the run. Skipping QC.\n' "${subject}" >&2
    record_status "${subject}" "incomplete_outputs" "Subject report or forward outputs are incomplete after run."
    if ! contains_subject "${subject}" "${FAILED_SUBJECTS[@]}"; then
      FAILED_SUBJECTS+=("${subject}")
    fi
    return 1
  fi

  printf 'Running QC summary for %s...\n' "${subject}"
  python "${SCRIPT_DIR}/06_qc_from_confounds.py" --subject "${subject}"
  if ! contains_subject "${subject}" "${SKIPPED_SUBJECTS[@]}"; then
    record_status "${subject}" "completed" "${subject_log}"
    COMPLETED_SUBJECTS+=("${subject}")
  fi
  return 0
}

if ! command -v fmriprep-docker >/dev/null 2>&1; then
  echo "fmriprep-docker is not available. Install the conda environment first." >&2
  exit 1
fi

FS_LICENSE_FILE=""
if FS_LICENSE_FILE="$(find_fs_license)"; then
  printf 'Using FreeSurfer license file: %s\n' "${FS_LICENSE_FILE}"
elif [[ "${REQUIRE_FS_LICENSE}" == "1" ]]; then
  echo "FreeSurfer license not found. Set FS_LICENSE or place license.txt in \$HOME, \$HOME/Downloads, \$HOME/Desktop, or /Applications/freesurfer." >&2
  echo "Current fMRIPrep/Docker setup on this machine requires a valid FreeSurfer license even when --fs-no-reconall is used." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${WORK_DIR}" "${QCDIR}" "${LOG_DIR}"
printf 'timestamp\tsubject_id\tstatus\tdetail\n' > "${BATCH_STATUS_FILE}"

if [[ "$#" -gt 0 ]]; then
  INPUTS=("$@")
else
  INPUTS=("${DEFAULT_INPUTS[@]}")
fi

SUBJECTS=()
for input in "${INPUTS[@]}"; do
  while IFS= read -r subject; do
    [[ -z "${subject}" ]] && continue
    append_unique_subject "${subject}"
  done < <(collect_subjects_from_arg "${input}")
done

if [[ "${#SUBJECTS[@]}" -eq 0 ]]; then
  echo "No subjects found to preprocess." >&2
  exit 1
fi

COMPLETED_SUBJECTS=()
FAILED_SUBJECTS=()
SKIPPED_SUBJECTS=()
MISSING_RAW_SUBJECTS=()

for subject in "${SUBJECTS[@]}"; do
  if run_fmriprep_subject "${subject}"; then
    :
  elif [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    printf 'Continuing to the next subject after failure in %s.\n' "${subject}" >&2
  else
    printf 'Stopping after failure in %s because FMRIPREP_CONTINUE_ON_ERROR=%s\n' "${subject}" "${CONTINUE_ON_ERROR}" >&2
    break
  fi
done

python "${SCRIPT_DIR}/06_prepare_manual_qc_review.py" "${SUBJECTS[@]}"

printf '\nFinished fMRIPrep and QC for %s requested subjects.\n' "${#SUBJECTS[@]}"
printf 'Update data/processed/qc/manual_fmriprep_report_review.tsv after reviewing the HTML reports.\n'
printf 'Batch status log: %s\n' "${BATCH_STATUS_FILE}"
printf 'Completed: %s | Failed: %s | Skipped existing: %s | Missing raw: %s\n' \
  "${#COMPLETED_SUBJECTS[@]}" \
  "${#FAILED_SUBJECTS[@]}" \
  "${#SKIPPED_SUBJECTS[@]}" \
  "${#MISSING_RAW_SUBJECTS[@]}"

if [[ "${#FAILED_SUBJECTS[@]}" -gt 0 ]]; then
  printf 'Failed subjects: %s\n' "${FAILED_SUBJECTS[*]}" >&2
fi
