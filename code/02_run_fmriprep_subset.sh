#!/usr/bin/env bash

set -euo pipefail

# Run fMRIPrep on one downloaded ds005752 participant using fmriprep-docker.
#
# Requirements:
# - Docker Desktop installed and running
# - fmriprep-docker available in the active environment
# - A valid FreeSurfer license file
#
# Usage:
#   export FS_LICENSE=$HOME/license.txt
#   bash code/02_run_fmriprep_subset.sh
#   bash code/02_run_fmriprep_subset.sh sub-ON01016

SUBJECT="${1:-sub-ON01016}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIDS_DIR="${REPO_ROOT}/data/raw/ds005752"
OUT_DIR="${REPO_ROOT}/data/derivatives/fmriprep"
WORK_DIR="${REPO_ROOT}/data/work/fmriprep"

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

if ! command -v fmriprep-docker >/dev/null 2>&1; then
  echo "fmriprep-docker is not available. Install the conda environment first." >&2
  exit 1
fi

if [[ ! -d "${BIDS_DIR}/${SUBJECT}" ]]; then
  echo "Subject data not found at ${BIDS_DIR}/${SUBJECT}. Run the download script first." >&2
  exit 1
fi

if ! FS_LICENSE_FILE="$(find_fs_license)"; then
  echo "FreeSurfer license not found." >&2
  echo "Looked in these common locations:" >&2
  echo "  FS_LICENSE" >&2
  echo "  \$FREESURFER_HOME/license.txt" >&2
  echo "  \$HOME/license.txt" >&2
  echo "  \$HOME/Downloads/license.txt" >&2
  echo "  \$HOME/Desktop/license.txt" >&2
  echo "  /Applications/freesurfer/license.txt" >&2
  echo "Then set it manually, for example:" >&2
  echo "  export FS_LICENSE=\"/full/path/to/license.txt\"" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${WORK_DIR}"

fmriprep-docker "${BIDS_DIR}" "${OUT_DIR}" participant \
  --participant-label "${SUBJECT#sub-}" \
  --fs-license-file "${FS_LICENSE_FILE}" \
  --work-dir "${WORK_DIR}" \
  --output-spaces "MNI152NLin2009cAsym:res-2" "T1w" \
  --nthreads 4 \
  --omp-nthreads 4 \
  --mem_mb 16000 \
  --stop-on-first-crash

printf 'fMRIPrep finished for %s. Review %s/%s.html\n' \
  "${SUBJECT}" "${OUT_DIR}" "${SUBJECT}"
