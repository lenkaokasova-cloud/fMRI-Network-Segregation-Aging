#!/usr/bin/env bash

# What this script does:
#   Downloads the selected ds005752 subjects from OpenNeuro into the local raw
#   BIDS folder.
# How to run it:
#   Run from the repo root with:
#   bash code/primary/05_download_openneuro_subjects.sh
# Main output:
#   Subject folders inside data/raw/ds005752/

set -euo pipefail

# Download selected ds005752 subjects with anat/, func/, and fmap/ if present.
#
# Usage:
#   bash code/primary/05_download_openneuro_subjects.sh
#   bash code/primary/05_download_openneuro_subjects.sh data/processed/screening/ds005752_mri_participants_age_20_25_remote_anat_forward.tsv
#   bash code/primary/05_download_openneuro_subjects.sh sub-ON01016 sub-ON39099

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET="ds005752"
TARGET_DIR="${REPO_ROOT}/data/raw/${DATASET}"
OPENNEURO_PYTHON="${OPENNEURO_PYTHON:-$(head -n 1 "$(command -v openneuro-py)" | sed 's/^#!//')}"
YOUNG_SELECTED_INPUT="${REPO_ROOT}/data/processed/screening/ds005752_mri_participants_age_20_25_preprocessing_primary.tsv"
YOUNG_FALLBACK_INPUT="${REPO_ROOT}/data/processed/screening/ds005752_mri_participants_age_20_25_remote_anat_forward.tsv"
OLDER_DEFAULT_INPUT="${REPO_ROOT}/data/processed/screening/ds005752_mri_participants_age_50_75_remote_anat_forward.tsv"

if [[ -f "${YOUNG_SELECTED_INPUT}" ]]; then
  # If I already fixed a younger preprocessing list, I want that to be the default download input.
  YOUNG_DEFAULT_INPUT="${YOUNG_SELECTED_INPUT}"
else
  YOUNG_DEFAULT_INPUT="${YOUNG_FALLBACK_INPUT}"
fi

DEFAULT_INPUTS=(
  "${YOUNG_DEFAULT_INPUT}"
  "${OLDER_DEFAULT_INPUT}"
)

mkdir -p "${TARGET_DIR}"

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

download_subject() {
  local subject="$1"

  if [[ -d "${TARGET_DIR}/${subject}" ]]; then
    printf 'Skipping %s because it already exists in %s\n' "${subject}" "${TARGET_DIR}"
    return
  fi

  printf '\nDownloading %s...\n' "${subject}"
  "${OPENNEURO_PYTHON}" - <<PY
from pathlib import Path

import openneuro._download as d
from tqdm.std import tqdm

d.tqdm = tqdm

dataset = "${DATASET}"
subject = "${subject}"
target_dir = Path("${TARGET_DIR}")
includes = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    f"{subject}/ses-01/anat/*",
    f"{subject}/ses-01/func/*",
]

try:
    # I ask for fmap too when it exists, but I do not want the whole download to fail if it does not.
    d.download(
        dataset=dataset,
        target_dir=target_dir,
        include=includes + [f"{subject}/ses-01/fmap/*"],
    )
except RuntimeError as exc:
    if "Could not find path in the dataset" not in str(exc) or "fmap" not in str(exc):
        raise
    print(f"No fmap/ directory found for {subject}; retrying with anat/ and func/ only.")
    d.download(
        dataset=dataset,
        target_dir=target_dir,
        include=includes,
    )
PY
}

if [[ "$#" -gt 0 ]]; then
  INPUTS=("$@")
else
  INPUTS=("${DEFAULT_INPUTS[@]}")
fi

SUBJECTS=()
for input in "${INPUTS[@]}"; do
  while IFS= read -r subject; do
    [[ -z "${subject}" ]] && continue
    # This keeps the download queue unique even if the same subject shows up twice across inputs.
    append_unique_subject "${subject}"
  done < <(collect_subjects_from_arg "${input}")
done

if [[ "${#SUBJECTS[@]}" -eq 0 ]]; then
  echo "No subjects found to download." >&2
  exit 1
fi

for subject in "${SUBJECTS[@]}"; do
  download_subject "${subject}"
done

printf '\nFinished downloading %s requested subjects into %s\n' "${#SUBJECTS[@]}" "${TARGET_DIR}"
