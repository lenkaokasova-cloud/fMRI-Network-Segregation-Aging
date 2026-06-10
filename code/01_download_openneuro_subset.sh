#!/usr/bin/env bash

set -euo pipefail

# Download a minimal MRI-only subset of OpenNeuro ds005752 for a first
# preprocessing demonstration with fMRIPrep.
#
# Usage:
#   bash code/01_download_openneuro_subset.sh
#   bash code/01_download_openneuro_subset.sh sub-ON01016

DATASET="ds005752"
SUBJECT="${1:-sub-ON01016}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/data/raw/${DATASET}"

mkdir -p "${TARGET_DIR}"

openneuro-py download \
  --dataset="${DATASET}" \
  --target_dir="${TARGET_DIR}" \
  --include="dataset_description.json" \
  --include="participants.tsv" \
  --include="participants.json" \
  --include="${SUBJECT}/ses-01/anat/*" \
  --include="${SUBJECT}/ses-01/func/*" \
  --include="${SUBJECT}/ses-01/fmap/*"

printf 'Download complete for %s into %s\n' "${SUBJECT}" "${TARGET_DIR}"
