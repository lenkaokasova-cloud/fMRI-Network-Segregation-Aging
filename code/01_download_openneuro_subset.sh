#!/bin/bash

# Download a small subset of OpenNeuro ds005752 for preprocessing demonstration.
# Replace SUB with the participant you choose from the OpenNeuro browser.

DATASET="ds005752"
TARGET_DIR="$HOME/Desktop/fMRI-Network-Segregation-Aging/data/raw/ds005752"
SUB="sub-ON52083"

mkdir -p "$TARGET_DIR"

openneuro-py download \
  --dataset="$DATASET" \
  --target-dir="$TARGET_DIR" \
  --include=dataset_description.json \
  --include=participants.tsv \
  --include=participants.json \
  --include=${SUB}/ses-01/anat/* \
  --include=${SUB}/ses-01/func/* \
  --include=${SUB}/ses-01/fmap/*

echo "Download complete for $SUB"