#!/usr/bin/env bash
# Single source of truth for the local Conda environment name.
# Override without modifying the project: POSEFORGE_CONDA_ENV=my_env bash scripts/...
POSEFORGE_CONDA_ENV="${POSEFORGE_CONDA_ENV:-lab2_imp}"

if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
  CONDA_CMD="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
  CONDA_CMD="$(command -v conda)"
else
  echo 'Conda not found. Install Miniconda/Anaconda and open a terminal with conda available.' >&2
  exit 2
fi
