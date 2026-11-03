#!/usr/bin/env bash
# Run from anywhere: bash scripts/setup_ubuntu.sh [--pose] [--no-install]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/conda_common.sh"
INSTALL=1; POSE=0
for arg in "$@"; do
  case "$arg" in
    --pose) POSE=1 ;;
    --no-install) INSTALL=0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
cd "$ROOT"
if [[ ! -e "$ROOT/.git" ]]; then
  git init -b main
  echo '[OK] Local Git initialized (no remote configured).'
else
  echo '[OK] Existing local Git preserved.'
fi

# Keep the environment in Conda's env directory, not in the project root.
if "$CONDA_CMD" run -n "$POSEFORGE_CONDA_ENV" python -c 'import sys' >/dev/null 2>&1; then
  echo "[OK] Conda environment preserved: $POSEFORGE_CONDA_ENV"
else
  "$CONDA_CMD" create --yes --name "$POSEFORGE_CONDA_ENV" python=3.11 pip
  echo "[OK] Conda environment created: $POSEFORGE_CONDA_ENV"
fi
if [[ "$INSTALL" == 1 ]]; then
  "$CONDA_CMD" run --no-capture-output -n "$POSEFORGE_CONDA_ENV" python -m pip install --no-cache-dir --upgrade pip
  if [[ "$POSE" == 1 ]]; then
    "$CONDA_CMD" run --no-capture-output -n "$POSEFORGE_CONDA_ENV" python -m pip install --no-cache-dir -r requirements.txt
  else
    "$CONDA_CMD" run --no-capture-output -n "$POSEFORGE_CONDA_ENV" python -m pip install --no-cache-dir -r collector/requirements.txt pytest
  fi
fi
printf '\nNext:\n  bash scripts/test.sh full\n  bash scripts/run_collector.sh\n'
echo "Conda env: $POSEFORGE_CONDA_ENV (use: conda activate $POSEFORGE_CONDA_ENV)"
echo 'Before first commit, inspect: git status --short'
if [[ -z "$(git config user.name || :)" || -z "$(git config user.email || :)" ]]; then
  echo 'To commit, first configure local Git user.name and user.email for this repository.'
fi
