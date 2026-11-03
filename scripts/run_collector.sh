#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/conda_common.sh"
export PYTHONDONTWRITEBYTECODE=1
# Original module resolution: collector remains self-contained, no PYTHONPATH changes.
exec "$CONDA_CMD" run --no-capture-output -n "$POSEFORGE_CONDA_ENV" python -B "$ROOT/collector/main.py" "$@"
