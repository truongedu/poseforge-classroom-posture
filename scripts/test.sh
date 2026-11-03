#!/usr/bin/env bash
# PoseForge regression: quick | full | ui
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/conda_common.sh"
cd "$ROOT"

# ROS Humble may leak incompatible Python packages/plugins into Conda.
unset PYTHONPATH PYTHONHOME PYTEST_ADDOPTS PYTEST_PLUGINS
export PYTHONPATH="$ROOT:$ROOT/collector"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1

mode="${1:-full}"
if [[ $# -gt 1 ]]; then
  echo 'Usage: bash scripts/test.sh [quick|full|ui]' >&2
  exit 2
fi
case "$mode" in
  quick)
    tests=(
      tests/test_phase31_benchmark.py
      tests/test_phase32.py
      collector/tests/test_feature_v2.py
      collector/tests/test_runtime_layout.py
      collector/tests/test_single_person_reliability.py
    )
    ;;
  full)
    tests=(tests collector/tests --ignore=collector/tests/test_ui.py)
    ;;
  ui)
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
    tests=(collector/tests/test_ui.py)
    ;;
  *)
    echo 'Usage: bash scripts/test.sh [quick|full|ui]' >&2
    exit 2
    ;;
esac
"$CONDA_CMD" run --no-capture-output -n "$POSEFORGE_CONDA_ENV" \
  python -B -m pytest -p no:cacheprovider -q "${tests[@]}"
