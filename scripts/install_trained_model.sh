#!/usr/bin/env bash
# Usage: bash scripts/install_trained_model.sh [ZIP]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if (( $# > 1 )); then echo 'Usage: bash scripts/install_trained_model.sh [ZIP]' >&2; exit 2; fi
if (( $# )); then
  ZIP="$1"
else
  shopt -s nullglob
  files=("${DOWNLOAD_DIR:-$HOME/Downloads}"/final_model_bundle*.zip "${DOWNLOAD_DIR:-$HOME/Downloads}"/gru_final_model_bundle*.zip)
  (( ${#files[@]} )) || { echo '[ERROR] No MLP/GRU final model bundle in Downloads' >&2; exit 1; }
  ZIP="${files[0]}"
  for f in "${files[@]:1}"; do [[ "$f" -nt "$ZIP" ]] && ZIP="$f"; done
fi
[[ -f "$ZIP" ]] || { echo "[ERROR] Not found: $ZIP" >&2; exit 1; }

python - "$ROOT" "$ZIP" <<'PY'
import hashlib, io, json, re, shutil, sys, zipfile
from pathlib import Path
from datetime import datetime, timezone
import torch

root, src = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
required = ('model_final.pt', 'final_summary.json', 'feature_contract.json')
with zipfile.ZipFile(src) as z:
    missing = set(required) - set(z.namelist())
    if missing: sys.exit(f'[ERROR] Missing files: {sorted(missing)}')
    if bad := z.testzip(): sys.exit(f'[ERROR] Corrupt ZIP: {bad}')

    ckpt = torch.load(io.BytesIO(z.read('model_final.pt')), map_location='cpu', weights_only=True)
    name = ckpt.get('model_name', '')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', name):
        sys.exit('[ERROR] Invalid model_name in checkpoint')
    if ckpt.get('stage') != 'final' or not ckpt.get('model_state_dict'):
        sys.exit('[ERROR] This is not a FINAL checkpoint')
    contract = json.loads(z.read('feature_contract.json'))
    if ckpt.get('feature_contract') != contract or not ckpt.get('label_names'):
        sys.exit('[ERROR] Checkpoint contract / labels invalid')
    summary = json.loads(z.read('final_summary.json'))
    optional = ['experiment_config.json'] if 'experiment_config.json' in z.namelist() else []
    for name in optional: json.loads(z.read(name))

    base = root / 'artifacts/models' / name
    base.mkdir(parents=True, exist_ok=True)
    versions = [int(p.name[1:]) for p in base.iterdir() if p.is_dir() and re.fullmatch(r'v\d+', p.name)]
    version = max(versions, default=0) + 1
    while True:
        dest = base / f'v{version:03d}'
        try: dest.mkdir(); break  # Atomic: never overwrite.
        except FileExistsError: version += 1

    try:
        code_files = [p for p in z.namelist()
                      if re.fullmatch(r'shared/[A-Za-z_][A-Za-z0-9_]*\.py', p)]
        for filename in (*required, *optional, *code_files):
            out = dest / filename
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(filename) as source, out.open('xb') as target:
                shutil.copyfileobj(source, target)
        with src.open('rb') as zipfile_handle:
            sha = hashlib.file_digest(zipfile_handle, 'sha256').hexdigest()
        (dest / 'install_info.json').write_text(json.dumps({
            'model_name': name, 'candidate': summary.get('candidate'),
            'feature_version': contract.get('version'), 'zip_name': src.name,
            'zip_sha256': sha, 'installed_utc': datetime.now(timezone.utc).isoformat()
        }, indent=2) + '\n')
    except Exception:
        shutil.rmtree(dest)
        raise

print(f'[OK] {name} | candidate={summary.get("candidate")} | {dest.relative_to(root)}')
print(f'[OK] Feature contract: {contract.get("version")}')
for filename in code_files:
    local = root / filename
    if local.is_file() and local.read_bytes() != (dest / filename).read_bytes():
        print(f'[WARN] {filename} differs; existing source NOT overwritten')
PY
