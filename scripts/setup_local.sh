#!/usr/bin/env bash
# Set up homesim in a local virtualenv and verify it works.
#   bash scripts/setup_local.sh          # engine + tests
#   bash scripts/setup_local.sh --train  # also install torch/transformers/peft (needs a GPU to be useful)
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10 or newer, or set PYTHON=/path/to/python" >&2
  exit 1
fi

VERSION=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$VERSION" in
  3.1[0-9]|3.[2-9][0-9]) ;;
  *) echo "Python $VERSION found; homesim needs 3.10 or newer." >&2; exit 1 ;;
esac

echo "==> creating virtualenv in .venv (Python $VERSION)"
"$PY" -m venv .venv
VENV_PY=.venv/bin/python
[ -x "$VENV_PY" ] || VENV_PY=.venv/Scripts/python.exe   # Git Bash on Windows

echo "==> installing homesim"
"$VENV_PY" -m pip install --quiet --upgrade pip
if [ "${1:-}" = "--train" ]; then
  "$VENV_PY" -m pip install -e ".[test,train]"
else
  "$VENV_PY" -m pip install -e ".[test]"
fi

echo "==> running the test suite"
"$VENV_PY" -m pytest -q

echo "==> generating a small sample dataset"
"$VENV_PY" scripts/generate_data.py --train-houses 20 --per-house 4 --eval-houses 8 --eval-per-house 2

echo "==> checking the oracle scores perfectly on it"
"$VENV_PY" scripts/run_eval.py --policy oracle --data data/eval_in_dist.jsonl | tail -n 4

cat <<'DONE'

Setup complete.

  Activate:      source .venv/bin/activate      (Windows: .venv\Scripts\activate)
  Live UI:       python scripts/serve_ui.py     then open http://127.0.0.1:8000
  Full dataset:  python scripts/generate_data.py --train-houses 400 --per-house 6

DONE
