#!/usr/bin/env bash
set -euo pipefail
export LANG=C.UTF-8
export PYTHONUTF8=1
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -x .venv-frontend/bin/python ]]; then
  echo "Creating .venv-frontend ..."
  python3 -m venv .venv-frontend
fi

# shellcheck disable=SC1091
source .venv-frontend/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-frontend.txt
export GRADIO_SHARE="${GRADIO_SHARE:-False}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
echo "Starting frontend UI ..."
python -m ui.app
