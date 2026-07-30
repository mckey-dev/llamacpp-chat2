#!/usr/bin/env bash
set -euo pipefail
export LANG=C.UTF-8
export PYTHONUTF8=1
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -x .venv-server/bin/python ]]; then
  echo "Creating .venv-server ..."
  python3 -m venv .venv-server
fi

# shellcheck disable=SC1091
source .venv-server/bin/activate
# requirements-server.txt は追加パッケージなし（コメントのみ）

export LLAMACPP_CHAT2_CONTROL_TOKEN="${LLAMACPP_CHAT2_CONTROL_TOKEN:-llamacpp-chat2}"

echo "Starting control API ..."
python -m server.control_api
