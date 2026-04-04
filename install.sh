#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-dis}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

python -m pip install -r requirements.txt
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. board.proto

echo "Setup complete. Generated board_pb2.py and board_pb2_grpc.py"