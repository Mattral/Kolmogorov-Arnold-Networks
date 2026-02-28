#!/usr/bin/env bash
# Train kanx on the default config (synthetic regression).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
exec python -m kanx train --config "${1:-configs/default.yaml}"
