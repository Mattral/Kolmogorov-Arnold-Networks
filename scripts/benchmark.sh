#!/usr/bin/env bash
# Run KAN vs MLP benchmark and refresh benchmarks/results.md
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
exec python benchmarks/compare_mlp.py
