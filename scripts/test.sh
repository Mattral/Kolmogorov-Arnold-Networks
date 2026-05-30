#!/usr/bin/env bash
# Run the full test suite locally (pytest + coverage).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src:$(pwd)"
exec pytest tests/ -v --tb=short --cov=src/kanx --cov-report=term-missing "$@"
