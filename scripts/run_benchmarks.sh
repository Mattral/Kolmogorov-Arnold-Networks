#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python benchmarks/compare_mlp.py --long
python benchmarks/real_world.py

mkdir -p benchmarks/results
cp benchmarks/results.md benchmarks/results/ || true
cp benchmarks/results/real_world_results.json benchmarks/results/ || true

echo "Benchmark suite complete."
