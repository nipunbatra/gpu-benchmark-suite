#!/usr/bin/env bash
# Uniform ML benchmark — same Docker image, same workloads, every server.
# Results are tagged by host in results/<hostname>.yaml so they can be compared.
#
# Usage:
#   ./run.sh                       # run everything with defaults
#   BENCH_DATA_DIR=/home/me ./run.sh   # benchmark storage on a specific mount
#
# Then collect results/*.yaml from each server onto one machine and run:
#   python aggregate.py            # -> console table + assets/benchmarks.svg + assets/benchmarks.html
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${BENCH_IMAGE:-gpu-bench}"
HOST="$(hostname)"
TS="$(date -Iseconds)"
DATA_DIR="${BENCH_DATA_DIR:-/tmp/gpu-bench-scratch}"

echo "==> host=$HOST  image=$IMAGE  storage-path=$DATA_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> image '$IMAGE' not found — building (first run only, downloads a few GB)..."
  docker build -t "$IMAGE" -f docker/Dockerfile .
fi

mkdir -p results "$DATA_DIR" "$HOME/.cache"

COMMON=(
  --rm --gpus all
  --shm-size=8g
  -e BENCH_HOST="$HOST"
  -e BENCH_TIMESTAMP="$TS"
  -e BENCH_DATA_DIR=/scratch
  -v "$(pwd)/benchmarks:/workspace/benchmarks"
  -v "$(pwd)/results:/workspace/results"
  -v "$DATA_DIR:/scratch"
  -v "$HOME/.cache:/root/.cache"   # persist HF/torch model downloads across runs
  -w /workspace
  "$IMAGE"
)

for b in llm detection classification storage; do
  echo "==> running $b ..."
  docker run "${COMMON[@]}" python "benchmarks/$b.py" || echo "!! $b failed — continuing"
done

echo "==> done -> results/$HOST.yaml"
