#!/bin/bash
set -e

mkdir -p results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="results/benchmark_summary_${TIMESTAMP}.txt"

echo "Running Benchmarks..." | tee -a $RESULT_FILE

echo -e "\n=== LLM Benchmark ===" | tee -a $RESULT_FILE
docker run --rm --gpus all -v $(pwd)/benchmarks:/workspace/benchmarks -w /workspace gpu-bench python benchmarks/llm.py | tee -a $RESULT_FILE

echo -e "\n=== Object Detection Benchmark ===" | tee -a $RESULT_FILE
docker run --rm --gpus all -v $(pwd)/benchmarks:/workspace/benchmarks -w /workspace gpu-bench python benchmarks/detection.py | tee -a $RESULT_FILE

echo -e "\n=== Classification Benchmark ===" | tee -a $RESULT_FILE
docker run --rm --gpus all -v $(pwd)/benchmarks:/workspace/benchmarks -w /workspace gpu-bench python benchmarks/classification.py | tee -a $RESULT_FILE

echo -e "\nAll benchmarks completed."
echo "Results saved in: $RESULT_FILE"
