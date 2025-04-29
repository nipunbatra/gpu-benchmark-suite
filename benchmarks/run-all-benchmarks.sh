mkdir -p results
rm -f results/final.yaml   # Remove old one if exists

docker run --rm --gpus all -v $(pwd)/benchmarks:/workspace/benchmarks -v $(pwd)/results:/workspace/results -w /workspace gpu-bench python benchmarks/llm.py
docker run --rm --gpus all -v $(pwd)/benchmarks:/workspace/benchmarks -v $(pwd)/results:/workspace/results -w /workspace gpu-bench python benchmarks/detection.py
docker run --rm --gpus all -v $(pwd)/benchmarks:/workspace/benchmarks -v $(pwd)/results:/workspace/results -w /workspace gpu-bench python benchmarks/classification.py

echo "Final combined result: results/final.yaml"
