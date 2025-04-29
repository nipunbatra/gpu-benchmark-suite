# GPU Benchmark Suite

Benchmarks:
- LLM Inference (HuggingFace Transformers)
- Object Detection (YOLOv8)
- Image Classification (ResNet)
- GPU Burn (optional stress test)

## Setup

```bash
sudo apt update
sudo apt install -y docker.io nvidia-container-toolkit
sudo systemctl restart docker
```

Test GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
```

## Run

```bash
git clone <your-repo-url>
cd gpu-benchmark-suite
docker build -t gpu-bench -f docker/Dockerfile .
docker run --rm --gpus all -v $(pwd)/benchmarks:/workspace/benchmarks -w /workspace gpu-bench python benchmarks/llm.py
docker run --rm --gpus all -v $(pwd)/benchmarks:/workspace/benchmarks -w /workspace gpu-bench python benchmarks/detection.py
docker run --rm --gpus all -v $(pwd)/benchmarks:/workspace/benchmarks -w /workspace gpu-bench python benchmarks/classification.py
```

Optional GPU Burn:

```bash
docker run --rm --gpus all -v $(pwd)/benchmarks:/workspace/benchmarks -w /workspace gpu-bench bash benchmarks/burn_gpu.sh
```