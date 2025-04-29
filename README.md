# GPU Benchmark Suite 

This suite benchmarks:
- ✅ **LLM Inference** (HuggingFace Transformers — OPT 1.3B)
- ✅ **Object Detection** (YOLOv8n)
- ✅ **Image Classification** (ResNet50)
- 🟡 **GPU Burn** (optional stress test)

Each benchmark reports:
- First run (cold start) latency and throughput
- Steady-state throughput (tokens/sec, FPS, images/sec)
- Results saved to a single clean YAML file: `results/final.yaml`

---

## ✅ Setup

```bash
sudo apt update
sudo apt install -y docker.io nvidia-container-toolkit
sudo systemctl restart docker
```

## Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
```     

## Build the Docker Image

```bash
git clone https://github.com/nipunbatra/gpu-benchmark-suite
cd gpu-benchmark-suite
docker build -t gpu-bench -f docker/Dockerfile .
```

## Run the Benchmarks

```bash
bash benchmarks/run_all_benchmarks.sh
```

## Example Output

```yaml
LLM:
  Generated_Tokens: 512
  First_Run_Sec: 12.43
  First_Tokens_per_Sec: 41.18
  Steady_Avg_Sec: 7.34
  Steady_Tokens_per_Sec: 69.78
Detection:
  Images_Processed: 100
  First_Run_Sec: 0.48
  Steady_Total_Sec: 2.91
  Steady_FPS: 34.36
Classification:
  Batch_Size: 32
  Batches_Processed: 100
  First_Batch_Sec: 0.025
  Steady_Total_Sec: 2.55
  Avg_Batch_Sec: 0.0255
  Images_per_Sec: 1254.90
```