# GPU Benchmark Suite

One Docker image, the **same ML workloads on every server**, results tagged by host so
you can compare them directly. Built for the lab's Ramanujan / Bhaskar / Sustain servers
(and the Aryabhata / Vikram workstations once they're on the network).

Benchmarks:
- ✅ **LLM inference** (HuggingFace Transformers — OPT-1.3B) → tokens/sec
- ✅ **Object detection** (YOLOv8n) → FPS
- ✅ **Image classification** (ResNet50) → images/sec
- ✅ **Storage / data-loading** (seq read/write + DataLoader images/sec) → catches data-starved training
- 🟡 **GPU burn** (optional stress test — `benchmarks/burn_gpu.sh`)

The workloads are deliberately light (fit in 12 GB VRAM) so the **same test runs unchanged on
an A100 80 GB and a 12 GB workstation**, keeping the numbers comparable.

---

## Setup (once per server)

```bash
sudo apt update
sudo apt install -y docker.io nvidia-container-toolkit
sudo systemctl restart docker

# verify GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
```

## Run

```bash
git clone https://github.com/nipunbatra/gpu-benchmark-suite
cd gpu-benchmark-suite
./run.sh                      # builds the image on first run, then benchmarks
```

Results land in `results/<hostname>.yaml` (host + GPU + CUDA/torch versions recorded
automatically). Re-run on each server.

**Benchmark a specific storage mount** (e.g. compare the slow HDD `/home` vs a new SSD, or the NAS):

```bash
BENCH_DATA_DIR=/home/you ./run.sh
BENCH_DATA_DIR=/mnt/nas   ./run.sh
```

Useful env vars: `BENCH_DATA_DIR` (storage path), `BENCH_WORKERS` (DataLoader workers),
`BENCH_N_IMAGES`, `BENCH_SEQ_MB`, `BENCH_IMAGE` (image tag).

## Compare across servers

Copy each server's `results/*.yaml` into one `results/` folder, then:

```bash
pip install pyyaml
python aggregate.py
```

This prints a comparison table and writes:
- `assets/benchmarks.svg` — crisp grouped bar chart (speedup vs slowest server)
- `assets/benchmarks.html` — raw-number table, ready to paste into the Quarto resources page

## Example `results/<host>.yaml`

```yaml
Meta:
  host: ramanujan
  timestamp: 2026-06-15T13:40:00
  gpu: NVIDIA A100-SXM4-80GB
  gpu_count: 4
  vram_gb_each: 85.9
  torch: 2.2.0
  cuda: '12.2'
LLM:
  Steady_Tokens_per_Sec: 69.78
Detection:
  Steady_FPS: 34.36
Classification:
  Images_per_Sec: 1340.0
Storage:
  Path: /scratch
  Seq_Write_MBps: 1850.0
  DataLoader_Images_per_Sec: 980.0
```
