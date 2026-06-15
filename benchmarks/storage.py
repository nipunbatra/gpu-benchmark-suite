"""Storage / data-loading benchmark.

Measures the disk path that actually feeds training:
  - sequential write throughput (fsync'd)
  - sequential read throughput (cache-influenced; labelled as such)
  - realistic dataset loading: small JPEGs read through a torch DataLoader (images/sec)

Point it at different mounts to compare them, e.g.:
  BENCH_DATA_DIR=/home/...   (slow HDD vs new SSD)
  BENCH_DATA_DIR=/mnt/nas    (network storage)

The DataLoader number is the one that matters for "why is my training data-starved" —
it is the small-file random-read + decode pattern, the exact thing the cache="ram"
workaround was papering over.
"""
import os
import time
import shutil

from utils import update_results

DATA_DIR = os.environ.get("BENCH_DATA_DIR", "./bench_scratch")
N_IMAGES = int(os.environ.get("BENCH_N_IMAGES", "2000"))
WORKERS = int(os.environ.get("BENCH_WORKERS", "8"))
SEQ_MB = int(os.environ.get("BENCH_SEQ_MB", "2048"))


def seq_write(path, size_mb, chunk_mb=8):
    chunk = os.urandom(chunk_mb * 1024 * 1024)
    n = size_mb // chunk_mb
    start = time.time()
    with open(path, "wb") as f:
        for _ in range(n):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    return round((n * chunk_mb) / (time.time() - start), 1)


def seq_read(path, chunk_mb=8):
    size = os.path.getsize(path)
    start = time.time()
    with open(path, "rb") as f:
        while f.read(chunk_mb * 1024 * 1024):
            pass
    return round(size / 1e6 / (time.time() - start), 1)


def make_images(folder, n):
    from PIL import Image
    import numpy as np
    cls = os.path.join(folder, "class0")
    os.makedirs(cls, exist_ok=True)
    for i in range(n):
        arr = (np.random.rand(224, 224, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(os.path.join(cls, f"img_{i}.jpg"), quality=90)


def dataloader_speed(folder):
    import torchvision as tv
    from torch.utils.data import DataLoader
    ds = tv.datasets.ImageFolder(
        folder,
        transform=tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
        ]),
    )
    dl = DataLoader(ds, batch_size=64, num_workers=WORKERS, shuffle=True)
    for _ in dl:  # warm-up (spawns workers, fills caches)
        break
    start, count = time.time(), 0
    for x, _ in dl:
        count += x.size(0)
    return round(count / (time.time() - start), 1)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    big = os.path.join(DATA_DIR, "_seqfile.bin")
    write_mbps = seq_write(big, SEQ_MB)
    read_mbps = seq_read(big)
    os.remove(big)

    img_dir = os.path.join(DATA_DIR, "_imgs")
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    make_images(img_dir, N_IMAGES)
    dl_ips = dataloader_speed(img_dir)
    shutil.rmtree(img_dir)

    update_results("Storage", {
        "Path": os.path.abspath(DATA_DIR),
        "Seq_Write_MBps": write_mbps,
        "Seq_Read_MBps_cached": read_mbps,
        "DataLoader_Images_per_Sec": dl_ips,
        "DataLoader_Workers": WORKERS,
        "Num_Images": N_IMAGES,
    })


if __name__ == "__main__":
    main()
