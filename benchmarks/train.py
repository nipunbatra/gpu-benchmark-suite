"""Training-throughput benchmark — ResNet50 full train step (forward + backward + optimizer),
mixed precision (AMP), single GPU. Reported as images/sec.

This is the counterpart to classification.py (which is inference-only / forward pass). Training
stresses the GPU very differently (backward pass, optimizer, more VRAM, tensor cores), so a
datacenter GPU's real advantage shows up here even when small-batch inference looks flat.
"""
import time
import torch
import torchvision
from utils import update_results

dev = "cuda:0"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BATCH = 32  # fits comfortably in 12 GB, so the same workload runs on every GPU
model = torchvision.models.resnet50(num_classes=1000).to(dev).train()
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
crit = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

x = torch.randn(BATCH, 3, 224, 224, device=dev)
y = torch.randint(0, 1000, (BATCH,), device=dev)


def step():
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        loss = crit(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


for _ in range(5):          # warm-up
    step()
torch.cuda.synchronize()

iters = 50
start = time.time()
for _ in range(iters):
    step()
torch.cuda.synchronize()
dur = time.time() - start

update_results("Training", {
    "Model": "resnet50",
    "Batch_Size": BATCH,
    "Steps": iters,
    "Images_per_Sec": round(iters * BATCH / dur, 1),
})
