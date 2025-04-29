import torch, torchvision, time
from utils import update_results

model = torchvision.models.resnet50(weights='IMAGENET1K_V1').cuda().eval()
dummy = torch.randn(32, 3, 224, 224).cuda()

# Warm-up
for _ in range(5):
    _ = model(dummy)

# First batch
start = time.time()
_ = model(dummy)
first_time = time.time() - start

# Steady state
start = time.time()
for _ in range(100):
    _ = model(dummy)
steady_time = time.time() - start
avg_batch_time = steady_time / 100
img_per_sec = (100 * 32) / steady_time

update_results("Classification", {
    "Batch_Size": 32,
    "Batches_Processed": 100,
    "First_Batch_Sec": round(first_time, 4),
    "Steady_Total_Sec": round(steady_time, 4),
    "Avg_Batch_Sec": round(avg_batch_time, 4),
    "Images_per_Sec": round(img_per_sec, 2)
})
