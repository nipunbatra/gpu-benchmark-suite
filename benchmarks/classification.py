import torch, torchvision, time
import yaml

model = torchvision.models.resnet50(weights='IMAGENET1K_V1').cuda().eval()
dummy = torch.randn(32, 3, 224, 224).cuda()

start = time.time()
with torch.no_grad():
    for _ in range(100):
        model(dummy)
end = time.time()

batches = 100
total_time = end - start
avg_batch_time = total_time / batches
images_per_sec = (32 * batches) / total_time

result = {
    "Classification": {
        "Batches_Processed": batches,
        "Total_Time_Sec": round(total_time, 4),
        "Avg_Batch_Time_Sec": round(avg_batch_time, 4),
        "Images_per_Sec": round(images_per_sec, 2)
    }
}

print(yaml.dump(result))
