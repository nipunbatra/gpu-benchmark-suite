import torch, torchvision, time

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

print(f"Classification Benchmark Results:")
print(f"Batches Processed: {batches}")
print(f"Total Time (s): {total_time:.4f}")
print(f"Avg Batch Time (s): {avg_batch_time:.4f}")
print(f"Images per Second: {images_per_sec:.2f}")
