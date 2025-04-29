import torch, torchvision, time
model = torchvision.models.resnet50(weights='IMAGENET1K_V1').cuda().eval()
dummy = torch.randn(32, 3, 224, 224).cuda()
start = time.time()
with torch.no_grad():
    for _ in range(100): model(dummy)
print(f"Avg time/batch: {(time.time() - start)/100:.4f}s")