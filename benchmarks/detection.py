"""Object-detection benchmark.

Uses torchvision's Faster R-CNN (ResNet50-FPN) so the only dependencies are torch +
torchvision, both shipped (ABI-matched) in the NGC container. This deliberately avoids
ultralytics/opencv, whose prebuilt opencv conflicts with the container (cv2.dnn.DictValue).
"""
import time
import torch
import torchvision
from utils import update_results

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").cuda().eval()
img = [torch.rand(3, 640, 640).cuda()]

with torch.no_grad():
    for _ in range(3):  # warm-up
        _ = model(img)
    torch.cuda.synchronize()

    start = time.time()
    _ = model(img)
    torch.cuda.synchronize()
    first_time = time.time() - start

    start = time.time()
    for _ in range(100):
        _ = model(img)
    torch.cuda.synchronize()
    steady_time = time.time() - start

fps = 100 / steady_time
update_results("Detection", {
    "Model": "fasterrcnn_resnet50_fpn",
    "Images_Processed": 100,
    "First_Run_Sec": round(first_time, 4),
    "Steady_Total_Sec": round(steady_time, 4),
    "Steady_FPS": round(fps, 2),
})
