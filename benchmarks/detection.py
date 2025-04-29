from ultralytics import YOLO
import time
model = YOLO("yolov8n.pt")
start = time.time()
for _ in range(10):
    results = model("https://ultralytics.com/images/bus.jpg")
print(f"Avg time/image: {(time.time() - start)/10:.4f}s")