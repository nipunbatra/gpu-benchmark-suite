from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")

images = ["https://ultralytics.com/images/bus.jpg"] * 10  # 10 images

start = time.time()
for img in images:
    results = model(img)
end = time.time()

num_images = len(images)
total_time = end - start
fps = num_images / total_time

print(f"Detection Benchmark Results:")
print(f"Images Processed: {num_images}")
print(f"Total Time (s): {total_time:.4f}")
print(f"FPS: {fps:.2f}")
