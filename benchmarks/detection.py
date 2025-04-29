from ultralytics import YOLO
import time
import yaml

model = YOLO("yolov8n.pt")
images = ["https://ultralytics.com/images/bus.jpg"] * 10

start = time.time()
for img in images:
    results = model(img)
end = time.time()

num_images = len(images)
total_time = end - start
fps = num_images / total_time

result = {
    "Detection": {
        "Images_Processed": num_images,
        "Total_Time_Sec": round(total_time, 4),
        "FPS": round(fps, 2)
    }
}

print(yaml.dump(result))
