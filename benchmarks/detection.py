from ultralytics import YOLO
import time
from utils import update_results

model = YOLO("yolov8n.pt")
img = "https://ultralytics.com/images/bus.jpg"

# Warm-up
_ = model(img)

# First run
start = time.time()
_ = model(img)
first_time = time.time() - start

# Steady state
start = time.time()
for _ in range(100):
    _ = model(img)
steady_time = time.time() - start
fps = 100 / steady_time

update_results("Detection", {
    "Images_Processed": 100,
    "First_Run_Sec": round(first_time, 4),
    "Steady_Total_Sec": round(steady_time, 4),
    "Steady_FPS": round(fps, 2)
})
