import torch
from ultralytics import YOLO


model = YOLO("/home/Patel/Dokumente/lightning-hydra-template/runs/detect/train/weights/best.pt")

results = model("Predict_conventional/testimageLeft_1.png")

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    confidence = result.boxes.conf.tolist()[0]
    result.show()
    print(boxes)