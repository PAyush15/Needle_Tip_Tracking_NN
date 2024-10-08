from ultralytics import YOLO

# Path to your config.yaml file
config_path = 'src/models/YOLOv8/config_right.yaml'


model = YOLO('yolov8n.yaml')

results = model.train(data=config_path, epochs=10, imgsz=640)
