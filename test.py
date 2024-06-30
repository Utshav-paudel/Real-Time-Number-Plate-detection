import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
from ultralytics import YOLO

coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

coco_model.to(device)