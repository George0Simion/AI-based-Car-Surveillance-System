import sys
import os
import heapq
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# PARSE COMMAND-LINE ARGUMENTS
# ---------------------------
if len(sys.argv) != 4:
    print("Usage: python3 video.py <model_checkpoint> <video_file> <data_folder>")
    sys.exit(1)

model_checkpoint = sys.argv[1]
video_file = sys.argv[2]
path = sys.argv[3]

print(f"Using dataset path: {path}")


# ---------------------------
# BUILD brand_names
# ---------------------------
directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
data = {directory: i for i, directory in enumerate(directories)}
brand_names = {v: k for k, v in data.items()}


# ---------------------------
# LOAD THE MODEL
# ---------------------------
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
model.fc = torch.nn.Linear(in_features=2048, out_features=len(directories))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_checkpoint, map_location=device))
model.eval()
model.to(device)


# ---------------------------
# INFERENCE TRANSFORM
# ---------------------------
eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ---------------------------
# YOLOv5 (for car detection)
# ---------------------------
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_yolo.to(device)


# ---------------------------
# DETECT CARS
# ---------------------------
def detect_cars(frame):
    results = model_yolo(frame)
    detections = []
    for *xyxy, conf, cls_id in results.xyxy[0]:
        class_name = results.names[int(cls_id)]
        if class_name == 'car' and conf > 0.5:   # Confidence threshold
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append((x1, y1, x2, y2, float(conf)))
    return detections


# ---------------------------
# CLASSIFY CAR
# ---------------------------
def classify_car(car_crop, model_brand, transform, device):
    rgb_crop = cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB)
    tensor_crop = transform(rgb_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model_brand(tensor_crop)
        probs = F.softmax(output, dim=1)
        top_prob, predicted = torch.max(probs, 1)
        top_prob = top_prob.item()
        brand_id = predicted.item()

    if top_prob < 0.4:
        return None
    return brand_id


# ---------------------------
# PROCESS FRAME
# ---------------------------
def process_frame(frame):
    detections = detect_cars(frame)
    for x1, y1, x2, y2, conf in detections:
        car_crop = frame[y1:y2, x1:x2]
        brand_id = classify_car(car_crop, model, eval_transform, device)
        if brand_id is not None:
            brand_name = brand_names.get(brand_id, 'Unknown')

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, brand_name, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return frame


# ---------------------------
# MAIN: READ VIDEO AND RUN INFERENCE
# ---------------------------
def main():
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video file:", video_file)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or can't read frame.")
            break

        output_frame = process_frame(frame)
        cv2.imshow('Car Brand Identifier', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
