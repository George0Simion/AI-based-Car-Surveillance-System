import os
import numpy as np
import cv2
import pandas as pd
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Objective -> Car Recognition


# -----------------------------
# PATHS & DIRECTORY HANDLING
# -----------------------------
path = "/home/simion/Desktop/AI/AI-based-Car-Surveillance-System/data"


# -----------------------------
# CONFIG AND HYPERPARAMETERS
# -----------------------------
TARGET_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
STEP_SIZE = 7
GAMMA = 0.1

directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
data = {directory: i for i, directory in enumerate(directories)}
brand_names = {v: k for k, v in data.items()} # Inversam dictionarul pt a obtine numele brandurilor dupa label


# -----------------------------
# MODEL
# -----------------------------
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
model.fc = torch.nn.Linear(in_features=2048, out_features=len(directories))
model.load_state_dict(torch.load('CarBrandIdentifier_resnet50.pth', map_location='cpu'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# -----------------------------
# INTERFERENCE TRANSFORM
# -----------------------------
eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -----------------------------
# YOLOv5 used for car detection
# -----------------------------
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_yolo.to('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------
# DETECT CARS
# -----------------------------
def detect_cars(frame):
    results = model_yolo(frame)
    
    detections = []
    for *xyxy, conf, cls_id in results.xyxy[0]:
        class_name = results.names[int(cls_id)]
        if class_name == 'car' and conf > 0.5:   # conf threshold
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append((x1, y1, x2, y2, float(conf)))

    return detections # returns a list of all the bounding boxes in the frams

def classify_cars(car_crop, model_brand, transform, device):
    rgb_crop = cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB)
    tensor_crop = transform(rgb_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model_brand(tensor_crop)
        _, predicted = torch.max(output, 1)
        label_id = predicted.item()

    return label_id

def process_frame(frame, model_brand, transform, device, brand_names):
    detections = detect_cars(frame)

    for x1, y1, x2, y2, conf in detections:
        car_crop = frame[y1:y2, x1:x2]
        brand_id = classify_cars(car_crop, model_brand, transform, device)
        brand_name = brand_names.get(brand_id, 'Unknown')

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{brand_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# -----------------------------
# VIDEO CAPTURE
# -----------------------------
def live_video_interferance():
    cap = cv2.VideoCapture("myvideo.mp4")
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?).")
            break

        output_frame = process_frame(frame, model, eval_transform, device, brand_names)

        cv2.imshow('Car Brand Identifier', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    live_video_interferance()
