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
import math

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
# GLOBAL TRACKING LIST
# ---------------------------
tracked_cars = []
car_id_counter = 0

DISTANCE_THRESHOLD = 60     # max distance to consider a match
MAX_MISSED_FRAMES = 10      # how many frames to allow without an update


# ---------------------------
# DETECT CARS
# ---------------------------
def detect_cars(frame):
    """Use YOLOv5 to detect cars in the frame."""
    results = model_yolo(frame)
    detections = []
    for *xyxy, conf, cls_id in results.xyxy[0]:
        class_name = results.names[int(cls_id)]
        if class_name == 'car' and conf > 0.5:   # confidence threshold
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
        probs = F.softmax(output, dim=1).squeeze()  # shape = (num_classes,)

    probs_np = probs.cpu().numpy() # convert to numpy array
    heap = [(-p, idx) for idx, p in enumerate(probs_np)] # build a max-heap (negate probabilities since heapq is min-heap)
    heapq.heapify(heap)

    top_neg_prob, brand_id = heapq.heappop(heap)
    top_prob = -top_neg_prob

    return brand_id, top_prob


# ---------------------------
# EUCLIDEAN DISTANCE
# ---------------------------
def euclidean_distance(cx1, cy1, cx2, cy2):
    return math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


# ---------------------------
# MATCH TRACKED CAR
# ---------------------------
def match_tracked_car(cx, cy, tracked_cars, threshold=DISTANCE_THRESHOLD):
    best_id = None
    best_dist = threshold
    for car in tracked_cars:
        dist = euclidean_distance(cx, cy, car['cx'], car['cy'])
        if dist < best_dist:
            best_dist = dist
            best_id = car['id']
    return best_id


# ---------------------------
# PROCESS FRAME
# ---------------------------
def process_frame(frame):
    global tracked_cars, car_id_counter
    detections = detect_cars(frame)

    # keep track of which car IDs were updated in this frame
    updated_ids = []

    for x1, y1, x2, y2, conf in detections:
        car_crop = frame[y1:y2, x1:x2]
        brand_id, brand_prob = classify_car(car_crop, model, eval_transform, device)

        # Find the center of this bounding box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Attempt to match it with existing tracked cars
        match_id = match_tracked_car(cx, cy, tracked_cars, DISTANCE_THRESHOLD)
        if match_id is not None:
            # Update the matched car if the new probability is higher
            for car in tracked_cars:
                if car['id'] == match_id:
                    # Reset the "missed frames" counter because we found it again
                    car['frames_since_update'] = 0

                    # Only update brand if brand_prob is higher than the best
                    if brand_prob > car['best_prob']:
                        car['best_prob'] = brand_prob
                        car['best_brand_id'] = brand_id

                    # Update location
                    car['cx'] = cx
                    car['cy'] = cy
                    car['bbox'] = (x1, y1, x2, y2)
                    updated_ids.append(match_id)
                    break
        else:
            # No match found -> add as new car
            new_car = {
                'id': car_id_counter,
                'cx': cx,
                'cy': cy,
                'bbox': (x1, y1, x2, y2),
                'best_brand_id': brand_id,
                'best_prob': brand_prob,
                'frames_since_update': 0
            }
            tracked_cars.append(new_car)
            updated_ids.append(car_id_counter)
            car_id_counter += 1

    # For every car not updated this frame, increment frames_since_update
    for car in tracked_cars:
        if car['id'] not in updated_ids:
            car['frames_since_update'] += 1

    # Remove cars that have not been updated for too many frames
    tracked_cars = [c for c in tracked_cars if c['frames_since_update'] <= MAX_MISSED_FRAMES]

    # Draw bounding boxes for all tracked cars
    for car in tracked_cars:
        x1, y1, x2, y2 = car['bbox']
        brand_name = brand_names.get(car['best_brand_id'], 'Unknown')
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
