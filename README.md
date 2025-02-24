# Car Brand Recognition System

## Overview
This project implements a deep learning-based system for car brand recognition using video footage. The pipeline consists of two primary components:

1. **Car Detection** - YOLOv5 is used to detect cars in video frames.
2. **Brand Classification** - A ResNet-based learning model classifies detected cars into their respective brands.

---

## How the System Works
1. **Dataset Preparation**
   - The dataset consists of car images labeled with their respective brands.
   - The dataset is structured into directories where each folder represents a different car brand.
   - Two datasets were used:
     - [Deep Visual Marketing Dataset](https://deepvisualmarketing.github.io/)
     - [Kaggle - 20+ Car Brands Dataset](https://www.kaggle.com/datasets/alirezaatashnejad/over-20-car-brands-dataset)

2. **Model Training**
   - Two models were trained using the datasets:
     1. **ResNet18** trained for **6 epochs** on an **800MB dataset**, saved as `model1.pth`
     2. **ResNet50** trained for **6 epochs** on an **800MB dataset**, saved as `model2.pth`
   - Training was performed using **AdamW optimizer** with a **learning rate of 0.001**.
   - Data augmentation techniques such as **random cropping, horizontal flipping, rotation, and color jittering** were used to improve generalization.
   - The dataset was split into **70% training, 20% validation, and 10% test sets**.

3. **Model Differences and Performance**
   - **ResNet18**:
     - **Accuracy:** 79%
     - **Precision:** 76%
     - **Training Time:** ~4 hours
   - **ResNet50**:
     - **Accuracy:** 87%
     - **Precision:** 85%
     - **Training Time:** ~8 hours
   - ResNet50 provides better accuracy and precision but takes longer to train due to its deeper architecture.
   - Both were trained on CPU.

4. **Video Processing (Real-time Inference)**
   - A **YOLOv5 model** detects cars in each frame.
   - The detected car **bounding boxes** are cropped and passed to the ResNet model.
   - The model **classifies the car brand** based on the highest probability.

---

## Code Explanation

### 1. Training the Classification Model
- The script reads command-line arguments to determine the model architecture (`resnet18`, `resnet50`, or `resnet101`).
- The dataset is processed, and images are loaded using a PyTorch `Dataset` class.
- The model is initialized with pre-trained weights and fine-tuned on the car brand dataset.
- The model is trained using **cross-entropy loss** and optimized using **AdamW**.
- The learning rate scheduler adjusts the learning rate every 7 epochs.
- After training, the model is evaluated on the test set and saved for inference.

### 2. Video Processing Pipeline
- The script loads a **pre-trained YOLOv5 model** to detect cars in video frames.
- A **pre-trained ResNet50 model** (fine-tuned on car brands) classifies detected cars.
- There are 2 video processing ways: a **frame by frame** one (video.py), and a more complex one (advancedVideo.py), which tracks detected cars across **multiple frames** and for each car displays the brand with the highest probability
- Each frame is processed in real-time to overlay **bounding boxes and brand names** on detected cars.

   **Usage:**
   ```sh
   python3 video.py <model_checkpoint> <video_file> <data_folder>
   ```
   - `model_checkpoint`: Path to the trained model (`model1.pth` or `model2.pth`).
   - `video_file`: Path to the video file.
   - `data_folder`: Path to the dataset directory.

3. **Training a New Model (`main.py`)**
   - Trains a ResNet model on the dataset.
   - Saves the trained model.
   
   **Usage:**
   ```sh
   python3 main.py <MODEL_NAME> <NUM_EPOCHS> <DATASET_PATH> <OUTPUT_MODEL_NAME>
   ```
   - `MODEL_NAME`: `resnet18`, `resnet50`, or `resnet101`.
   - `NUM_EPOCHS`: Number of epochs.
   - `DATASET_PATH`: Path to dataset directory.
   - `OUTPUT_MODEL_NAME`: Name of the output model file.

---

## Results and Performance
- **ResNet50** outperforms **ResNet18** in accuracy and precision but requires more training time.
- **YOLOv5** successfully detects cars in real-time, with a confidence threshold of 50%.
- The system is able to **track** and **update** detected cars over multiple frames.
- The brand classification model can handle **over 20 car brands** with reasonable accuracy.

