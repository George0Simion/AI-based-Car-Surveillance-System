import os
import numpy as np
import cv2
import pandas as pd
import torch
from torchvision.models import resnet101, ResNet101_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

# Objective -> Car Recognition


# -----------------------------
# PATHS & DIRECTORY HANDLING
# -----------------------------
path = "/home/simion/Desktop/AI/AI-based-Car-Surveillance-System/data"

directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
data = {directory: i for i, directory in enumerate(directories)}
brand_names = {v: k for k, v in data.items()} # Inversam dictionarul pt a obtine numele brandurilor dupa label


# -----------------------------
# CONFIG AND HYPERPARAMETERS
# -----------------------------
TARGET_SIZE = (224, 224)   # Dimensiunea imaginilor
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
STEP_SIZE = 7
GAMMA = 0.1


# -----------------------------
# DATAFRAME CREATION
# -----------------------------
df = pd.DataFrame(columns=['image_path', 'label'])

for brand in directories:
    brand_path = Path(path) / brand
    # Recursively gather all files
    for file_path in brand_path.rglob('*.*'):
        if file_path.is_file():
            df.loc[len(df)] = {
                'image_path': str(file_path),
                'label': data[brand]
            }


# -----------------------------
# TRAIN / VALIDATION / TEST SPLIT
# -----------------------------
# 70% train, 20% val, 10% test
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.3333, random_state=42)


# -----------------------------
# TRANSFORM PIPELINE
# -----------------------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # random crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# for validation and test set no data augmentation
eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# DATASET CLASS
# -----------------------------
class CarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_paths = df['image_path'].values  # Image path as tensors
        self.labels = df['label'].values  # Corresponding labels
        self.transform = transform  # Image transformations

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert label to tensor

        # Load and trandform image
        image = cv2.imread(image)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8) # Create a black image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        return image, label
    

# -----------------------------
# DATALOADERS
# -----------------------------
train_dataset = CarDataset(train_df, transform=train_transform)
val_dataset = CarDataset(val_df, transform=eval_transform)
test_dataset = CarDataset(test_df, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


# -----------------------------
# MODEL SETUP
# -----------------------------
weights = ResNet101_Weights.IMAGENET1K_V2 # Load weights from ImageNet
model = resnet101(weights=weights) # Load ResNet-101 model

# Modify the final layer to match the number of car brands
num_classes = len(directories)
model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)

crit = torch.nn.CrossEntropyLoss() # loss function -> CrossEntropyLoss
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) # optimizer -> AdamW
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA) # learning rate scheduler

# Hyperparameters
model.hyperparameters = {
    'epochs': NUM_EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'step_size': STEP_SIZE,
    'gamma': GAMMA
}

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# -----------------------------
# VALIDATION FUNCTION
# -----------------------------
def validate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient computation during validation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability

            total += labels.size(0) # Total number of labels
            correct += (predicted == labels).sum().item() # Number of correct predictions

    # acuratetea modelului
    accuracy = 100 * correct / total
    return accuracy


# -----------------------------
# TRAINING LOOP
# -----------------------------
print("\nStarting Training...\n")

for epoch in range(NUM_EPOCHS):
    model.train()  # training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()         # clear gradients
        outputs = model(images)       # forward pass
        loss = crit(outputs, labels)  # compute loss
        loss.backward()               # backprop
        optimizer.step()              # update model params

        running_loss += loss.item()

    # step the scheduler
    scheduler.step()

    # average training loss for this epoch
    epoch_loss = running_loss / len(train_loader)

    # validate on the training set
    train_accuracy = validate_model(model, train_loader)

    # validate on the validation set
    val_accuracy = validate_model(model, val_loader)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Loss: {epoch_loss:.4f} | "
          f"Train Acc: {train_accuracy:.2f}% | "
          f"Val Acc: {val_accuracy:.2f}%")
    

print("\nTraining completed.\n")


# -----------------------------
# Evaluation
# -----------------------------
test_accuracy = validate_model(model, test_loader)
print(f'Test accuracy: {test_accuracy:.2f}%')


# -----------------------------
# Save model
# -----------------------------
torch.save(model.state_dict(), 'model2.pth')
print("Model saved successfully.")