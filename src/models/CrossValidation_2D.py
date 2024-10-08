import pandas as pd
import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from ultralytics import YOLO

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, df, image_folder_left, image_folder_right, transform):
        self.df = df
        self.image_folder_left = image_folder_left
        self.image_folder_right = image_folder_right
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        left_image_path = os.path.join(self.image_folder_left, f"{row['imageLeft']}.png")
        right_image_path = os.path.join(self.image_folder_right, f"{row['imageRight']}.png")
        
        left_image = cv2.imread(left_image_path)
        right_image = cv2.imread(right_image_path)
        
        # Convert images to RGB (if they are in BGR format)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        left_points = row['Left_2D']
        right_points = row['Right_2D']
        points_3d = row['3D_Point']

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        
        left_points = left_points.strip('()').split(',')
        left_points = [float(value) for value in left_points]
        
        right_points = right_points.strip('()').split(',')
        right_points = [float(value) for value in right_points]

        points_3d = points_3d.strip('()').split(',')
        points_3d = [float(value) for value in points_3d]

        normalized_3d_points = normalize_3d_points(points_3d)
        #normalized_2d_points = normalize_2d_points(left_points)

        left_image = torch.tensor(left_image, dtype=torch.float32)
        right_image = torch.tensor(right_image, dtype=torch.float32)
        left_points = torch.tensor(left_points, dtype=torch.float32)
        right_points = torch.tensor(right_points, dtype=torch.float32)
        points_3d = torch.tensor(normalized_3d_points, dtype=torch.float32)
        
        return left_image, right_image, left_points, right_points, points_3d


def normalize_3d_points(points):
    """
    Normalize multiple points to the 0 to 1 range.
    """
    min_values = (530.657593, -559.603027, 275.303467)
    max_values = (594.552063, -335.045776, 321.951843)
    
    x_norm = (points[0] - min_values[0]) / (max_values[0] - min_values[0])
    y_norm = (points[1] - min_values[1]) / (max_values[1] - min_values[1])
    z_norm = (points[2] - min_values[2]) / (max_values[2] - min_values[2])

    return [x_norm, y_norm, z_norm]


# Read CSV files
train_csv_files = ['data/CSVs/Training_5per.csv']
eval_csv_files = ['data/CSVs/Validation_5per.csv']

image_folders_left = ['src/data/Needle_Images_Left']
image_folders_right = ['src/data/Needle_Images_Right']

# Read CSV file with UTF-8 encoding
train_dfs = [pd.read_csv(file, encoding='utf-8').sample(frac=0.75, random_state=42) for file in train_csv_files]
eval_dfs = [pd.read_csv(file, encoding='utf-8').sample(frac=0.75, random_state=42) for file in eval_csv_files]

# Combine all data into one DataFrame
combined_df = pd.concat(train_dfs + eval_dfs).reset_index(drop=True)

# Define transformations for images (if needed)
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array or Tensor to PIL Image
    transforms.RandomHorizontalFlip(p=0.5),  # Apply random horizontal flip with probability 0.5
    transforms.RandomVerticalFlip(p=0.5),  # Apply random vertical flip with probability 0.5
    transforms.RandomRotation(degrees=20),  # Apply random rotation within the range of -20 to 20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Apply random changes in brightness, contrast, saturation, and hue
    transforms.ToTensor(),  # Convert PIL Image back to Tensor
])

basic_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Create the combined dataset
combined_dataset = CustomDataset(combined_df, image_folders_left[0], image_folders_right[0], transform=transform)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_var = 0
best_dist = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(combined_dataset)):
    # Split train_val into train and validation
    train_subset = Subset(combined_dataset, train_idx)
    val_subset = Subset(combined_dataset, val_idx)

    print(f"Fold {fold + 1}")
    print(f"Size of training dataset for fold {fold + 1}: {len(train_subset)}")
    print(f"Size of validation dataset for fold {fold + 1}: {len(val_subset)}")

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(val_subset, batch_size=32, shuffle=True)
    
    config_path = 'src/models/YOLOv8/config.yaml'
    model = YOLO('yolov8n.yaml')

    results = model.train(data=config_path, epochs=10, imgsz=640)      

    fold_var += 1

