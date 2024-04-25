import pandas as pd
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, df, image_folder_left, image_folder_right, transform=None):
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
        
        left_points = row['Left_2D']
        right_points = row['Right_2D']
        points_3d = row['3D_Point']

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        
        left_points = left_points.strip('()')
        values = left_points.split(',')
        # Convert string values to integers
        left_points = [float(value) for value in values]
        
        right_points = right_points.strip('()')
        values = right_points.split(',')
        # Convert string values to integers
        right_points = [float(value) for value in values]

        points_3d = points_3d.strip('()')
        values = points_3d.split(',')
        # Convert string values to integers
        points_3d = [float(value) for value in values]

        left_image = torch.tensor(left_image)
        right_image = torch.tensor(right_image)
        left_points = torch.tensor(left_points)
        right_points = torch.tensor(right_points)
        points_3d = torch.tensor(points_3d)
        return left_image, right_image, left_points, right_points, points_3d

# Example usage
csv_file = 'data/test_file.csv'
image_folder_left = 'src/data/Training_Data_Left_5per_gelatin/'
image_folder_right = 'src/data/Training_Data_Right_5per_gelatin/'

# Read CSV file with UTF-8 encoding
train_df, eval_df = train_test_split(pd.read_csv(csv_file, encoding='utf-8'), test_size=0.2)

# Define transformations for images (if needed)
transform = transforms.Compose([
    # Add any necessary transformations here
    transforms.ToTensor(),
])

# Create train and eval datasets
train_dataset = CustomDataset(train_df, image_folder_left, image_folder_right, transform=transform)
eval_dataset = CustomDataset(eval_df, image_folder_left, image_folder_right, transform=transform)

# Create train and eval data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False)

# Print information about the DataLoader objects
print("DataLoader objects created successfully.")
print(f"Size of training dataset: {len(train_loader.dataset)}")
print(f"Size of validation dataset: {len(eval_loader.dataset)}")
