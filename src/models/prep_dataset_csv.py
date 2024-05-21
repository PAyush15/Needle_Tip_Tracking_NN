import pandas as pd
import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

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
        # Rotate the image by 180 degrees
        #right_image = cv2.rotate(right_image_norm, cv2.ROTATE_180)

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

        # min_vals = (542.104736, -559.603027, 273.826233)
        # max_vals = (584.9863, -335.045776, 312.355347)

        # normalized_3d_point = normalize_3d_point(min_vals, max_vals, points_3d)

        left_image = torch.tensor(left_image, dtype=torch.float32)
        right_image = torch.tensor(right_image, dtype=torch.float32)
        left_points = torch.tensor(left_points, dtype=torch.float32)
        right_points = torch.tensor(right_points, dtype=torch.float32)
        points_3d = torch.tensor(points_3d, dtype=torch.float32)

        return left_image, right_image, left_points, right_points, points_3d
    

def normalize_3d_point(min_vals, max_vals, point):
    """
    Normalize a 3D point to a specified range.
    
    Args:
        point (tuple or list): 3D point (x, y, z)
        min_vals (tuple or list): Minimum values for each coordinate (x_min, y_min, z_min)
        max_vals (tuple or list): Maximum values for each coordinate (x_max, y_max, z_max)
        
    Returns:
        tuple: Normalized 3D point (x_normalized, y_normalized, z_normalized)
    """
    x_min, y_min, z_min = min_vals
    x_max, y_max, z_max = max_vals
    
    x_normalized = (2 * (point[0] - x_min) / (x_max - x_min)) - 1
    y_normalized = (2 * (point[1] - y_min) / (y_max - y_min)) - 1
    z_normalized = (2 * (point[2] - z_min) / (z_max - z_min)) - 1
    
    return [x_normalized, y_normalized, z_normalized]



# Example usage

"""

train_csv_files = ['data/CSVs/Training_5per.csv', 'data/CSVs/Training_10per.csv', 'data/CSVs/Training_15per.csv']
eval_csv_files = ['data/CSVs/Validation_5per.csv', 'data/CSVs/Validation_10per.csv', 'data/CSVs/Validation_15per.csv']

image_folders_left = ['src/data/Needle_Images_New/Training_Data_Left_5per_gelatin/', 
                     'src/data/Needle_Images_New/Training_Data_Left_10per_gelatin/',
                     'src/data/Needle_Images_New/Training_Data_Left_15per_gelatin/']
image_folders_right = ['src/data/Needle_Images_New/Training_Data_Right_5per_gelatin/',
                      'src/data/Needle_Images_New/Training_Data_Right_10per_gelatin/',
                      'src/data/Needle_Images_New/Training_Data_Right_15per_gelatin/']
"""

train_csv_files = ['data/CSVs/Training_Sample.csv']
eval_csv_files = ['data/CSVs/Validation_Sample.csv']

image_folders_left = ['src/data/Needle_Images_New/Training_Data_Left_5per_gelatin/']
image_folders_right = ['src/data/Needle_Images_New/Training_Data_Right_5per_gelatin/']


# Read CSV file with UTF-8 encoding
train_dfs = [pd.read_csv(file, encoding='utf-8') for file in train_csv_files]
eval_dfs = [pd.read_csv(file, encoding='utf-8') for file in eval_csv_files]


# Define transformations for images (if needed)
transform = transforms.Compose([
    # Add any necessary transformations here
    transforms.ToTensor(),
])

train_dataset_list = []
eval_dataset_list = []

for df, folder_left, folder_right in zip(train_dfs, image_folders_left, image_folders_right):
    train_dataset_list.append(CustomDataset(df, folder_left, folder_right, transform=transform))

for df, folder_left, folder_right in zip(eval_dfs, image_folders_left, image_folders_right):
    eval_dataset_list.append(CustomDataset(df, folder_left, folder_right, transform=transform))


# Create train and eval datasets
train_dataset = ConcatDataset(train_dataset_list)
eval_dataset = ConcatDataset(eval_dataset_list)

print(f'Datasets created. Moving to dataloaders.................')

# Create train and eval data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

print("DataLoader objects created successfully.")
print(f"Size of training dataset: {len(train_loader.dataset)}")
print(f"Size of validation dataset: {len(eval_loader.dataset)}")
