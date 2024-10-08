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

        normalized_3d_points = normalize_3d_points(points_3d)
        normalized_2d_points = normalize_2d_points(left_points)

        left_image = torch.tensor(left_image, dtype=torch.float32)
        right_image = torch.tensor(right_image, dtype=torch.float32)
        left_points = torch.tensor(normalized_2d_points, dtype=torch.float32)
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


def normalize_2d_points(points):
    """
    Normalize multiple points to the 0 to 1 range.
    """
    min_values = (77, 217)
    max_values = (984, 610)
    
    x_norm = (points[0] - min_values[0]) / (max_values[0] - min_values[0])
    y_norm = (points[1] - min_values[1]) / (max_values[1] - min_values[1])

    return [x_norm, y_norm]


train_csv_files = ['data/CSVs/Training_5per.csv', 'data/CSVs/Training_10per.csv', 'data/CSVs/Training_15per.csv']
eval_csv_files = ['data/CSVs/Validation_5per.csv', 'data/CSVs/Validation_10per.csv', 'data/CSVs/Validation_15per.csv']
test_csv_files = ['data/CSVs/Testing_5per.csv', 'data/CSVs/Testing_10per.csv', 'data/CSVs/Testing_15per.csv']

image_folders_left = ['src/data/Needle_Images_New/Training_Data_Left_5per_gelatin/', 
                     'src/data/Needle_Images_New/Training_Data_Left_10per_gelatin/', 'src/data/Needle_Images_New/Training_Data_Left_15per_gelatin/']
image_folders_right = ['src/data/Needle_Images_New/Training_Data_Right_5per_gelatin/',
                      'src/data/Needle_Images_New/Training_Data_Right_10per_gelatin/', 'src/data/Needle_Images_New/Training_Data_Right_15per_gelatin/']

"""
train_csv_files = ['data/CSVs/Training_Sample.csv']
eval_csv_files = ['data/CSVs/Validation_Sample.csv']
test_csv_files = ['data/CSVs/Validation_Sample.csv']

image_folders_left = ['src/data/Needle_Images_New/Training_Data_Left_5per_gelatin/']
image_folders_right = ['src/data/Needle_Images_New/Training_Data_Right_5per_gelatin/']
"""

# Read CSV file with UTF-8 encoding
train_dfs = [pd.read_csv(file, encoding='utf-8') for file in train_csv_files]
eval_dfs = [pd.read_csv(file, encoding='utf-8') for file in eval_csv_files]
test_dfs = [pd.read_csv(file, encoding='utf8') for file in test_csv_files]


# Define transformations for images (if needed)
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array or Tensor to PIL Image
    transforms.RandomVerticalFlip(p=0.5),  # Apply random vertical flip with probability 0.5
    transforms.RandomRotation(degrees=20),  # Apply random rotation within the range of -20 to 20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Apply random changes in brightness, contrast, saturation, and hue
    transforms.ToTensor(),  # Convert PIL Image back to Tensor
])

basic_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

train_dataset_list = []
eval_dataset_list = []
test_dataset_list = []

for df, folder_left, folder_right in zip(train_dfs, image_folders_left, image_folders_right):
    train_dataset_list.append(CustomDataset(df, folder_left, folder_right, transform=transform))

for df, folder_left, folder_right in zip(eval_dfs, image_folders_left, image_folders_right):
    eval_dataset_list.append(CustomDataset(df, folder_left, folder_right, transform=basic_transform))

for df, folder_left, folder_right in zip(test_dfs, image_folders_left, image_folders_right):
    test_dataset_list.append(CustomDataset(df, folder_left, folder_right, transform=basic_transform))

# Create train and eval datasets
train_dataset = ConcatDataset(train_dataset_list)
eval_dataset = ConcatDataset(eval_dataset_list)
test_dataset = ConcatDataset(test_dataset_list)

print(f'Datasets created. Creating dataloaders.................')

# Create train and eval data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("DataLoader objects created successfully.")
print(f"Size of training dataset: {len(train_loader.dataset)}")
print(f"Size of validation dataset: {len(eval_loader.dataset)}")
print(f"Size of test dataset: {len(test_loader.dataset)}")
