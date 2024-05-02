import pandas as pd
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
        
        left_points = row['Left_2D']
        right_points = row['Right_2D']
        points_3d = row['3D_Point']

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        
        left_points = left_points.strip('()').split(',')
        # Convert string values to integers
        left_points = [float(value) for value in left_points]
        
        right_points = right_points.strip('()').split(',')
        # Convert string values to integers
        right_points = [float(value) for value in right_points]

        points_3d = points_3d.strip('()').split(',')
        # Convert string values to integers
        points_3d = [float(value) for value in points_3d]

        left_image = torch.tensor(left_image)
        right_image = torch.tensor(right_image)
        left_points = torch.tensor(left_points)
        right_points = torch.tensor(right_points)
        points_3d = torch.tensor(points_3d)
        return left_image, right_image, left_points, right_points, points_3d

# Train loader
train_csv_files = ['data/CSVs/Training_5per.csv', 'data/CSVs/Training_10per.csv', 'data/CSVs/Training_15per.csv']
image_folders_left = ['src/data/Needle_Images_New/Training_Data_Left_5per_gelatin/', 
                     'src/data/Needle_Images_New/Training_Data_Left_10per_gelatin/',
                     'src/data/Needle_Images_New/Training_Data_Left_15per_gelatin/']
image_folders_right = ['src/data/Needle_Images_New/Training_Data_Right_5per_gelatin/',
                      'src/data/Needle_Images_New/Training_Data_Right_10per_gelatin/',
                      'src/data/Needle_Images_New/Training_Data_Right_15per_gelatin/']

# Read CSV file with UTF-8 encoding
train_dfs = [pd.read_csv(file, encoding='utf-8') for file in csv_files]

# Define transformations for images (if needed)
transform = transforms.Compose([
    # Add any necessary transformations here
    transforms.ToTensor(),
])

train_datasets = []
for train_dfs, folder_left, folder_right in zip(dfs, image_folders_left, image_folders_right):
    train_datasets.append(CustomDataset(df, folder_left, folder_right, transform=transform))

# Combine datasets into a single dataset
combined_train_dataset = ConcatDataset(datasets)


eval_csv_files = ['data/CSVs/Validation_5per.csv', 'data/CSVs/Validation_10per.csv', 'data/CSVs/Validation_15per.csv']

# Read CSV file with UTF-8 encoding
eval_dfs = [pd.read_csv(file, encoding='utf-8') for file in csv_files]

eval_datasets = []
for eval_dfs, folder_left, folder_right in zip(dfs, image_folders_left, image_folders_right):
    eval_datasets.append(CustomDataset(df, folder_left, folder_right, transform=transform))

# Combine datasets into a single dataset
combined_eval_dataset = ConcatDataset(datasets)

# Create train and eval data loaders
train_loader = DataLoader(combined_train_dataset, batch_size=4, shuffle=True)
eval_loader = DataLoader(combined_eval_dataset, batch_size=4, shuffle=False)

print("DataLoader objects created successfully.")
print(f"Size of training dataset: {len(train_loader.dataset)}")
#print(f"Size of validation dataset: {len(eval_loader.dataset)}")
