import cv2
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, hdf5_file, group_names, transform=None):
        self.hdf5_file = hdf5_file
        self.group_names = group_names
        self.transform = transform

        # Initialize lists to hold datasets from all groups
        self.left_images = []
        self.right_images = []
        self.left_points = []
        self.right_points = []
        self.points_3d = []

        # Iterate over each HDF5 file and corresponding group
        for group_name in group_names:
            with h5py.File(hdf5_file, 'r') as h5f:
                dataset_group = h5f[group_name]
                # Append datasets from current group to the lists
                self.left_images.extend(dataset_group['imageLeft'][:])
                self.right_images.extend(dataset_group['imageRight'][:])
                self.left_points.extend(dataset_group['Left_2D'][:])
                self.right_points.extend(dataset_group['Right_2D'][:])
                self.points_3d.extend(dataset_group['3D_Point'][:])

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        
        left_image = self.left_images[idx]
        right_image = self.right_images[idx]

        height, width, channels = 1024, 1280, 3
        left_image = np.reshape(left_image, (height, width, channels))
        right_image = np.reshape(right_image, (height, width, channels))
        
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        left_points = self.left_points[idx]
        right_points = self.right_points[idx]
        points_3d = self.points_3d[idx]

        left_points = left_points.decode('utf-8').strip('()').split(',')
        left_points = [float(value) for value in left_points]
        
        right_points = right_points.decode('utf-8').strip('()').split(',')
        right_points = [float(value) for value in right_points]

        points_3d = points_3d.decode('utf-8').strip('()').split(',')
        points_3d = [float(value) for value in points_3d]

        left_image = torch.tensor(left_image)
        right_image = torch.tensor(right_image)
        left_points = torch.tensor(left_points)
        right_points = torch.tensor(right_points)
        points_3d = torch.tensor(points_3d)

        return left_image, right_image, left_points, right_points, points_3d

# Load files and define group names
train_hdf5_file = 'data/Training.h5' 
#eval_hdf5_file = 'data/Validation.h5'
group_names = ['5per_dataset', '10per_dataset', '15per_dataset']

# Define transformations for images (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
])

print(f'Read files..........')

# Create datasets and data loaders
train_dataset = CustomDataset(train_hdf5_file, group_names, transform=transform)
#eval_dataset = CustomDataset(eval_hdf5_file, group_names, transform=transform)

print(f'Datasets created. Moving to dataloaders.................')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
#eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

print("DataLoader objects created successfully.")

print(f"Size of training dataset: {len(train_loader.dataset)}")
#print(f"Size of validation dataset: {len(eval_loader.dataset)}")