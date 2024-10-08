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
from ultralytics import YOLO
from torch.utils.data import Dataset


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
        normalized_2d_points = normalize_2d_points(left_points)

        left_image = torch.tensor(left_image, dtype=torch.float32)
        right_image = torch.tensor(right_image, dtype=torch.float32)
        left_points = torch.tensor(left_points, dtype=torch.float32)
        right_points = torch.tensor(right_points, dtype=torch.float32)
        points_3d = torch.tensor(normalized_3d_points, dtype=torch.float32)
        
        return left_image, right_image, left_points, right_points, points_3d


def test(model, device, test_loader , vid_index):

    total_rmse = 0.0
    num_samples = 0
    total_distance = 0.0

    # Setup video writer
    output_path = f'Prediction_results/cross_validation_mapping_{vid_index}.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30
    height, width = None, None
    video_writer = None

    for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(test_loader):
        
        left_image, targets = left_image.to(device), left_points.to(device)
        results = model(left_image, verbose=False)
        outputs = torch.zeros_like(targets, device=device)
        confidence = [0] * len(results)

        for i, result in enumerate(results):
            if result is not None and results is not None:
                boxes = result.boxes  # Boxes object for bounding box outputs
                outputs[i] = torch.tensor([boxes.cpu().xywh[0,0], boxes.cpu().xywh[0,1]])

                confidence[i] = result.boxes.conf.tolist()[0]
            else:
                confidence[i] = 0
                outputs[i] = torch.tensor([0, 0])

        for i in range(left_image.size(0)):  # Loop through each image in the batch
            frame = visualize(left_image[i].cpu().numpy().transpose((1, 2, 0)), outputs[i].cpu().tolist(), confidence[i])

            if video_writer is None:
                height, width, _ = frame.shape
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Convert the frame to 8-bit
            frame = cv2.convertScaleAbs(frame)
            video_writer.write(frame)


        #result.save(filename="result.jpg")  # save to disk

        # Calculate RMSE for each point
        batch_rmse = torch.sqrt(torch.mean((outputs - targets.to(device))**2, dim=1))
        total_rmse += torch.sum(batch_rmse).item()
        num_samples += len(batch_rmse)

        # Calculate distance for each point
        batch_distance = torch.sqrt(torch.sum((outputs - targets.to(device))**2, dim=1))
        total_distance += torch.sum(batch_distance).item()

            
    average_distance = total_distance / num_samples
    print(f'Average distance: {average_distance:.4f}')

    if video_writer is not None:
        video_writer.release()
        print(f'Video saved as {output_path}')


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


def project_points(image, point, confidence):
    projected_image = image.copy()
    x, y = point
    x, y = int(x), int(y)
    cv2.circle(projected_image, (x, y), 10, (0, 255, 0), 3)
    cv2.putText(projected_image, f"Conf: {confidence:.2f}", (int(x+10), int(y-10) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
    return projected_image


def visualize(image, predicted_points, confidence):
    image = (image * 255).astype(np.uint8)  # Scale image to 0-255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    projected_image = project_points(image, predicted_points, confidence)
    return projected_image


# Read CSV files
train_csv_files = ['data/CSVs/Training_5per.csv', 'data/CSVs/Training_10per.csv', 'data/CSVs/Training_15per.csv']
eval_csv_files = ['data/CSVs/Validation_5per.csv', 'data/CSVs/Validation_10per.csv', 'data/CSVs/Validation_15per.csv']
test_csv_files = ['data/CSVs/Testing_5per.csv', 'data/CSVs/Testing_10per.csv', 'data/CSVs/Testing_15per.csv']

image_folders_left = ['src/data/Needle_Images_New/Training_Data_Left_5per_gelatin/', 
                     'src/data/Needle_Images_New/Training_Data_Left_10per_gelatin/', 'src/data/Needle_Images_New/Training_Data_Left_15per_gelatin/']
image_folders_right = ['src/data/Needle_Images_New/Training_Data_Right_5per_gelatin/',
                      'src/data/Needle_Images_New/Training_Data_Right_10per_gelatin/', 'src/data/Needle_Images_New/Training_Data_Right_15per_gelatin/']

# Read CSV file with UTF-8 encoding
train_dfs = [pd.read_csv(file, encoding='utf-8') for file in train_csv_files]
eval_dfs = [pd.read_csv(file, encoding='utf-8') for file in eval_csv_files]
test_dfs = [pd.read_csv(file, encoding='utf8') for file in test_csv_files]

# Combine all data into one DataFrame
combined_df = pd.concat(train_dfs + eval_dfs + test_dfs).reset_index(drop=True)

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

for fold, (train_val_idx, test_idx) in enumerate(kf.split(combined_dataset)):
    # Split train_val into train and validation
    train_val_subset = Subset(combined_dataset, train_val_idx)
    test_subset = Subset(combined_dataset, test_idx)

    train_val_len = len(train_val_subset)
    val_split = int(train_val_len * 0.2)
    train_split = train_val_len - val_split

    train_subset, val_subset = torch.utils.data.random_split(train_val_subset, [train_split, val_split])

    print(f"Fold {fold + 1}")
    print(f"Size of training dataset for fold {fold + 1}: {len(train_subset)}")
    print(f"Size of validation dataset for fold {fold + 1}: {len(val_subset)}")
    print(f"Size of test dataset for fold {fold + 1}: {len(test_subset)}")

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)
    
    
    model = YOLO("/home/Patel/Dokumente/lightning-hydra-template/Network_Checkpoints/best.pt")  
    device = 'cuda'

    test(model, device, test_loader, fold_var)

    fold_var += 1