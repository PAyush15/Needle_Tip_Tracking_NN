import pandas as pd
import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import KFold

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

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
        left_points = torch.tensor(normalized_2d_points, dtype=torch.float32)
        right_points = torch.tensor(right_points, dtype=torch.float32)
        points_3d = torch.tensor(normalized_3d_points, dtype=torch.float32)
        
        return left_image, right_image, left_points, right_points, points_3d
    
class EarlyStopping:
    def __init__(self, patience, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'Saved_Models/checkpoint.pt')
        self.val_loss_min = val_loss


class SiameseNetwork(nn.Module):
    """
        Siamese network for predicting the 3D points using 2 input images.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-2]))

        # Init average pool layer that needs to be manually added to the EfficientNet feature extractor
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

        self.dropout = nn.Dropout(0.5)  # Add dropout layer
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = self.avg_pooling(output)  # Shape: [batch, # feature maps, 1, 1]
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        #output1 = self.dropout(output1)  # Apply dropout
        #output2 = self.dropout(output2)  # Apply dropout

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        return output


def train(args, model, device, train_loader, optimizer, scheduler, epoch):
    model.train()

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.MSELoss()

    for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(train_loader):

        #if batch_idx % 2 != 0:
        #    continue
        
        left_image, right_image, targets = left_image.to(device), right_image.to(device), points_3d.to(device)
        optimizer.zero_grad()
        outputs = model(left_image, right_image).squeeze()

        # Ensure outputs and targets have the same shape
        outputs = outputs.view_as(targets)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(left_image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    scheduler.step()
    return loss


def test(model, device, eval_loader):
    model.eval()
    test_loss = 0
    total_mae = 0.0
    num_samples = 0
    total_distance = 0.0
    mae_list = []

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(eval_loader):

            #if batch_idx % 2 != 0:
            #    continue

            left_image, right_image, targets = left_image.to(device), right_image.to(device), points_3d.to(device)
            outputs = model(left_image, right_image).squeeze()

            # Ensure outputs and targets have the same shape
            outputs = outputs.view_as(targets)

            loss = criterion(outputs, targets)
            test_loss += loss.sum().item()  # sum up batch loss

            outputs_denorm = denormalize_3d_points(outputs.to('cpu'))
            targets_denorm = denormalize_3d_points(targets.to('cpu'))

            outputs_denorm = torch.tensor(outputs_denorm)
            targets_denorm = torch.tensor(targets_denorm)

            # Calculate MAE for each point
            batch_mae = torch.mean(torch.abs(outputs_denorm - targets_denorm), dim=1)
            total_mae += torch.sum(batch_mae).item()
            num_samples += len(batch_mae)
            mae_list.extend(batch_mae.tolist())

            # Calculate distance for each point
            batch_distance = torch.sqrt(torch.sum((outputs_denorm- targets_denorm)**2, dim=1))
            total_distance += torch.sum(batch_distance).item()

    test_loss /= len(eval_loader.dataset)

    print(f'Original points: {targets_denorm}')
    print(f'Predicted points: {outputs_denorm}')

    print('\nTest set: Average loss: {:.4f}'.format(
        test_loss, len(eval_loader.dataset)))
    
    # Calculate average MAE
    average_mae = total_mae / num_samples
    mae_std = torch.std(torch.tensor(mae_list)).item()
    print(f'Average MAE: {average_mae:.4f}')
    print(f'MAE Standard Deviation: {mae_std:.4f}')

    # Calculate average distance
    average_distance = total_distance / num_samples
    print(f'Average distance: {average_distance:.4f}')
    print()

    return test_loss, average_distance, average_mae, mae_std


def denormalize_3d_points(points):
    """
    Denormalize multiple points from the 0 to 1 range back to the original range.
    """
    min_values = torch.tensor([530.657593, -559.603027, 275.303467])
    max_values = torch.tensor([594.552063, -335.045776, 321.951843])
    
    denormalized_points = []
    for point in points:
        point_tensor = torch.tensor(point)
        denorm_point = point_tensor * (max_values - min_values) + min_values
        denormalized_points.append(denorm_point.tolist())
    
    return denormalized_points

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


# Read CSV files
train_csv_files = ['data/CSVs/Training_5per.csv', 'data/CSVs/Training_10per.csv', 'data/CSVs/Training_15per.csv']
eval_csv_files = ['data/CSVs/Validation_5per.csv', 'data/CSVs/Validation_10per.csv', 'data/CSVs/Validation_15per.csv']

image_folders_left = ['src/data/Needle_Images_Left']
image_folders_right = ['src/data/Needle_Images_Right']

# Read CSV file with UTF-8 encoding
train_dfs = [pd.read_csv(file, encoding='utf-8').sample(frac=0.7, random_state=42) for file in train_csv_files]
eval_dfs = [pd.read_csv(file, encoding='utf-8').sample(frac=0.7, random_state=42) for file in eval_csv_files]

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
eval_params_list = []

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
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--lr', type=float, default=0.008, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.2, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--patience', type=int, default=5, metavar='N',
                        help='patience for early stopping (default: 5)')
    parser.add_argument('--delta', type=float, default=0, metavar='M',
                        help='delta for early stopping (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
        device_ids = [0, 1, 2, 3]
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Initialize the model
    model = SiameseNetwork().to(device)
    if device_ids is not None:
        model = nn.DataParallel(model, device_ids=device_ids)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)

    scheduler = StepLR(optimizer, step_size=3, gamma=args.gamma)

    # Load the previously trained model
    #model.load_state_dict(torch.load("siamese_network_resnet_v02.pt"))
    #model.eval()

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, scheduler, epoch)
        test_loss, avg_dist, mae, mae_std = test(model, device, eval_loader)
        eval_params_list.append([epoch, avg_dist, mae, mae_std])
        print("Evaluation Parameters (epoch, avg_dist, mae, std): ", eval_params_list) 

        with open("Prediction_Results/CV_3D.txt", 'w') as file:
            for item in eval_params_list:
                file.write("%s\n" % item)
        file.close()

        if fold_var == 0:
            best_dist = avg_dist
            torch.save(model.state_dict(), "Saved_Models/CrossValidationResults.pt")
        
        if avg_dist < best_dist:
            torch.save(model.state_dict(), "Saved_Models/CrossValidationResults.pt")
            best_dist = avg_dist

        # Check for early stopping
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    fold_var += 1