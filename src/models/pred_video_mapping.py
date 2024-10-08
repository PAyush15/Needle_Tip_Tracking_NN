from __future__ import print_function
import argparse
import numpy as np
import cv2
import torchvision.models as models

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from prep_dataset_csv import train_loader, eval_loader, test_loader
import os


class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=False):
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


class NeedleLocalizationNetwork(nn.Module):
    def __init__(self, num_coordinates=2):
        super(NeedleLocalizationNetwork, self).__init__()
        
        # Load pre-trained ResNet-18 model
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the fully connected layer with a new one for regression
        num_ftrs = self.resnet.fc.in_features

        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-2]))

        # Add AdaptiveAvgPooling2d layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(num_ftrs, num_coordinates)

        self.dropout = nn.Dropout(0.3)  # Add dropout layer   
        

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        # Pass input through ResNet-18
        features = self.resnet(x)

        # Apply AdaptiveAvgPooling2d
        features = self.avg_pool(features)
        features = features.view(features.size()[0], -1)

        features = self.fc(features)

        return features


def train(args, model, device, train_loader, optimizer, scheduler, epoch):
    model.train()

    criterion = nn.MSELoss()

    for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(train_loader):
        left_image, targets = left_image.to(device), left_points.to(device)
        optimizer.zero_grad()
        outputs = model(left_image).squeeze()
        
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


def test(model, device, eval_loader):
    model.eval()
    test_loss = 0
    total_rmse = 0
    num_samples = 0
    total_distance = 0.0

    criterion = nn.MSELoss()

    # Setup video writer
    output_path = 'predictions_video_3.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30
    height, width = None, None
    video_writer = None

    with torch.no_grad():
        for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(eval_loader):
            left_image, targets = left_image.to(device), left_points.to(device)
            left_points = [[f"{value:.1f}" for value in point] for point in left_points.tolist()]
            outputs = model(left_image).squeeze()

            outputs = outputs.view_as(targets)

            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss

            outputs_denorm = denormalize_2d_points(outputs.to('cpu'))
            targets_denorm = denormalize_2d_points(targets.to('cpu'))

            outputs_denorm = torch.tensor(outputs_denorm)
            targets_denorm = torch.tensor(targets_denorm)

            
            batch_rmse = torch.sqrt(torch.mean((outputs_denorm - targets_denorm)**2, dim=1))
            total_rmse += torch.sum(batch_rmse).item()
            num_samples += len(batch_rmse)
            
            batch_distance = torch.sqrt(torch.sum((outputs_denorm - targets_denorm)**2, dim=1))
            total_distance += torch.sum(batch_distance).item()

            for i in range(left_image.size(0)):  # Loop through each image in the batch
                frame = visualize(left_image[i].cpu().numpy().transpose((1, 2, 0)), targets_denorm[i].cpu().tolist(), outputs_denorm[i].cpu().tolist())

                if video_writer is None:
                    height, width, _ = frame.shape
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # Convert the frame to 8-bit
                frame = cv2.convertScaleAbs(frame)
                video_writer.write(frame)

    test_loss /= len(eval_loader.dataset)

    print('\nTest set: Average loss: {:.4f}'.format(
        test_loss, len(eval_loader.dataset)))
    average_distance = total_distance / num_samples
    print(f'Average distance: {average_distance:.4f}')

    if video_writer is not None:
        video_writer.release()
        print(f'Video saved as {output_path}')

    return test_loss, average_distance


def denormalize_2d_points(points):
    """
    Denormalize multiple points from the 0 to 1 range back to the original range.
    """
    min_values = torch.tensor([77, 217])
    max_values = torch.tensor([984, 610])
    
    denormalized_points = []
    for point in points:
        point_tensor = torch.tensor(point)
        denorm_point = point_tensor * (max_values - min_values) + min_values
        denormalized_points.append(denorm_point.tolist())
    
    return denormalized_points


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray[:, :, None]
    return gray.astype(np.uint8)


def project_points(image, point):
    projected_image = image.copy()
    x, y = point
    x, y = int(x), int(y)
    cv2.circle(projected_image, (x, y), 10, (0, 255, 0), 3)
    return projected_image


def visualize(image, original_points, predicted_points):
    image = (image * 255).astype(np.uint8)  # Scale image to 0-255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    projected_image = project_points(image, predicted_points)
    return projected_image


def main():
    parser = argparse.ArgumentParser(description='PyTorch Needle Localization Network')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--lr', type=float, default=0.008, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--patience', type=int, default=3, metavar='N',
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
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
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

    model = NeedleLocalizationNetwork().to(device)
    if use_cuda:
        model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)
    scheduler = StepLR(optimizer, step_size=2, gamma=args.gamma)

    model.load_state_dict(torch.load("Saved_Models/cross_valid_model_2D.pt"))
    model.eval()

    for epoch in range(1, args.epochs + 1):
        #train_loss = train(args, model, device, train_loader, optimizer, scheduler, epoch)
        test_loss = test(model, device, test_loader)

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if args.save_model:
        torch.save(model.state_dict(), "Saved_Models/cross_valid_model_2D.pt")


if __name__ == '__main__':
    main()
