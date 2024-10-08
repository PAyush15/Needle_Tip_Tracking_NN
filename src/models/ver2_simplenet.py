import torch
import torch.nn as nn

import argparse
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from prep_dataset_csv import train_loader, eval_loader

torch.cuda.empty_cache()

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


class NeedleTipCNN(nn.Module):
    def __init__(self):
        super(NeedleTipCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 160 * 128, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 160 * 128)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(args, model, device, train_loader, optimizer, scheduler, epoch):
    model.train()

    criterion = nn.MSELoss()

    for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(train_loader):

        if batch_idx % 2 != 0:
            continue
        
        left_image, targets = left_image.to(device), left_points.to(device)

        optimizer.zero_grad()
        outputs = model(left_image).squeeze()
    
        print("Outputs shape: ", outputs.shape)
        print("Targets shape: ",)
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
    total_rmse = 0
    num_samples = 0
    total_distance = 0.0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(eval_loader):

            if batch_idx % 2 != 0:
                continue

            left_image, targets = left_image.to(device), left_points.to(device)
            left_points = [[f"{value:.1f}" for value in point] for point in left_points.tolist()]
            outputs = model(left_image).squeeze()

            outputs = outputs.view_as(targets)

            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            
            outputs_denorm = denormalize_2d_points(outputs.to('cpu'))
            targets_denorm = denormalize_2d_points(targets.to('cpu'))

            outputs_denorm = torch.tensor(outputs)
            targets_denorm = torch.tensor(targets)

            # Calculate RMSE for each point
            batch_rmse = torch.sqrt(torch.mean((outputs_denorm - targets_denorm)**2, dim=1))
            total_rmse += torch.sum(batch_rmse).item()
            num_samples += len(batch_rmse)

            # Calculate distance for each point
            batch_distance = torch.sqrt(torch.sum((outputs_denorm - targets_denorm)**2, dim=1))
            total_distance += torch.sum(batch_distance).item()
            

    test_loss /= len(eval_loader.dataset)

    print(f'Original points: {targets_denorm}')
    print(f'Predicted points: {outputs_denorm}')

    print('\nTest set: Average loss: {:.4f}'.format(
        test_loss, len(eval_loader.dataset)))

    # Calculate average distance
    average_distance = total_distance / num_samples
    print(f'Average distance: {average_distance:.4f}')
    print()

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


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.2, metavar='M',
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

    
    # Initialize the model
    model = NeedleTipCNN().to(device)
    if device_ids is not None:
        model = nn.DataParallel(model, device_ids=device_ids)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)

    scheduler = StepLR(optimizer, step_size=3, gamma=args.gamma)

    # Load the previously trained model
    #model.load_state_dict(torch.load("Saved_Models/SimpleNet_2D.pt"))
    #model.eval()

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, scheduler, epoch)
        test_loss, avg_dist = test(model, device, eval_loader)

        #if avg_dist < 70:
        #    break

        # Check for early stopping
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if args.save_model:
        torch.save(model.state_dict(), "Saved_Models/SimpleNet_2D.pt")


if __name__ == '__main__':
    main()
