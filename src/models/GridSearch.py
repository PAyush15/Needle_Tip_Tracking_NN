from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import ParameterGrid
from prep_dataset_csv import train_loader, val_loader


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
        output = self.avg_pooling(output)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        #output1 = self.dropout(output1)
        #output2 = self.dropout(output2)

        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        
        return output


def train(args, model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    criterion = nn.MSELoss()

    for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(train_loader):
        #if batch_idx % 150 != 0:
        #    continue

        left_image, right_image, targets = left_image.to(device), right_image.to(device), points_3d.to(device)
        optimizer.zero_grad()
        outputs = model(left_image, right_image).squeeze()

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
    total_rmse = 0.0
    num_samples = 0
    total_distance = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(eval_loader):
            #if batch_idx % 150 != 0:
            #    continue

            left_image, right_image, targets = left_image.to(device), right_image.to(device), points_3d.to(device)
            outputs = model(left_image, right_image).squeeze()

            outputs = outputs.view_as(targets)

            test_loss += criterion(outputs, targets).sum().item()

            outputs_denorm = denormalize_3d_points(outputs.to('cpu'))
            targets_denorm = denormalize_3d_points(targets.to('cpu'))

            outputs_denorm = torch.tensor(outputs_denorm)
            targets_denorm = torch.tensor(targets_denorm)

            batch_rmse = torch.sqrt(torch.mean((outputs_denorm - targets_denorm)**2, dim=1))
            total_rmse += torch.sum(batch_rmse).item()
            num_samples += len(batch_rmse)

            batch_distance = torch.sqrt(torch.sum((outputs_denorm - targets_denorm)**2, dim=1))
            total_distance += torch.sum(batch_distance).item()

    test_loss /= len(eval_loader.dataset)
    average_distance = total_distance / num_samples

    print(f'\nTest set: Average loss: {test_loss:.4f}, Average distance: {average_distance:.4f}\n')
    return test_loss, average_distance


def denormalize_3d_points(points):
    min_values = torch.tensor([530.657593, -559.603027, 275.303467])
    max_values = torch.tensor([594.552063, -335.045776, 321.951843])
    
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


def main():
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
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

    param_grid = {
        'lr': [0.0003, 0.001, 0.003, 0.008, 0.01],
        'batch_size': [4],
        'gamma': [0.1, 0.2],
        'patience': [3, 5],
        'step_size': [2, 3, 4],
    }

    best_params = None
    best_avg_distance = float('inf')

    for params in ParameterGrid(param_grid):
        print(f'Trying parameters: {params}')
        args.lr = params['lr']
        args.batch_size = params['batch_size']
        args.gamma = params['gamma']
        args.patience = params['patience']
        step_size = params['step_size']

        model = SiameseNetwork().to(device)
        if use_cuda:
            model = nn.DataParallel(model, device_ids=device_ids)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=args.gamma)
        early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)

        # Load the previously trained model
        #model.load_state_dict(torch.load("Saved_Models/siamese_densenet_norm.pt"))
        #model.eval()

        for epoch in range(1, args.epochs + 1):
            train_loss = train(args, model, device, train_loader, optimizer, scheduler, epoch)
            test_loss, avg_distance = test(model, device, eval_loader)

            if avg_distance < best_avg_distance:
                best_avg_distance = avg_distance
                best_params = params
                print(f'New best average distance: {avg_distance}')

            early_stopping(test_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    print(f'Best parameters: {best_params}')
    print(f'Best average distance: {best_avg_distance}')

    if args.save_model:
        torch.save(model.state_dict(), "Saved_Models/siamese_densenet_best.pt")


if __name__ == '__main__':
    main()
