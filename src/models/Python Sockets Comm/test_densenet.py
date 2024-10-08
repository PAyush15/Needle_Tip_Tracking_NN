from __future__ import print_function
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import DenseNet121_Weights
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from prep_dataset_csv import train_loader, eval_loader, test_loader


class SiameseNetwork(nn.Module):
    """
        Siamese network for predicting the 3D points using 2 input images.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get densenet model
        self.densenet = torchvision.models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)


        self.densenet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.densenet.classifier.in_features

        # remove the last layer of densenet18 (linear layer which is before avgpool layer)
        self.densenet = torch.nn.Sequential(*(list(self.densenet.children())[:-2]))

        # Init average pool layer that needs to be manually added to the EfficientNet feature extractor
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )

        # initialize the weights
        self.densenet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

        self.dropout = nn.Dropout(0.5)  # Add dropout layer
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.densenet(x)
        output = self.avg_pooling(output)  # Shape: [batch, # feature maps, 1, 1]
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output1 = self.dropout(output1)  # Apply dropout
        output2 = self.dropout(output2)  # Apply dropout

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        return output


def test(model, device, left_image, right_image):
    model.eval()
    test_loss = 0
    total_rmse = 0.0
    num_samples = 0
    total_distance = 0.0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.MSELoss()

    with torch.no_grad():

        left_image, right_image = left_image.to(device), right_image.to(device)
        outputs = model(left_image, right_image).squeeze()

        outputs_denorm = denormalize_3d_points(outputs.to('cpu'))

        outputs_denorm = torch.tensor(outputs_denorm)

        print(f'Predicted points: {outputs_denorm}')

    return outputs_denorm


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


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray[:, :, None]

    return gray.astype(np.uint8)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

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

    # Load the previously trained model
    model.load_state_dict(torch.load("Saved_Models/siamese_densenet_norm_v02.pt"))
    model.eval()

    outputs_denormalized = test(model, device, left_img, right_img)


if __name__ == '__main__':
    main()