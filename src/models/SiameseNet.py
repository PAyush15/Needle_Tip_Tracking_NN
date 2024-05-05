from __future__ import print_function
import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
from prep_dataset_csv import train_loader

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)
        
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

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
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

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.MSELoss()

    for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(train_loader):
        left_image, right_image, targets = left_image.to(device), right_image.to(device), points_3d.to(device)
        optimizer.zero_grad()
        outputs = model(left_image, right_image).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(left_image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, eval_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(eval_loader):
            left_image, right_image, targets = left_image.to(device), right_image.to(device), points_3d.to(device)
            outputs = model(left_image, right_image).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(eval_loader.dataset)

    # for the 1st epoch, the average loss is 0.0001 and the accuracy 97-98%
    # using default settings. After completing the 10th epoch, the average
    # loss is 0.0000 and the accuracy 99.5-100% using default settings.
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(eval_loader.dataset),
        100. * correct / len(eval_loader.dataset)))


def rgb2gray(rgb):
    # From https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray[:, :, None]

    return gray.astype(np.uint8)


def overlay_activation_map(imgs: list[np.ndarray], heatmaps: list[np.ndarray]) -> list:
    """
    Format the heatmaps and overlay it on the original image
    :param imgs: list with the original images
    :param heatmaps: list with the subnet outputs
    :return: list with the superimposed images
    """
    heatmaps = [np.maximum(np.mean(heatmap, axis=0), 0) / np.max(heatmap) for heatmap in heatmaps]
    heatmaps = [cv2.resize(heatmap, (imgs[0].shape[1], imgs[0].shape[0])) for heatmap in heatmaps]
    heatmaps = [np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))) for heatmap in
                heatmaps]
    heatmaps = [cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) for heatmap in heatmaps]
    # Convert images to grayscale
    if imgs[0].shape[-1] > 1:
        imgs = [np.repeat(rgb2gray(img)[:, :, :], 3, axis=2) for img in imgs]
    else:
        imgs = [np.repeat(img[:, :, :], 3, axis=2) for img in imgs]
    overlayed_imgs = [cv2.addWeighted(heatmap, 0.35, img, 0.65, 0.0) for heatmap, img in
                      zip(heatmaps, imgs)]
    return overlayed_imgs


def get_feature_maps(model: SiameseNetwork, device, dataloader, cam_inputs):
    """
    Get the saliency maps for images in the given dataloader
    :param model: Siamese model with the weights preloaded
    :param device: device used to get feature maps
    :param dataloader: dataloader with the images
    :param cam_inputs: list with the cameras being used
    :return:
    """
    with torch.no_grad():
        for i, (images, targets) in tqdm(enumerate(dataloader)):
            images = [img.to(device) for img in images]
            outputs = model.extract_features_subnets(images)
            # Reformat images
            images = [img[0, ...].detach().cpu().numpy() for img in images]
            outputs = [output[0, ...].detach().cpu().numpy() for output in outputs]
            images = [(img.transpose((1, 2, 0)) * 255).astype('uint8') for img in images]
            # Overlay feature maps
            heatmap_imgs = overlay_activation_map(images, outputs)
            # Reshape images if  grayscale
            if images[0].shape[-1] == 1:
                images = [np.repeat(img[:, :, :], 3, axis=2) for img in images]
            # Display results
            heatmaps = [np.maximum(np.mean(heatmap, axis=0), 0) / np.max(heatmap) for heatmap in outputs]
            plt.figure(figsize=(3 * len(heatmap_imgs), 7))
            img_name = dataloader.dataset.img_names[i]
            for j, (heatmap_img, heatmap, img, cam) in enumerate(zip(heatmap_imgs, heatmaps, images, cam_inputs)):
                plt.subplot(3, len(heatmap_imgs), j + 1)
                plt.title(cam)
                plt.imshow(heatmap_img)
                plt.axis('off')
                plt.subplot(3, len(heatmap_imgs), j + 4)
                if j == 0:
                    plt.title('Subnet outputs:', loc='left')
                plt.imshow(heatmap)
                # plt.axis('off')
                plt.subplot(3, len(heatmap_imgs), j + 7)
                if j == 0:
                    plt.title('Input images:', loc='left')
                plt.imshow(img)
                # plt.axis('off')
            plt.suptitle(f'{img_name} saliency map')
            plt.tight_layout()
            plt.show(block=True)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    
    model = SiameseNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(model, device, eval_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "siamese_network.pt")


if __name__ == '__main__':
    main()