import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights

class SiameseNetwork(nn.Module):
    
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        # Load pretrained DenseNet models
        self.densenet_left = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.densenet_right = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.config = config

        # Modify the last layer for 2D point prediction
        num_features_left = self.densenet_left.classifier.in_features
        num_features_right = self.densenet_right.classifier.in_features
        self.densenet_left.classifier = nn.Linear(num_features_left, config['num_classes'])  # 2D point prediction
        self.densenet_right.classifier = nn.Linear(num_features_right, config['num_classes'])  # 2D point prediction
        
        # Concatenation layer
        self.concatenation_layer = nn.Linear(2*num_features_left, config['num_classes'])  # Concatenating 2D points from left and right cameras
        

    def forward(self, left_image, right_image):
        # Forward pass through left and right DenseNet branches
        left_features = self.densenet_left.features(left_image)
        right_features = self.densenet_right.features(right_image)
        
        # Ensure spatial dimensions are preserved
        left_features = torch.flatten(left_features, 1)
        right_features = torch.flatten(right_features, 1)
        
        # Concatenate features
        concatenated_features = torch.cat((left_features, right_features), dim=1)

        # Adjust the size of the linear layer dynamically based on the size of the concatenated features
        num_features_concatenated = concatenated_features.shape[1]
        self.concatenation_layer = nn.Linear(num_features_concatenated, self.config['num_classes'])
        self.concatenation_layer.to(self.config['device'])

        # Prediction of 3D point based on concatenated features
        predicted_3d_point = self.concatenation_layer(concatenated_features)
        
        return predicted_3d_point
