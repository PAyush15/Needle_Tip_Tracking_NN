import torch
import torch.nn as nn
from src.models.SiameseNet import SiameseNetwork
from src.data.prep_dataset_csv import train_loader
import yaml

torch.cuda.empty_cache()

# Load configurations from config.yaml
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device(config['device'])

# Initialize Siamese network model
model = SiameseNetwork(config).to(device)
model = nn.DataParallel(model)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

# Training loop
for epoch in range(config['num_epochs']):
    model.train()
    running_loss = 0.0
    for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(train_loader):
        optimizer.zero_grad()
        left_images, right_images = left_image.to(device), right_image.to(device)
        left_2d_points, right_2d_points = left_points.to(device), right_points.to(device)
        points_3d = points_3d.to(device)
        
        print(f'Read data batch: {batch_idx}')

        # Forward pass
        predicted_points_3d = model(left_images, right_images)
        
        # Compute loss
        loss = criterion(predicted_points_3d, points_3d)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}, Loss: {running_loss}")

