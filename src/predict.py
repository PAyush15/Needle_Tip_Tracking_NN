import torch
import torch.nn as nn
from SiameseNet import SiameseNetwork
from src.data.prep_dataset_csv import eval_loader
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

# Set the model to evaluation mode
model.eval()
model.to(device)

total_rmse = 0.0
num_samples = 0

# Forward pass for inference
with torch.no_grad():
    for left_image, right_image, left_points, right_points, points_3d in eval_loader:
        #print(f'Left shape: {left_image.shape} and right shape: {right_image.shape}')
        left_images, right_images = left_image.to(device), right_image.to(device)
        predicted_points_3d = model(left_images, right_images)
        
        # Calculate RMSE for each point
        batch_rmse = torch.sqrt(torch.mean((predicted_points_3d - points_3d.to(device))**2, dim=1))
        total_rmse += torch.sum(batch_rmse).item()
        num_samples += len(batch_rmse)
        
        print(f'Predicted point: {predicted_points_3d}, Original point: {points_3d}')

        # Print RMSE for each predicted point in the batch
        for i, rmse in enumerate(batch_rmse):
            print(f'RMSE for predicted point {i+1}: {rmse:.4f}')      

# Calculate average RMSE
average_rmse = total_rmse / num_samples
print(f'Average RMSE: {average_rmse:.4f}')
