import torch
import cv2
import numpy as np
import yaml
import csv
from ultralytics import YOLO
from prep_dataset_csv import test_loader

def test(model_left, model_right, device, test_loader):
    total_mae = 0.0
    total_distance = 0.0
    num_samples = 0
    mae_list = []

    # Open CSV file to write data
    with open('Y3D/Y3D_Traj_13.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Left_2D', 'Right_2D', 'Point_3d', 'Pred_Left', 'Pred_Right'])

        for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(test_loader):
            left_image, right_image, left_point, right_point, point_3d = left_image.to(device), right_image.to(device), left_points.to(device), right_points.to(device), points_3d.to(device)
            results_left = model_left(left_image, verbose=False)
            results_right = model_right(right_image, verbose=False)

            targets = torch.zeros_like(left_point)
            outputs = torch.zeros_like(left_point, device=device)

            for i, result in enumerate(results_left):

                if result.boxes.cpu().shape[0] >= 1 and results_right[i].boxes.cpu().shape[0] >= 1:

                    boxes_left = result.boxes.cpu()
                    boxes_right = results_right[i].boxes.cpu()

                    if len(boxes_left) > 1:
                        boxes_left = boxes_left[torch.argmax(boxes_left.conf)]
                    if len(boxes_right) > 1:
                        boxes_right = boxes_right[torch.argmax(boxes_right.conf)]

                    left_pt_tensor = torch.tensor([boxes_left.cpu().xywh[0, 0], boxes_left.cpu().xywh[0, 1]])
                    right_pt_tensor = torch.tensor([boxes_right.cpu().xywh[0, 0], boxes_right.cpu().xywh[0, 1]])

                    # Round the tensors to nearest integer
                    left_pt_list = [round(val.item(), 2) for val in left_pt_tensor]
                    right_pt_list = [round(val.item(), 2) for val in right_pt_tensor]

                    outputs[i] = left_pt_tensor
                    targets[i] = left_point[i]

                    # Convert tensors to lists for CSV writing
                    left_point_target = f"({left_point[i][0].item()}, {left_point[i][1].item()})"
                    right_point_target = f"({right_point[i][0].item()}, {right_point[i][1].item()})"
                    point_3d_target = f"({point_3d[i][0].item()}, {point_3d[i][1].item()}, {point_3d[i][2].item()})"
                    predicted_left = f"({left_pt_list[0]}, {left_pt_list[1]})"
                    predicted_right = f"({right_pt_list[0]}, {right_pt_list[1]})"

                    # Write to CSV
                    writer.writerow([left_point_target, right_point_target, point_3d_target, predicted_left, predicted_right])

                    #print("outputs: ", outputs[i])
                    #print("targets: ", targets[i])

                else:
                    continue

            # Calculate MAE for each point
            batch_mae = torch.mean(torch.abs(outputs - targets), dim=1)
            total_mae += torch.sum(batch_mae).item()
            num_samples += len(batch_mae)
            mae_list.extend(batch_mae.tolist())

            # Calculate distance for each point
            batch_distance = torch.sqrt(torch.sum((outputs - targets.to(device))**2, dim=1))
            total_distance += torch.sum(batch_distance).item()

    # Calculate average MAE
    average_mae = total_mae / num_samples
    mae_std = torch.std(torch.tensor(mae_list)).item()
    print(f'Average MAE: {average_mae:.4f}')
    print(f'MAE Standard Deviation: {mae_std:.4f}')

    # Calculate average distance
    average_distance = total_distance / num_samples
    print(f'Average distance: {average_distance:.4f}')
    print()

def main():
    # Load the model
    model_left = YOLO("/home/Patel/Dokumente/lightning-hydra-template/runs/detect/train/weights/best.pt")  
    model_right = YOLO("/home/Patel/Dokumente/lightning-hydra-template/runs/detect/train7/weights/best.pt")  

    device = 'cuda'

    test(model_left, model_right, device, test_loader)

if __name__ == '__main__':
    main()
