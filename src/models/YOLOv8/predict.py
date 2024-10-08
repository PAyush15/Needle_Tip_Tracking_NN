import torch
import numpy as np
from ultralytics import YOLO
from prep_dataset_csv import test_loader

def calculate_iou(boxA, boxB):
    # Compute the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the width and height of the intersection rectangle
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    interArea = inter_width * inter_height
    
    # Compute the area of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def create_bounding_box(center, size=50):
    half_size = size / 2
    x_center, y_center = center
    return [
        x_center - half_size,  # x_min
        y_center - half_size,  # y_min
        x_center + half_size,  # x_max
        y_center + half_size   # y_max
    ]

def test(model, device, test_loader, bbox_size=50, iou_threshold=0.5):
    total_mae = 0.0
    total_rmse = 0.0
    num_samples = 0
    total_distance = 0.0
    total_conf = 0.0
    mae_list = []
    total_true_positives = 0

    for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(test_loader):
        
        left_image, targets_all = left_image.to(device), left_points.to(device)
        results = model(left_image, verbose=False)
        
        confidence = torch.zeros_like(targets_all, device=device)
        outputs = torch.zeros_like(targets_all, device=device)
        targets = torch.zeros_like(targets_all, device=device)

        for i, result in enumerate(results):
            
            if result.boxes.cpu().shape[0] == 1:
                boxes = result.boxes  # Boxes object for bounding box outputs
                outputs[i] = torch.tensor([boxes.cpu().xywh[0, 0], boxes.cpu().xywh[0, 1]])
                confidence[i] = torch.tensor(boxes.conf.tolist()[0])
                targets[i] = targets_all[i]
                
                # Create the predicted bounding box (50x50 centered at the predicted point)
                pred_center = outputs[i].cpu().numpy()
                pred_bbox = create_bounding_box(pred_center, bbox_size)

                # Create the target bounding box (50x50 centered at the target point)
                target_center = targets[i].cpu().numpy()
                target_bbox = create_bounding_box(target_center, bbox_size)
                
                #print(f'Predicted Point: {pred_center}')
                #print(f'Predicted Bounding Box: {pred_bbox}')
                #print(f'Target Point: {target_center}')
                #print(f'Target Bounding Box: {target_bbox}')
                
                # Calculate IoU between predicted and target bounding boxes
                iou = calculate_iou(pred_bbox, target_bbox)

                # Determine if it's a true positive
                if iou >= iou_threshold:
                    total_true_positives += 1

            else:
                continue
 
        # Calculate MAE for each point
        batch_mae = torch.mean(torch.abs(outputs - targets), dim=1)
        total_mae += torch.sum(batch_mae).item()
        num_samples += len(batch_mae)
        total_conf += torch.sum(confidence)
        mae_list.extend(batch_mae.tolist())

        # Calculate distance for each point
        batch_distance = torch.sqrt(torch.sum((outputs - targets.to(device)) ** 2, dim=1))
        total_distance += torch.sum(batch_distance).item()

    # Calculate average MAE
    average_mae = total_mae / num_samples
    mae_std = torch.std(torch.tensor(mae_list)).item()
    print(f'Average MAE: {average_mae:.4f}')
    print(f'MAE Standard Deviation: {mae_std:.4f}')

    # Calculate average distance
    average_distance = total_distance / num_samples
    avg_confidence = total_conf / num_samples
    print(f'Average distance: {average_distance:.4f}')
    print(f'Average confidence: {avg_confidence:.4f}')

    # Calculate and print IoU-based true positive percentage
    true_positive_percentage = (total_true_positives / num_samples) * 100
    print(f'IoU-based True Positive Percentage: {true_positive_percentage:.2f}%')
    print()

def main():
    # Load the model
    model = YOLO("/home/Patel/Dokumente/lightning-hydra-template/runs/detect/train/weights/best.pt")  

    device = 'cuda'

    test(model, device, test_loader, bbox_size=50, iou_threshold=0.5)

if __name__ == '__main__':
    main()
