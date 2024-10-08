from __future__ import print_function
import numpy as np
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from prep_dataset_csv import  test_loader


def test(model, device, eval_loader):

    total_rmse = 0.0
    num_samples = 0
    total_distance = 0.0

    # Setup video writer
    output_path = 'yolo_vid_1.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30
    height, width = None, None
    video_writer = None

    for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(test_loader):
        
        left_image, targets = left_image.to(device), left_points.to(device)
        results = model(left_image, verbose=False)
        outputs = torch.zeros_like(targets, device=device)
        confidence = [0] * len(results)

        for i, result in enumerate(results):

            if result.boxes.cpu().shape[0] >= 1:
                    
                boxes_all = result.boxes.cpu()  # Boxes object for bounding box outputs
                boxes = boxes_all[torch.argmax(boxes_all.conf)]
                outputs[i] = torch.tensor([boxes.cpu().xywh[0,0], boxes.cpu().xywh[0,1]])

                confidence[i] = result.boxes.conf.tolist()[0]
            else:
                confidence[i] = 0

        for i in range(left_image.size(0)):  # Loop through each image in the batch
            frame = visualize(left_image[i].cpu().numpy().transpose((1, 2, 0)), outputs[i].cpu().tolist(),confidence[i])

            if video_writer is None:
                height, width, _ = frame.shape
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Convert the frame to 8-bit
            frame = cv2.convertScaleAbs(frame)
            video_writer.write(frame)


        #result.save(filename="result.jpg")  # save to disk

        # Calculate RMSE for each point
        batch_rmse = torch.sqrt(torch.mean((outputs - targets.to(device))**2, dim=1))
        total_rmse += torch.sum(batch_rmse).item()
        num_samples += len(batch_rmse)

        # Calculate distance for each point
        batch_distance = torch.sqrt(torch.sum((outputs - targets.to(device))**2, dim=1))
        total_distance += torch.sum(batch_distance).item()

            
    average_distance = total_distance / num_samples
    print(f'Average distance: {average_distance:.4f}')

    if video_writer is not None:
        video_writer.release()
        print(f'Video saved as {output_path}')


def project_points(image, point, confidence):
    projected_image = image.copy()
    x, y = point
    x, y = int(x), int(y)
    cv2.circle(projected_image, (x, y), 10, (0, 255, 0), 3)
    cv2.putText(projected_image, f"Conf: {confidence:.2f}", (int(x+10), int(y-10) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2) 
    return projected_image


def visualize(image, predicted_points, confidence):
    image = (image * 255).astype(np.uint8)  # Scale image to 0-255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    projected_image = project_points(image, predicted_points, confidence)
    return projected_image


def main():
    # Load the model
    model = YOLO("/home/Patel/Dokumente/lightning-hydra-template/runs/detect/train/weights/best.pt")  

    device = 'cuda'

    test(model, device, test_loader)


if __name__ == '__main__':
    main()
