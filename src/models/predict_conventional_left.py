import torch
import cv2
import numpy as np
from ultralytics import YOLO
from prep_dataset_csv import test_loader


def detect_needle_tip(tensor, threshold_value=90, roi_height_ratio=0.2, roi_width_ratio=0.2):
    
    # Convert the tensor to a NumPy array directly
    frame = tensor.squeeze().cpu().numpy()
    frame = (frame * 255).astype(np.uint8)
    
    if frame.shape[0] == 3:  # If the tensor has 3 channels, it's likely RGB
        frame = frame.transpose((1, 2, 0))  # Convert to (H, W, C) format
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert RGB to Grayscale
    elif frame.shape[0] == 1:  # If it has 1 channel, it's already grayscale
        image = frame[0]  # Take the first channel (as it's the only one)
    else:
        raise ValueError(f"Unexpected number of channels: {frame.shape[0]}")

    #cv2.imshow("Image", image)
    #cv2.waitKey(3000)
    #print("Shape", image.shape)
    
    # Set a region of interest (ROI) excluding both the top, bottom, and right edges
    roi_height = int(image.shape[0] * (1 - 2 * roi_height_ratio))
    roi_width = int(image.shape[1] * (1 - roi_width_ratio))
    roi = image[int(image.shape[0] * roi_height_ratio):int(image.shape[0] * (1 - roi_height_ratio)),
                :roi_width]

    # Apply thresholding to the ROI and invert the image
    _, binary_roi = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY)
    binary_roi_inverted = cv2.bitwise_not(binary_roi)

    # Erosion and dilation to remove noise
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_roi_inverted = cv2.erode(binary_roi_inverted, element, iterations=1)
    binary_roi_inverted = cv2.dilate(binary_roi_inverted, element, iterations=1)

     # Find contours in the binary ROI
    contours, _ = cv2.findContours(binary_roi_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the rightmost contour (assuming it represents the needle)
    if contours:
        needle_contour = max(contours, key=lambda x: cv2.boundingRect(x)[0] + cv2.boundingRect(x)[2])

        x, y, w, h = cv2.boundingRect(needle_contour)

        # Adjust the coordinates to the original image coordinates 
        y += int(image.shape[0] * roi_height_ratio)

        # Draw a rectangle around the detected needle tip
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # print(f'Detected coord: {x+(w//2), y+(h//2)}')
        cv2.destroyAllWindows()
        return [x+(w//2), y+(h//2)]
    

def test(device, test_loader):
    total_mae = 0.0
    num_samples = 0
    total_distance = 0.0
    mae_list = []
    
    for batch_idx, (left_images, right_images, left_points, right_points, points_3d) in enumerate(test_loader):
        left_images, left_points = left_images.to(device), left_points.to(device)
        
        batch_outputs = []
        for i in range(left_images.size(0)):
            single_image = left_images[i]
            output = detect_needle_tip(single_image)
            batch_outputs.append(output)

        batch_outputs = torch.tensor(batch_outputs).to(device)

        batch_mae = torch.mean(torch.abs(batch_outputs - left_points), dim=1)
        total_mae += torch.sum(batch_mae).item()
        num_samples += len(batch_mae)
        mae_list.extend(batch_mae.tolist())

        batch_distance = torch.sqrt(torch.sum((batch_outputs - left_points)**2, dim=1))
        total_distance += torch.sum(batch_distance).item()

    average_mae = total_mae / num_samples
    mae_std = torch.std(torch.tensor(mae_list)).item()
    print(f'Average MAE: {average_mae:.4f}')
    print(f'MAE Standard Deviation: {mae_std:.4f}')

    average_distance = total_distance / num_samples
    print(f'Average distance: {average_distance:.4f}')
    print()


def main():
    
    device = 'cuda'
    test(device, test_loader)

if __name__ == '__main__':
    main()