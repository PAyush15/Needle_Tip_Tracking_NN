import torch
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from prep_dataset_csv import test_loader

def load_calibration_parameters(calibration_file):
    with open(calibration_file) as file:
        calibration_data = yaml.load(file, Loader=yaml.FullLoader)
    
    CameraMatrixL = np.array(calibration_data['CameraMatrixL']['data'], dtype=np.float64).reshape(3, 3)
    DistortionL = np.array(calibration_data['DistortionL']['data'], dtype=np.float64).reshape(1, 5)
    CameraMatrixR = np.array(calibration_data['CameraMatrixR']['data'], dtype=np.float64).reshape(3, 3)
    DistortionR = np.array(calibration_data['DistortionR']['data'], dtype=np.float64).reshape(1, 5)
    R = np.array(calibration_data['Rotation']['data'], dtype=np.float64).reshape(3, 3)
    T = np.array(calibration_data['Translation']['data'], dtype=np.float64).reshape(3, 1)
    
    return CameraMatrixL, DistortionL, CameraMatrixR, DistortionR, R, T


def undistort_points(point, camera_matrix, dist_coeffs):
    point = np.array([point], dtype=np.float32)
    undistorted_point = cv2.undistortPoints(point, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted_point[0][0]


def triangulate_3d_point(point1, point2, calibration_file):
    CameraMatrixL, DistortionL, CameraMatrixR, DistortionR, R, T = load_calibration_parameters(calibration_file)
    
    # Undistort points
    undistorted_point1 = undistort_points(point1, CameraMatrixL, DistortionL)
    undistorted_point2 = undistort_points(point2, CameraMatrixR, DistortionR)
    
    # Projection matrices for each camera
    P1 = np.hstack((CameraMatrixL, np.zeros((3, 1))))  # P1 = [K1 | 0]
    P2 = np.hstack((np.dot(CameraMatrixR, R), np.dot(CameraMatrixR, T)))  # P2 = [K2 * R | K2 * T]
    
    # Triangulate the 3D point
    points_4d = cv2.triangulatePoints(P1, P2, undistorted_point1.reshape(2, 1), undistorted_point2.reshape(2, 1))
    
    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d / points_4d[3]
    
    return points_3d[:3].ravel()


def detect_needle_tip_left(tensor, threshold_value=90, roi_height_ratio=0.2, roi_width_ratio=0.2):
    
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
    

def detect_needle_tip_right(tensor, threshold_value=90, roi_height_ratio=0.2, roi_width_ratio=0.2):
    
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
    roi_width = int(image.shape[1] * roi_width_ratio)
    roi = image[int(image.shape[0] * roi_height_ratio):int(image.shape[0] * (1 - roi_height_ratio)),
                roi_width:]

    # Apply thresholding to the ROI and invert the image
    _, binary_roi = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY)
    binary_roi_inverted = cv2.bitwise_not(binary_roi)

     # Find contours in the binary ROI
    contours, _ = cv2.findContours(binary_roi_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the rightmost contour (assuming it represents the needle)
    if contours:
        needle_contour = min(contours, key=lambda x: cv2.boundingRect(x)[0])

        x, y, w, h = cv2.boundingRect(needle_contour)

        # Adjust the coordinates to the original image coordinates 
        x += roi_width
        y += int(image.shape[0] * roi_height_ratio)

        # Draw a rectangle around the detected needle tip
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # print(f'Detected coord: {x+(w//2), y+(h//2)}')
        cv2.destroyAllWindows()
        return [x+(w//2), y+(h//2)]
    

def test(device, test_loader, calib_file):
    total_mae = 0.0
    num_samples = 0
    total_distance = 0.0
    mae_list = []
    
    for batch_idx, (left_images, right_images, left_points, right_points, points_3d) in enumerate(test_loader):
        left_images, points_3d = left_images.to(device), points_3d.to(device)
        
        batch_outputs = []
        for i in range(left_images.size(0)):
            img_left = left_images[i]
            img_right = right_images[i]
            p1 = detect_needle_tip_left(img_left)
            p2 = detect_needle_tip_right(img_right)

            point_3d = triangulate_3d_point(p1, p2, calib_file)

            batch_outputs.append(point_3d)

        batch_outputs = torch.tensor(batch_outputs).to(device)

        batch_mae = torch.mean(torch.abs(batch_outputs - points_3d), dim=1)
        total_mae += torch.sum(batch_mae).item()
        num_samples += len(batch_mae)
        mae_list.extend(batch_mae.tolist())

        batch_distance = torch.sqrt(torch.sum((batch_outputs - points_3d)**2, dim=1))
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
    calibration_file = 'src/models/calibration_parameter_Mon_Mar__4_12_13_17_2024.yml'
    test(device, test_loader, calibration_file)

if __name__ == '__main__':
    main()