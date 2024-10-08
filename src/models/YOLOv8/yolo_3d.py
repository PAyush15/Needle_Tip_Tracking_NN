import torch
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from prep_dataset_csv import test_loader

def load_calibration_parameters(calibration_file):
    with open(calibration_file) as file:
        calibration_data = yaml.load(file, Loader=yaml.FullLoader)

    CameraMatrixL = np.array(calibration_data['CameraMatrixL']['data'], dtype=np.float32).reshape(3, 3)
    DistortionL = np.array(calibration_data['DistortionL']['data'], dtype=np.float32).reshape(1, 5)
    CameraMatrixR = np.array(calibration_data['CameraMatrixR']['data'], dtype=np.float32).reshape(3, 3)
    DistortionR = np.array(calibration_data['DistortionR']['data'], dtype=np.float32).reshape(1, 5)
    RotationMatrixL = np.array(calibration_data['RotationMatrixL']['data'], dtype=np.float32).reshape(3, 3)
    RotationMatrixR = np.array(calibration_data['RotationMatrixR']['data'], dtype=np.float32).reshape(3, 3)
    ProjectionMatrixL = np.array(calibration_data['ProjectionMatrixL']['data'], dtype=np.float32).reshape(3, 4)
    ProjectionMatrixR = np.array(calibration_data['ProjectionMatrixR']['data'], dtype=np.float32).reshape(3, 4)

    return CameraMatrixL, DistortionL, CameraMatrixR, DistortionR, RotationMatrixL, RotationMatrixR, ProjectionMatrixL, ProjectionMatrixR

def undistort_points(point, camera_matrix, dist_coeffs, rotation_matrix, projection_matrix):
    point = np.array([point], dtype=np.float32)
    undistorted_point = cv2.undistortPoints(point, camera_matrix, dist_coeffs, R=rotation_matrix, P=projection_matrix)
    return undistorted_point[0][0]

def triangulate_3d_point(point1, point2, calibration_file):
    CameraMatrixL, DistortionL, CameraMatrixR, DistortionR, RotationMatrixL, RotationMatrixR, ProjectionMatrixL, ProjectionMatrixR = load_calibration_parameters(calibration_file)
    
    # Undistort the points using the provided rotation matrix and camera matrix
    undistorted_point1 = undistort_points(point1, CameraMatrixL, DistortionL, RotationMatrixL, ProjectionMatrixL)
    undistorted_point2 = undistort_points(point2, CameraMatrixR, DistortionR, RotationMatrixR, ProjectionMatrixR)
    
    # Ensure the points are in the correct shape (2, 1)
    undistorted_point1 = np.array(undistorted_point1).reshape(2, 1)
    undistorted_point2 = np.array(undistorted_point2).reshape(2, 1)
    
    # Triangulate the 3D point
    points_4d = cv2.triangulatePoints(ProjectionMatrixL, ProjectionMatrixR, undistorted_point1, undistorted_point2)
    
    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3] / points_4d[3]
    #points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

    return points_3d.ravel()

def test(model_left, model_right, device, test_loader, calibration_file):
    total_mae = 0.0
    total_distance = 0.0
    num_samples = 0
    mae_list = []
    
    for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(test_loader):
        left_image, right_image, lft, rgt, targets_all = left_image.to(device), right_image.to(device), left_points.to(device), right_points.to(device), points_3d.to(device)
        results_left = model_left(left_image, verbose=False)
        results_right = model_right(right_image, verbose=False)

        targets = torch.zeros_like(targets_all)
        outputs = torch.zeros_like(targets_all, device=device)

        for i, result in enumerate(results_left):
            if result.boxes.cpu().shape[0] == 1 and results_right[i].boxes.cpu().shape[0] == 1:
                boxes_left = result.boxes
                boxes_right = results_right[i].boxes
                left_pt_tensor = torch.tensor([boxes_left.cpu().xywh[0,0], boxes_left.cpu().xywh[0,1]])
                right_pt_tensor = torch.tensor([boxes_right.cpu().xywh[0,0], boxes_right.cpu().xywh[0,1]])

                print("left and right outputs: ", left_pt_tensor, "and ", right_pt_tensor)
                print("left and right targets: ", lft[i], "and ", rgt[i])

                # Triangulate the 3D point
                outputs[i] = torch.tensor(triangulate_3d_point(left_pt_tensor, right_pt_tensor, calibration_file))
                targets[i] = targets_all[i]
                print("Targets: ", targets[i])
                print("Outputs: ", outputs[i])
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
    calibration_file = 'src/models/calibration_parameter_Mon_Mar__4_12_13_17_2024.yml'

    device = 'cuda'

    test(model_left, model_right, device, test_loader, calibration_file)

if __name__ == '__main__':
    main()
