import socket
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from ultralytics import YOLO


HOST = "134.28.45.101"  
PORT = 5005  


def test(model, device, test_loader):

    total_rmse = 0.0
    num_samples = 0
    total_distance = 0.0
    
    for batch_idx, (left_image, right_image, left_points, right_points, points_3d) in enumerate(test_loader):
        
        left_image, targets = left_image.to(device), left_points.to(device)
        results = model(left_image)
        outputs = torch.zeros_like(targets, device=device)

        # Process results list
        for i, result in enumerate(results):
            boxes = result.boxes  # Boxes object for bounding box outputs
            outputs[i] = torch.tensor([boxes.cpu().xywh[0,0], boxes.cpu().xywh[0,1]])

        #result.save(filename="result.jpg")  # save to disk

        # Calculate RMSE for each point
        batch_rmse = torch.sqrt(torch.mean((outputs - targets.to(device))**2, dim=1))
        total_rmse += torch.sum(batch_rmse).item()
        num_samples += len(batch_rmse)

        # Calculate distance for each point
        batch_distance = torch.sqrt(torch.sum((outputs - targets.to(device))**2, dim=1))
        total_distance += torch.sum(batch_distance).item()

    # Calculate average distance
    rmse = total_rmse / num_samples
    print(f'RMSE: {rmse:.4f}')
    print()

    # Calculate average distance
    average_distance = total_distance / num_samples
    print(f'Average distance: {average_distance:.4f}')
    print()


def receive_images(conn):
    images = []
    for i in range(2):
        # Receive the size of the image data (4 bytes)
        size_data = conn.recv(4)
        if not size_data:
            break
        image_size = int.from_bytes(size_data, byteorder='big')
        
        # Receive the image data until we have received the expected amount
        image_data = b''
        while len(image_data) < image_size:
            packet = conn.recv(image_size - len(image_data))
            if not packet:
                break
            image_data += packet
        
        images.append(image_data)
        print(f"Image {i+1} received, size: {len(image_data)} bytes")

    return images

def main():
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0, 1, 2, 3]
    model = YOLO("/home/Patel/Dokumente/lightning-hydra-template/runs/detect/train/weights/best.pt")  
    model = nn.DataParallel(model, device_ids=device_ids)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                images_data = receive_images(conn)
                
                # Check if received images data is not empty
                if not images_data:
                    print("No images received, ending connection.")
                    break

                # Decode images and prepare them for the model
                images = []
                for i, image_data in enumerate(images_data):
                    # Convert image data to numpy array for OpenCV
                    image_np = np.frombuffer(image_data, dtype=np.uint8)
                    # Decode image
                    image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                    
                    # Display the image
                    #cv2.imshow(f"Image {i+1}", image_cv)
                    #cv2.waitKey(1000)  # Wait for 1000ms

                    # Convert image to tensor
                    image_tensor = torchvision.transforms.functional.to_tensor(image_cv).unsqueeze(0)
                    images.append(image_tensor)

                cv2.destroyAllWindows()  # Close all OpenCV windows

                # Ensure we have exactly 2 images
                if len(images) == 2:
                    left_img, right_img = images[0], images[1]
                    points_3d = test(model, device, left_img, right_img)
                    points_3d_str = f"3D Point: {points_3d.numpy().tolist()}"
                else:
                    points_3d_str = f"Error: Expected exactly 2 images but received {len(images)}."

                # Send the 3D points back to the client
                conn.sendall(points_3d_str.encode('utf-8'))

if __name__ == "__main__":
    main()
