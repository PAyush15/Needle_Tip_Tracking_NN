#!/usr/bin/env python3

import socket
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.models import DenseNet121_Weights

HOST = "127.0.0.1"  
PORT = 65432  

class SiameseNetwork(nn.Module):
    """
    Siamese network for predicting the 3D points using 2 input images.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get densenet model
        self.densenet = torchvision.models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        self.densenet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.densenet.classifier.in_features

        # remove the last layer of densenet121 (linear layer which is before avgpool layer)
        self.densenet = torch.nn.Sequential(*(list(self.densenet.children())[:-2]))

        # Init average pool layer that needs to be manually added to the DenseNet feature extractor
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )

        # initialize the weights
        self.densenet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

        self.dropout = nn.Dropout(0.5)  # Add dropout layer
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.densenet(x)
        output = self.avg_pooling(output)  # Shape: [batch, # feature maps, 1, 1]
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output1 = self.dropout(output1)  # Apply dropout
        output2 = self.dropout(output2)  # Apply dropout

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        return output

def denormalize_3d_points(point):
    """
    Denormalize multiple points from the 0 to 1 range back to the original range.
    """
    min_values = torch.tensor([530.657593, -559.603027, 275.303467])
    max_values = torch.tensor([594.552063, -335.045776, 321.951843])
    
    denormalized_points = []
    point_tensor = torch.tensor(point)
    denorm_point = point_tensor * (max_values - min_values) + min_values
    denormalized_points.append(denorm_point.tolist())
    
    return denormalized_points

def test(model, device, left_image, right_image):
    model.eval()
    with torch.no_grad():
        left_image, right_image = left_image.to(device), right_image.to(device)
        outputs = model(left_image, right_image).squeeze()

        outputs_denorm = denormalize_3d_points(outputs.to('cpu'))
        outputs_denorm = torch.tensor(outputs_denorm)

        print(f'Predicted points: {outputs_denorm}')

    return outputs_denorm

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
    model = SiameseNetwork().to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load("siamese_densenet_norm.pt"))
    model.eval()

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
                    
                    '''
                    # Display the image
                    cv2.imshow(f"Image {i+1}", image_cv)
                    cv2.waitKey(500)  # Wait for 500ms
                    '''

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
