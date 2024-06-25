#!/usr/bin/env python3

import socket
import numpy as np
import cv2
import io


HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

def calculate_3d_points(image1, image2):
    # Placeholder for the actual 3D points calculation logic
    # For demonstration, we'll return a dummy 3D point
    return "3D Point: (581.4, -345.46, 283.97)"

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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            images_data = receive_images(conn)

            # Display the received images using OpenCV
            for i, image_data in enumerate(images_data):
                # Convert image data to numpy array for OpenCV
                image_np = np.frombuffer(image_data, dtype=np.uint8)
                # Decode image
                image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                
                # Display the image
                cv2.imshow(f"Image {i+1}", image_cv)
                cv2.waitKey(5)  # Wait for any key press to close the image window

            cv2.destroyAllWindows()  # Close all OpenCV windows

            # Calculate the 3D points from the images
            if len(images_data) == 2:
                points_3d = calculate_3d_points(image_cv, image_cv)
            else:
                points_3d = "Error: Expected exactly 2 images but received {len(images_data)}."

            # Send the 3D points back to the client
            conn.sendall(points_3d.encode('utf-8'))

if __name__ == "__main__":
    main()
