#!/usr/bin/env python3

import socket
import os

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server
IMAGE_PATHS = [
    "imageLeft_1710238379977.000000.png", "imageRight_1710238379977.000000.png",
    "imageLeft_1710238380021.000000.png", "imageRight_1710238380021.000000.png"
]

def send_images(image_paths, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        
        for i in range(0, len(image_paths), 2):
            left_image_path = image_paths[i]
            right_image_path = image_paths[i + 1]

            # Read the left image file in binary mode
            with open(left_image_path, 'rb') as image_file:
                left_image_data = image_file.read()

            # Read the right image file in binary mode
            with open(right_image_path, 'rb') as image_file:
                right_image_data = image_file.read()

            # Send the size of the left image first (as a 4-byte integer)
            s.sendall(len(left_image_data).to_bytes(4, byteorder='big'))
            # Send the left image data
            s.sendall(left_image_data)
            print(f"Image {left_image_path} sent successfully.")

            # Send the size of the right image first (as a 4-byte integer)
            s.sendall(len(right_image_data).to_bytes(4, byteorder='big'))
            # Send the right image data
            s.sendall(right_image_data)
            print(f"Image {right_image_path} sent successfully.")
        
        # Shutdown the writing part of the socket to signal end of transmission
        s.shutdown(socket.SHUT_WR)
        
        # Receive the response from the server
        response = b''
        while True:
            data = s.recv(1024)
            if not data:
                break
            response += data
    
    # Assuming the server sends back a string representing the 3D coordinates
    print(f"Received 3D coordinates: {response.decode('utf-8')}")

if __name__ == "__main__":
    send_images(IMAGE_PATHS, HOST, PORT)
