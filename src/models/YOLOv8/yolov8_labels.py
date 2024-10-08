import os
import pandas as pd
from PIL import Image

# Paths
data_dir = 'src'
csv_file_path = os.path.join(data_dir, 'data/Validation_5per.csv')
images_dir = os.path.join(data_dir, 'data/data/images')
labels_dir = os.path.join(data_dir, 'data/data/labels')

# Ensure label directories exist
os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'val'), exist_ok=True)

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Function to create YOLO-formatted label file
def create_yolo_label(image_path, points, label_file_path):
    image = Image.open(image_path)
    img_width, img_height = image.size
    x, y = points

    # Normalize coordinates
    center_x = x / img_width
    center_y = y / img_height
    width = height = 0.03  # A small value to represent the point

    # YOLO format annotation: class_id, center_x, center_y, width, height
    class_id = 0  # Assuming class_id for needle tip is 0
    annotation = f"{class_id} {center_x} {center_y} {width} {height}\n"

    # Write annotation to file
    with open(label_file_path, 'w') as f:
        f.write(annotation)

# Iterate through the DataFrame and create labels
for index, row in df.iterrows():
    image_name_no_ext = row['imageLeft']
    points_str = row['Left_2D']
    points = tuple(map(float, points_str.strip('()').split(',')))  # Convert string to (x, y) tuple

    image_subdir = 'train' if os.path.exists(os.path.join(images_dir, 'train', image_name_no_ext + '.png')) else 'val'
    image_name = image_name_no_ext + '.png'
    image_path = os.path.join(images_dir, image_subdir, image_name)

    if os.path.exists(image_path):
        label_file_name = image_name_no_ext + '.txt'
        label_file_path = os.path.join(labels_dir, image_subdir, label_file_name)
        create_yolo_label(image_path, points, label_file_path)