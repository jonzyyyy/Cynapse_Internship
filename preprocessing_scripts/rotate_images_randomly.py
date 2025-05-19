import os
import random
import logging
import numpy as np
import cv2
import albumentations as A
from PIL import Image

"""USER SHOULD CHANGE AS REQUIRED"""

# Maximum absolute rotation angle in degrees (rotations will be randomly chosen from -max to +max).
MAX_ROTATION_ANGLE = 45
MIN_ROTATION_ANGLE = 5  # Minimum absolute rotation angle in degrees

# Define the base source and destination directories.
base_src = "scooter_datasets/scooter_dataset_V3/"
base_dest = "scooter_datasets/scooter_dataset_V4/"

logging.basicConfig(level=logging.INFO)
postfix = "_rotated"

"""CHANGE END"""

# Process both the 'train' and 'val' folders.
for subset in ['train', 'val']:
    img_source_folder = os.path.join(base_src, subset, "images")
    label_source_folder = os.path.join(base_src, subset, "labels")
    img_destination_folder = os.path.join(base_dest, subset, "images")
    label_destination_folder = os.path.join(base_dest, subset, "labels")
    
    os.makedirs(img_destination_folder, exist_ok=True)
    os.makedirs(label_destination_folder, exist_ok=True)
    
    for filename in os.listdir(img_source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(img_source_folder, filename)
            
            # Determine new filenames based on the postfix option.
            name, ext = os.path.splitext(filename)
            if postfix is None:
                new_filename = filename
                new_label_filename = name + ".txt"
            else:
                new_filename = name + postfix + ext
                new_label_filename = name + postfix + ".txt"
            
            new_img_path = os.path.join(img_destination_folder, new_filename)
            label_path = os.path.join(label_source_folder, name + ".txt")
            new_label_path = os.path.join(label_destination_folder, new_label_filename)
            
            try:
                with Image.open(img_path) as pil_img:
                    img_np = np.array(pil_img)
                    img_height, img_width = img_np.shape[:2]

                    # Randomly select an angle with an absolute value between MIN_ROTATION_ANGLE and MAX_ROTATION_ANGLE.
                    angle = random.uniform(MIN_ROTATION_ANGLE, MAX_ROTATION_ANGLE)
                    # Randomly decide the sign of the rotation.
                    if random.choice([True, False]):
                        angle = -angle
                    logging.info(f"Rotating {filename} by {angle:.2f}° in {subset} folder")

                    # Define the transformation.
                    transform = A.Compose(
                        [
                            A.Rotate(
                                limit=(angle, angle),
                                border_mode=cv2.BORDER_CONSTANT,
                                rotate_method="largest_box",
                                p=1.0
                            ),
                        ],
                        bbox_params=A.BboxParams(
                            format='yolo',
                            label_fields=['class_labels'],
                        )
                    )

                    annotations = []
                    # Process the corresponding label file if it exists.
                    if os.path.exists(label_path):
                        with open(label_path, "r") as lf:
                            lines = lf.readlines()
                        
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id, x, y, w, h = parts
                                x, y, w, h = map(float, (x, y, w, h))
                                # Use YOLO normalized coordinates directly
                                bbox = [x, y, w, h]
                                annotations.append((class_id, bbox))
                    
                    # Apply the transformation to the image and bounding boxes.
                    transformed = transform(image=img_np, bboxes=[bbox for _, bbox in annotations], class_labels=[class_id for class_id, _ in annotations])
                    rotated_img_np = transformed['image']
                    rotated_img_pil = Image.fromarray(rotated_img_np)

                    # Save the rotated image.
                    rotated_img_pil.save(new_img_path)

                    # Save transformed bounding boxes.
                    new_label_lines = []
                    for class_id, bbox in zip([class_id for class_id, _ in annotations], transformed['bboxes']):
                        new_x, new_y, new_w, new_h = bbox
                        new_label_lines.append(f"{class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n")
                    
                    with open(new_label_path, "w") as new_lf:
                        new_lf.writelines(new_label_lines)
                    
                    logging.info(f"Processed: {filename} in {subset} folder")
                    
            except Exception as e:
                logging.error(f"Error processing {filename} in {subset} folder: {e}")

logging.info("Augmentation complete!")