#!/usr/bin/env python3
"""
Script for Visualising YOLO Bounding Boxes on Images
------------------------------------------------------
This script processes a dataset containing images and corresponding YOLO-format label files.
Each label file should have lines formatted as:
    <class_id> <center_x> <center_y> <width> <height>
where the coordinates are normalized (i.e. values between 0 and 1).

The script reads images from the specified DATASET_PATH (which must contain two subdirectories:
    - images: containing the image files (.jpg, .jpeg, .png)
    - labels: containing the label files (.txt) with bounding box annotations)
and draws bounding boxes on each image according to the labels. A subset of the resulting
annotated images (as defined by the NUM_IMAGES_TO_SAVE variable) are then saved to a folder called
visualised_images, which is created within the same DATASET_PATH.

This script is designed to be run in headless environments (such as via SSH) where no display is available.
"""

import os
import cv2

# Mapping from class IDs to human-readable names
CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
    4: "bicycle",
    5: "scooter"
}

# ================================================
# MODIFY THESE VARIABLES AS NEEDED
# ================================================
# Path to the dataset directory containing images and labels.
DATASET_PATH = "/mnt/nas/TAmob/old_data/scooter_datasets/scooter_dataset_V6/train"
NUM_IMAGES_TO_SAVE = 99999999999  # Define the number of images to process and save

# Define directory paths based on the DATASET_PATH.
IMAGES_DIR = os.path.join(DATASET_PATH, "images")
LABELS_DIR = os.path.join(DATASET_PATH, "labels")
OUTPUT_DIR = os.path.join(DATASET_PATH, "visualised_images")  # Output folder for annotated images

# Create the output directory if it doesn't exist.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def visualize_yolo_bounding_boxes():
    """
    Process images by drawing YOLO bounding boxes and saving annotated images.
    
    This function iterates through image files in the IMAGES_DIR, searches for a corresponding label file in the 
    LABELS_DIR, parses the YOLO-format bounding box annotations, converts the normalized coordinates to pixel 
    coordinates, draws bounding boxes along with class labels on the image, and saves the output image in the 
    OUTPUT_DIR. The function stops after processing the number of images specified in NUM_IMAGES_TO_SAVE.
    """
    saved_images = 0  # Counter to track the number of images processed and saved
    
    # Iterate over each file in the images directory.
    for img_filename in os.listdir(IMAGES_DIR):
        # Ensure the file is one of the recognized image formats.
        if img_filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(IMAGES_DIR, img_filename)
            # Construct the path to the corresponding label file (same base filename with a .txt extension).
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(LABELS_DIR, label_filename)

            # Read the image from disk.
            image = cv2.imread(image_path)
            if image is None:
                print(f"Unable to read image: {image_path}")
                continue
            
            # Retrieve image dimensions to convert normalized bbox coordinates to pixel values.
            img_height, img_width, _ = image.shape

            # Process the label file if it exists.
            if os.path.exists(label_path):
                with open(label_path, "r") as file:
                    lines = file.readlines()
                for line in lines:
                    # Expected line format: class_id center_x center_y width height
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, bbox_width, bbox_height = parts
                    try:
                        # Convert coordinate strings to float values.
                        x_center = float(x_center)
                        y_center = float(y_center)
                        bbox_width = float(bbox_width)
                        bbox_height = float(bbox_height)
                    except ValueError:
                        print("Error converting label values to float in file:", label_path)
                        continue

                    # Map numeric class_id to name
                    label_name = CLASS_NAMES.get(int(class_id), class_id)

                    # Convert the normalized coordinates to actual pixel coordinates.
                    cx_pixel = x_center * img_width
                    cy_pixel = y_center * img_height
                    w_pixel = bbox_width * img_width
                    h_pixel = bbox_height * img_height

                    # Calculate the top-left and bottom-right coordinates of the bounding box.
                    x1 = int(cx_pixel - w_pixel / 2)
                    y1 = int(cy_pixel - h_pixel / 2)
                    x2 = int(cx_pixel + w_pixel / 2)
                    y2 = int(cy_pixel + h_pixel / 2)

                    # Draw the bounding box on the image.
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    # Add the class label text above the bounding box.
                    cv2.putText(image, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), thickness=2)
            else:
                print(f"No label file found for image: {img_filename}")

            # Construct the output path and save the annotated image.
            output_path = os.path.join(OUTPUT_DIR, img_filename)
            cv2.imwrite(output_path, image)
            saved_images += 1
            print(f"Saved annotated image to {output_path}")

            # Check if we've reached the limit of images to save.
            if saved_images >= NUM_IMAGES_TO_SAVE:
                print(f"Reached the specified limit of {NUM_IMAGES_TO_SAVE} images.")
                break

if __name__ == "__main__":
    visualize_yolo_bounding_boxes()