"""
trainvalsplit_rand.py

This script is designed to split a dataset of images (and their corresponding label files)
into training and validation sets. The source directory should contain two subfolders:
    - images: contains image files (e.g. .jpg, .jpeg, .png)
    - labels: contains corresponding label files (with the same base filename and a .txt extension)
The user only needs to specify the source directory (SRC_PATH) and a destination base directory (DEST_BASE).

The script performs the following steps:
    1. Sets up logging for status messages.
    2. Defines user-specified paths and the splitting ratio (default is 85% for training).
    3. Creates the required destination subdirectories:
         - train/images
         - train/labels
         - val/images
         - val/labels
    4. (Optionally) Clears existing files in the destination directories.
    5. Retrieves all image files from the 'images' subfolder in the source directory.
    6. Randomly shuffles and splits the image list based on the specified TRAIN_RATIO.
    7. Copies each image and its corresponding label file (from the 'labels' subfolder) into
       the appropriate destination subdirectory.
    8. Logs warnings if any expected label file is missing.

This script helps in organising datasets for training machine learning models by ensuring that both
images and labels are consistently split into training and validation sets.
"""

import os
import random
import shutil
import logging

# Configure logging to output timestamped information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set a seed to ensure reproducibility in the random shuffling process
SEED = 42
random.seed(SEED)

# ------------------------------------------------------------------------------
# User-Specified Paths and Splitting Ratio
# ------------------------------------------------------------------------------
# SRC_PATH: Directory containing the 'images' and 'labels' subfolders.
# DEST_BASE: Base destination directory where the split datasets will be saved.
SRC_PATH = "/mnt/nas/TAmob/old_data/final_extracted_frames/05_05_2025 09_30_00 (UTC+03_00)_processed_fr20_10.197.21.23/_relabeled/obj_train_data/"
DEST_BASE = "/mnt/nas/TAmob/old_data/final_extracted_frames/05_05_2025 09_30_00 (UTC+03_00)_processed_fr20_10.197.21.23/05_05_2025 09_30_00 (UTC+03_00)_processed_fr20_10.197.21.23_relabeled_split/"

# TRAIN_RATIO: Proportion of the dataset to be used for training.
TRAIN_RATIO = 0.85

# ------------------------------------------------------------------------------
# Define Source Subdirectories for Images and Labels
# ------------------------------------------------------------------------------
# images_dir: Path to the subfolder containing image files.
# labels_dir: Path to the subfolder containing corresponding label files.
images_dir = os.path.join(SRC_PATH, "images")
labels_dir = os.path.join(SRC_PATH, "labels")

# ------------------------------------------------------------------------------
# Define Destination Subdirectories for Training and Validation
# ------------------------------------------------------------------------------
# The splits dictionary maps the split type ('train' or 'val') to its corresponding
# images and labels subdirectories.
splits = {
    "train": {
        "images": os.path.join(DEST_BASE, "train", "images"),
        "labels": os.path.join(DEST_BASE, "train", "labels")
    },
    "val": {
        "images": os.path.join(DEST_BASE, "val", "images"),
        "labels": os.path.join(DEST_BASE, "val", "labels")
    }
}

# Create the destination subdirectories if they do not already exist.
for split in splits:
    for subfolder in splits[split].values():
        os.makedirs(subfolder, exist_ok=True)

# ------------------------------------------------------------------------------
# Function: clear_directory
# ------------------------------------------------------------------------------
def clear_directory(directory):
    """
    Clears all files in the specified directory.

    This function iterates through all files in the provided directory and removes them.
    It is useful for ensuring that the destination folders are empty before copying new files.

    Parameters:
        directory (str): The path to the directory to be cleared.
    """
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            os.remove(file_path)
        logging.info(f"Cleared files in: {directory}")

# Optionally clear destination folders to start with empty directories.
for split in splits:
    clear_directory(splits[split]["images"])
    clear_directory(splits[split]["labels"])

# ------------------------------------------------------------------------------
# Retrieve and Shuffle Image Files from the Source Images Subfolder
# ------------------------------------------------------------------------------
# This retrieves all files with image extensions (case-insensitive) from the images subfolder.
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(image_files)

# ------------------------------------------------------------------------------
# Split the Image List into Training and Validation Sets
# ------------------------------------------------------------------------------
split_index = int(len(image_files) * TRAIN_RATIO)
split_files = {
    "train": image_files[:split_index],
    "val": image_files[split_index:]
}

logging.info(f"Training set: {len(split_files['train'])} images")
logging.info(f"Validation set: {len(split_files['val'])} images")

# ------------------------------------------------------------------------------
# Function: copy_file_safe
# ------------------------------------------------------------------------------
def copy_file_safe(src, dst):
    """
    Copies a file from the source path to the destination path.

    If the destination file already exists, a warning is logged and the file is not overwritten.

    Parameters:
        src (str): The source file path.
        dst (str): The destination file path.
    """
    if os.path.exists(dst):
        logging.warning(f"Skipping {dst}, file already exists.")
        return
    shutil.copy2(src, dst)

# ------------------------------------------------------------------------------
# Loop Through the Files and Copy Images and Corresponding Labels
# ------------------------------------------------------------------------------
for split, files in split_files.items():
    for file_name in files:
        # Construct source and destination paths for the image file.
        src_img = os.path.join(images_dir, file_name)
        dst_img = os.path.join(splits[split]["images"], file_name)
        copy_file_safe(src_img, dst_img)

        # Construct the label file name by replacing the image extension with .txt.
        label_name = os.path.splitext(file_name)[0] + ".txt"
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(splits[split]["labels"], label_name)
        
        # Copy the label file if it exists; otherwise, log a warning.
        if os.path.exists(src_label):
            copy_file_safe(src_label, dst_label)
        else:
            logging.warning(f"Label file {src_label} does not exist.")

logging.info("Dataset split completed successfully!")