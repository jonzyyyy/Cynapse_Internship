import os
import shutil
import logging
import random

# ------------------------------------------------------------------------------
# Global Parameters and Logging Setup
# ------------------------------------------------------------------------------

# Set to True to actually delete files; False to simulate deletion.
activateDelete = False

# Define the unsorted source directory containing .jpg and .txt files.
# This directory should be structured as:
# unsorted_source_dir/
#   ├── (all your .jpg and .txt files)
unsorted_source_dir = "scooter_datasets/new_scooter_datasets/new_scooter_dataset_updated/obj_train_data/"

# After sorting, images and labels will be organised in these subdirectories:
jpg_folder = os.path.join(unsorted_source_dir, "images")
txt_folder = os.path.join(unsorted_source_dir, "labels")

# Parameters for the train/validation split
SRC_PATH = unsorted_source_dir  # The source now should contain sorted subfolders "images" and "labels"
DEST_BASE = "scooter_datasets/new_scooter_datasets/new_scooter_dataset_updated_split/"
TRAIN_RATIO = 0.85  # Proportion of data used for training
SEED = 42  # Seed for reproducibility

# Logging configuration
def setup_logger():
    logger = logging.getLogger("FileCleaner")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# ------------------------------------------------------------------------------
# File-Cleaning and Sorting Functions
# ------------------------------------------------------------------------------

def get_filenames(folder, extension):
    """Retrieve a set of filenames (without extensions) from a folder."""
    return {os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(extension)}

def delete_files(folder, filenames, extension):
    """Delete files that do not have corresponding pairs."""
    deleted_count = 0
    for file_name in filenames:
        file_path = os.path.join(folder, f"{file_name}{extension}")
        if os.path.exists(file_path):
            if activateDelete:
                os.remove(file_path)
                logger.info(f"Deleted {file_path}")
            deleted_count += 1
    return deleted_count

def remove_unmatched_files():
    """
    Finds and removes unmatched files between the images and labels folders.
    Unmatched files are those .jpg files without a corresponding .txt file and vice versa.
    """
    txt_files = get_filenames(txt_folder, ".txt")
    jpg_files = get_filenames(jpg_folder, ".jpg")

    unmatched_txt_files = txt_files - jpg_files  # TXT files without JPG counterparts
    unmatched_jpg_files = jpg_files - txt_files  # JPG files without TXT counterparts

    txt_deleted = delete_files(txt_folder, unmatched_txt_files, ".txt")
    jpg_deleted = delete_files(jpg_folder, unmatched_jpg_files, ".jpg")

    logger.info(f"{jpg_deleted} unmatched .jpg files and {txt_deleted} unmatched .txt files deleted!")

def check_empty_txt_files():
    """Check if any TXT files are empty and delete them."""
    empty_count = 0
    for file_name in os.listdir(txt_folder):
        file_path = os.path.join(txt_folder, file_name)
        if file_name.endswith(".txt"):
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    if activateDelete:
                        os.remove(file_path) 
                        logger.info(f"Deleted empty file: {file_path}")
                    empty_count += 1
    logger.info(f"{empty_count} empty .txt files deleted!")

def remove_unwanted_txt_files(classes):
    """
    Removes .txt files that start with any of the unwanted class IDs.
    
    Parameters:
        classes (list): List of unwanted class IDs.
    """
    unwanted_files_count = 0
    for file_name in os.listdir(txt_folder):
        file_path = os.path.join(txt_folder, file_name)
        if file_name.endswith(".txt"):
            with open(file_path, 'r') as f:
                file_deleted = False
                for line in f:
                    content = line.strip()
                    for id in classes:
                        if content.startswith(str(id)):
                            if activateDelete:
                                os.remove(file_path)
                                logger.info(f"Deleted {file_path} because a line started with {id}.")
                            unwanted_files_count += 1
                            file_deleted = True
                            break
                    if file_deleted:
                        break
    logger.info(f"{unwanted_files_count} unwanted .txt files deleted!")

def sort_labels_files(source_dir):
    """
    Sorts .jpg and .txt files into separate 'images' and 'labels' subdirectories within the specified source directory.
    
    Parameters:
        source_dir (str): Directory containing unsorted .jpg and .txt files.
    """
    image_dir = os.path.join(source_dir, "images")
    label_dir = os.path.join(source_dir, "labels")

    # Create the destination subdirectories if they do not exist
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Iterate through files in the source directory and move them to the appropriate folder
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        if os.path.isfile(file_path):
            if file_name.endswith(".jpg"):
                shutil.move(file_path, os.path.join(image_dir, file_name))
            elif file_name.endswith(".txt"):
                shutil.move(file_path, os.path.join(label_dir, file_name))
    logger.info("Files moved successfully into 'images' and 'labels' subfolders!")

# ------------------------------------------------------------------------------
# Train/Validation Split Functionality
# ------------------------------------------------------------------------------

def clear_directory(directory):
    """
    Clears all files in the specified directory.
    
    Parameters:
        directory (str): The path to the directory to be cleared.
    """
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            os.remove(file_path)
        logger.info(f"Cleared files in: {directory}")

def copy_file_safe(src, dst):
    """
    Copies a file from the source path to the destination path.
    If the destination file already exists, a warning is logged.
    
    Parameters:
        src (str): The source file path.
        dst (str): The destination file path.
    """
    if os.path.exists(dst):
        logger.warning(f"Skipping {dst}, file already exists.")
        return
    shutil.copy2(src, dst)

def trainval_split(src_path, dest_base, train_ratio):
    """
    Splits a dataset of images (and corresponding label files) into training and validation sets.
    
    Parameters:
        src_path (str): Source directory containing the 'images' and 'labels' subfolders.
        dest_base (str): Base destination directory where the split datasets will be saved.
        train_ratio (float): Proportion of the dataset to be used for training.
    """
    random.seed(SEED)
    
    images_dir = os.path.join(src_path, "images")
    labels_dir = os.path.join(src_path, "labels")

    # Define destination subdirectories for training and validation.
    splits = {
        "train": {
            "images": os.path.join(dest_base, "train", "images"),
            "labels": os.path.join(dest_base, "train", "labels")
        },
        "val": {
            "images": os.path.join(dest_base, "val", "images"),
            "labels": os.path.join(dest_base, "val", "labels")
        }
    }

    # Create the destination subdirectories if they do not exist.
    for split in splits:
        for subfolder in splits[split].values():
            os.makedirs(subfolder, exist_ok=True)
            # Optionally clear the directory to start fresh.
            clear_directory(subfolder)

    # Retrieve all image files (supports .jpg, .jpeg, .png) from the images subfolder.
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(image_files)

    # Split the image list into training and validation sets.
    split_index = int(len(image_files) * train_ratio)
    split_files = {
        "train": image_files[:split_index],
        "val": image_files[split_index:]
    }

    logger.info(f"Training set: {len(split_files['train'])} images")
    logger.info(f"Validation set: {len(split_files['val'])} images")

    # Create a Train.txt or Validation.txt file listing the training images.
    subset_file = os.path.join(dest_base, "Train.txt") if train_ratio == 1.0 or train_ratio >= 0.5 else os.path.join(dest_base, "Validation.txt")
    with open(subset_file, "w") as txt_file:
        for split, files in split_files.items():
            for file_name in files:
                # Copy image
                src_img = os.path.join(images_dir, file_name)
                dst_img = os.path.join(splits[split]["images"], file_name)
                copy_file_safe(src_img, dst_img)

                # Determine corresponding label file name
                label_name = os.path.splitext(file_name)[0] + ".txt"
                src_label = os.path.join(labels_dir, label_name)
                dst_label = os.path.join(splits[split]["labels"], label_name)
                if os.path.exists(src_label):
                    copy_file_safe(src_label, dst_label)
                else:
                    logger.warning(f"Label file {src_label} does not exist.")

                # Write the path of the image relative to the destination base to the subset file.
                txt_file.write(f"data/{split}/images/{file_name}\n")

    logger.info("Dataset split completed successfully!")

# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1. Organise files into 'images' and 'labels' subfolders
    sort_labels_files(unsorted_source_dir)

    # Step 2. Remove any unmatched files between images and labels
    remove_unmatched_files()

    # (Optional) Further cleaning:
    # check_empty_txt_files()
    # remove_unwanted_txt_files([0, 2, 3, 4])  # Modify unwanted class IDs as needed

    # Step 3. Split the dataset into training and validation sets
    trainval_split(SRC_PATH, DEST_BASE, TRAIN_RATIO)