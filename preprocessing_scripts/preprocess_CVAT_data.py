import os
import shutil
import logging
import random
import argparse # Import argparse

# ------------------------------------------------------------------------------
# Global Parameters and Logging Setup
# ------------------------------------------------------------------------------

# Set to True to actually delete files; False to simulate deletion.
activateDelete = False

# Parameters for the train/validation split
# SRC_PATH will be derived from the --src argument
# DEST_BASE will be derived from the --dest argument
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
            else:
                logger.info(f"Simulated deletion of {file_path}") # Added for clarity when activateDelete is False
            deleted_count += 1
    return deleted_count

def remove_unmatched_files(txt_folder_path, jpg_folder_path):
    """
    Finds and removes unmatched files between the images and labels folders.
    Unmatched files are those .jpg files without a corresponding .txt file and vice versa.
    """
    txt_files = get_filenames(txt_folder_path, ".txt")
    jpg_files = get_filenames(jpg_folder_path, ".jpg")

    unmatched_txt_files = txt_files - jpg_files  # TXT files without JPG counterparts
    unmatched_jpg_files = jpg_files - txt_files  # JPG files without TXT counterparts

    txt_deleted = delete_files(txt_folder_path, unmatched_txt_files, ".txt")
    jpg_deleted = delete_files(jpg_folder_path, unmatched_jpg_files, ".jpg")

    logger.info(f"{jpg_deleted} unmatched .jpg files and {txt_deleted} unmatched .txt files processed!")

def check_empty_txt_files(txt_folder_path):
    """Check if any TXT files are empty and delete them."""
    empty_count = 0
    for file_name in os.listdir(txt_folder_path):
        file_path = os.path.join(txt_folder_path, file_name)
        if file_name.endswith(".txt"):
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                if activateDelete:
                    os.remove(file_path)
                    logger.info(f"Deleted empty file: {file_path}")
                else:
                    logger.info(f"Simulated deletion of empty file: {file_path}")
                empty_count += 1
            else: # If not 0 bytes, check content
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        if activateDelete:
                            os.remove(file_path)
                            logger.info(f"Deleted empty (content-wise) file: {file_path}")
                        else:
                            logger.info(f"Simulated deletion of empty (content-wise) file: {file_path}")
                        empty_count += 1
    logger.info(f"{empty_count} empty .txt files processed!")


def remove_unwanted_txt_files(txt_folder_path, classes):
    """
    Removes .txt files that start with any of the unwanted class IDs.

    Parameters:
        txt_folder_path (str): Path to the labels folder.
        classes (list): List of unwanted class IDs.
    """
    unwanted_files_count = 0
    for file_name in os.listdir(txt_folder_path):
        file_path = os.path.join(txt_folder_path, file_name)
        if file_name.endswith(".txt"):
            with open(file_path, 'r') as f:
                file_deleted = False
                for line in f:
                    content = line.strip()
                    for class_id in classes: # Renamed id to class_id to avoid conflict with built-in
                        if content.startswith(str(class_id)):
                            if activateDelete:
                                os.remove(file_path)
                                logger.info(f"Deleted {file_path} because a line started with {class_id}.")
                            else:
                                logger.info(f"Simulated deletion of {file_path} because a line started with {class_id}.")
                            unwanted_files_count += 1
                            file_deleted = True
                            break
                    if file_deleted:
                        break
    logger.info(f"{unwanted_files_count} unwanted .txt files processed!")

def sort_labels_files(source_dir):
    """
    Sorts .jpg and .txt files into separate 'images' and 'labels' subdirectories within the specified source directory.

    Parameters:
        source_dir (str): Directory containing unsorted .jpg and .txt files.
    Returns:
        tuple: Paths to the images and labels folders.
    """
    image_dir = os.path.join(source_dir, "images")
    label_dir = os.path.join(source_dir, "labels")

    # Create the destination subdirectories if they do not exist
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Iterate through files in the source directory and move them to the appropriate folder
    moved_jpg_count = 0
    moved_txt_count = 0
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        if os.path.isfile(file_path): # Ensure it's a file, not a directory
            if file_name.lower().endswith(".jpg"): # Use lower() for case-insensitivity
                shutil.move(file_path, os.path.join(image_dir, file_name))
                moved_jpg_count +=1
            elif file_name.lower().endswith(".txt"): # Use lower() for case-insensitivity
                shutil.move(file_path, os.path.join(label_dir, file_name))
                moved_txt_count +=1
    logger.info(f"{moved_jpg_count} .jpg files and {moved_txt_count} .txt files moved successfully into 'images' and 'labels' subfolders!")
    return image_dir, label_dir

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
        for file_entry in os.listdir(directory): # Renamed file to file_entry to avoid conflict
            file_path = os.path.join(directory, file_entry)
            if os.path.isfile(file_path): # Ensure it's a file before removing
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
    try:
        shutil.copy2(src, dst)
    except FileNotFoundError:
        logger.error(f"Source file not found for copying: {src}")
    except Exception as e:
        logger.error(f"Error copying file {src} to {dst}: {e}")


def trainval_split(src_path_for_split, dest_base_dir, train_ratio_val):
    """
    Splits a dataset of images (and corresponding label files) into training and validation sets.

    Parameters:
        src_path_for_split (str): Source directory containing the 'images' and 'labels' subfolders (this is the 'src' dir after sorting).
        dest_base_dir (str): Base destination directory where the split datasets will be saved (this is the 'dest' dir).
        train_ratio_val (float): Proportion of the dataset to be used for training.
    """
    random.seed(SEED)

    images_dir = os.path.join(src_path_for_split, "images")
    labels_dir = os.path.join(src_path_for_split, "labels")

    if not os.path.isdir(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        return
    if not os.path.isdir(labels_dir):
        logger.error(f"Labels directory not found: {labels_dir}")
        return

    # Define destination subdirectories for training and validation.
    splits = {
        "train": {
            "images": os.path.join(dest_base_dir, "train", "images"),
            "labels": os.path.join(dest_base_dir, "train", "labels")
        },
        "val": {
            "images": os.path.join(dest_base_dir, "val", "images"),
            "labels": os.path.join(dest_base_dir, "val", "labels")
        }
    }

    # Create the destination subdirectories if they do not exist.
    for split_type in splits: # Renamed split to split_type
        for subfolder in splits[split_type].values():
            os.makedirs(subfolder, exist_ok=True)
            # Optionally clear the directory to start fresh.
            # clear_directory(subfolder) # Consider if this is always desired

    # Retrieve all image files (supports .jpg, .jpeg, .png) from the images subfolder.
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        logger.warning(f"No image files found in {images_dir}. Splitting process aborted.")
        return

    random.shuffle(image_files)

    # Split the image list into training and validation sets.
    split_index = int(len(image_files) * train_ratio_val)
    split_files_map = { # Renamed split_files to split_files_map
        "train": image_files[:split_index],
        "val": image_files[split_index:]
    }

    logger.info(f"Training set: {len(split_files_map['train'])} images")
    logger.info(f"Validation set: {len(split_files_map['val'])} images")

    # Create a Train.txt or Validation.txt file listing the training images.
    # Determine filename based on which set is larger or if it's full training
    if train_ratio_val == 1.0:
        subset_filename = "Train.txt"
    elif train_ratio_val == 0.0:
        subset_filename = "Validation.txt"
    elif train_ratio_val >= 0.5 :
        subset_filename = "Train.txt" # If primarily training, list train images
    else:
        subset_filename = "Validation.txt" # If primarily validation, list val images (less common scenario)

    subset_file_path = os.path.join(dest_base_dir, subset_filename)

    # Create parent directory for subset_file_path if it doesn't exist
    os.makedirs(os.path.dirname(subset_file_path), exist_ok=True)

    with open(subset_file_path, "w") as txt_file_out: # Renamed txt_file to txt_file_out
        for split_type, files_list in split_files_map.items(): # Renamed files to files_list
            for file_name in files_list:
                # Copy image
                src_img = os.path.join(images_dir, file_name)
                dst_img = os.path.join(splits[split_type]["images"], file_name)
                copy_file_safe(src_img, dst_img)

                # Determine corresponding label file name
                label_name = os.path.splitext(file_name)[0] + ".txt"
                src_label = os.path.join(labels_dir, label_name)
                dst_label = os.path.join(splits[split_type]["labels"], label_name)
                if os.path.exists(src_label):
                    copy_file_safe(src_label, dst_label)
                else:
                    logger.warning(f"Label file {src_label} does not exist for image {src_img}.")

                # Write the path of the image relative to the destination base to the subset file.
                # This path is often used for training configurations (e.g. YOLO)
                # The path should be relative to the location of Train.txt/Validation.txt
                # Assuming Train.txt/Validation.txt is in dest_base_dir
                relative_img_path = os.path.join(split_type, "images", file_name)
                txt_file_out.write(f"./{relative_img_path}\n") # Changed to relative path from the file itself

    logger.info(f"Dataset split completed successfully! Subset file created at {subset_file_path}")

# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean, sort, and split image and label datasets.")
    parser.add_argument("--src", type=str, required=True, help="Source directory containing .jpg and .txt files (or 'images' and 'labels' subfolders if already sorted).")
    parser.add_argument("--dest", type=str, required=True, help="Base destination directory where the split datasets (train/val) will be saved.")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO, help=f"Proportion of data used for training (default: {TRAIN_RATIO}).")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Seed for reproducibility (default: {SEED}).")
    parser.add_argument("--activate_delete", action='store_true', help="Set to actually delete files; otherwise, simulates deletion.")


    args = parser.parse_args()

    # Update global parameters from args
    activateDelete = args.activate_delete
    SEED = args.seed
    current_train_ratio = args.train_ratio # Use a different variable name

    # The unsorted_source_dir is now the src argument
    unsorted_source_dir = args.src
    DEST_BASE = args.dest # DEST_BASE is now the dest argument

    logger.info(f"Script started with source: {unsorted_source_dir}, destination: {DEST_BASE}, train_ratio: {current_train_ratio}, seed: {SEED}")
    if activateDelete:
        logger.warning("File deletion is ACTIVATED.")
    else:
        logger.info("File deletion is DEACTIVATED (simulation mode).")


    # Step 1. Organise files into 'images' and 'labels' subfolders
    # Check if images and labels subfolders already exist in src. If so, skip sorting.
    src_images_folder = os.path.join(unsorted_source_dir, "images")
    src_labels_folder = os.path.join(unsorted_source_dir, "labels")

    if os.path.isdir(src_images_folder) and os.path.isdir(src_labels_folder):
        logger.info("'images' and 'labels' subfolders already exist in the source directory. Skipping sorting step.")
        # In this case, SRC_PATH for trainval_split is the unsorted_source_dir itself
        SRC_PATH_for_split = unsorted_source_dir
        # And the jpg_folder and txt_folder are these existing ones
        jpg_folder = src_images_folder
        txt_folder = src_labels_folder
    else:
        logger.info(f"Attempting to sort files from {unsorted_source_dir} into 'images' and 'labels' subfolders.")
        try:
            created_image_dir, created_label_dir = sort_labels_files(unsorted_source_dir)
            jpg_folder = created_image_dir
            txt_folder = created_label_dir
            # After sorting, the SRC_PATH for trainval_split is the unsorted_source_dir,
            # as it now contains the 'images' and 'labels' subfolders.
            SRC_PATH_for_split = unsorted_source_dir
        except Exception as e:
            logger.error(f"Error during file sorting: {e}. Please check the source directory and permissions.")
            exit() # Exit if sorting fails and subfolders weren't pre-existing


    # Ensure jpg_folder and txt_folder are defined before proceeding
    if not 'jpg_folder' in locals() or not 'txt_folder' in locals() or not os.path.isdir(jpg_folder) or not os.path.isdir(txt_folder):
        logger.error(f"Critical error: 'images' ({jpg_folder if 'jpg_folder' in locals() else 'N/A'}) or 'labels' ({txt_folder if 'txt_folder' in locals() else 'N/A'}) directory not found or not created properly in {unsorted_source_dir}. Exiting.")
        exit()


    # Step 2. Remove any unmatched files between images and labels
    logger.info(f"Checking for unmatched files in {jpg_folder} and {txt_folder}.")
    remove_unmatched_files(txt_folder, jpg_folder) # Pass the correct folder paths

    # (Optional) Further cleaning:
    # logger.info(f"Checking for empty .txt files in {txt_folder}.")
    # check_empty_txt_files(txt_folder)
    # logger.info(f"Removing unwanted .txt files from {txt_folder}.")
    # remove_unwanted_txt_files(txt_folder, [0, 2, 3, 4])  # Modify unwanted class IDs as needed

    # Step 3. Split the dataset into training and validation sets
    logger.info(f"Starting dataset split. Source for split: {SRC_PATH_for_split}, Destination base: {DEST_BASE}")
    trainval_split(SRC_PATH_for_split, DEST_BASE, current_train_ratio)

    logger.info("Script execution finished.")