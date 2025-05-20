#!/usr/bin/env python3
import os
import shutil
import random

def generate_new_folder_name(src_path):
    """
    Generates a new folder name based on the source path.
    - If src_path ends with '/_relabeled_split' or '_relabeled_split/',
      it returns 'parent_folder_name_relabeled_split'.
    - Otherwise, it returns the base name of src_path.
    
    Args:
        src_path (str): The source directory path.
        
    Returns:
        str: The generated folder name.
    """
    norm_path = os.path.normpath(src_path) # Normalize path (e.g., remove trailing slashes)
    base_name = os.path.basename(norm_path)
    
    suffix_to_check = "_relabeled_split"
    
    if base_name == suffix_to_check:
        parent_dir_path = os.path.dirname(norm_path)
        parent_dir_name = os.path.basename(parent_dir_path)
        # Handle cases where parent_dir_path might be root or empty
        if not parent_dir_name or parent_dir_name == os.sep: 
            # This case might occur if src_path is like "/_relabeled_split"
            # Depending on desired behavior, you might want to raise an error or return base_name
            return base_name 
        return parent_dir_name + suffix_to_check
    else:
        return base_name

def convert_directory(src_dir, dest_dir, new_folder_name, files_limit):
    """
    Convert directory structure from:
      src_dir/
        ├── train/
        │    ├── images/
        │    └── labels/
        ├── test/
        │    ├── images/
        │    └── labels/
        └── val/
             ├── images/
             └── labels/
    
    To:
      dest_dir/
         ├── images/
         │    ├── train/
         │    │    └── new_folder_name/   <-- contains train image files (.jpg)
         │    ├── test/
         │    │    └── new_folder_name/   <-- contains test image files (.jpg)
         │    └── val/
         │         └── new_folder_name/   <-- contains val image files (.jpg)
         └── labels/
              ├── train/
              │    └── new_folder_name/   <-- contains train label files (.txt)
              ├── test/
              │    └── new_folder_name/   <-- contains test label files (.txt)
              └── val/
                   └── new_folder_name/   <-- contains val label files (.txt)
    
    Additionally, for each split (train, test, val) it randomly selects a fixed number of files to transfer,
    if a limit is specified in the files_limit dictionary.
    
    It prints aggregated progress every 200 files and reports the counts per split.
    
    Args:
       src_dir: Source directory path.
       dest_dir: Destination directory path.
       new_folder_name: The folder name that will be created inside each split.
       files_limit: Dictionary mapping each split ("train", "test", "val") to the number of files to transfer.
                    If a value is 0 (or key is absent), all files are transferred.
    """
    splits = ["train", "val", "test"]
    valid_image_extensions = [".jpg", ".jpeg"]
    
    total_images_copied = 0
    total_labels_copied = 0

    images_per_split = {}
    labels_per_split = {}

    print(f"Source Directory: {src_dir}")
    print(f"Destination Directory: {dest_dir}")
    print(f"Generated New Folder Name for subdirectories: {new_folder_name}\n")

    for split in splits:
        split_images_count = 0
        split_labels_count = 0

        # Process images for the current split
        source_images_path = os.path.join(src_dir, split, "images")
        dest_images_path = os.path.join(dest_dir, "images", split, new_folder_name)
        os.makedirs(dest_images_path, exist_ok=True)
        print(f"Ensured destination images folder exists: {dest_images_path}")
        
        if not os.path.exists(source_images_path):
            print(f"Warning: Source images folder does not exist: {source_images_path}")
            images_per_split[split] = 0
            # No need to set labels_per_split[split] here, it will be handled below
        else:
            # List all image files with valid extensions
            image_files = [f for f in os.listdir(source_images_path)
                           if os.path.splitext(f)[1].lower() in valid_image_extensions]
            
            # Determine file limit for this split (if provided)
            limit = files_limit.get(split, 0)
            if limit > 0 and limit < len(image_files):
                image_files_to_copy = random.sample(image_files, limit)
                print(f"For split '{split}', randomly selecting {limit} of {len(image_files)} images.")
            else:
                image_files_to_copy = image_files
                print(f"For split '{split}', selecting all {len(image_files)} images.")

            # Copy image files and update counters with aggregated prints
            for file_name in image_files_to_copy:
                src_file_path = os.path.join(source_images_path, file_name)
                dest_file_path = os.path.join(dest_images_path, file_name)
                shutil.copy2(src_file_path, dest_file_path)
                total_images_copied += 1
                split_images_count += 1
                if total_images_copied % 200 == 0:
                    print(f"Total images copied so far: {total_images_copied}")
            
            # Process labels for the current split using the same selection as for images
            source_labels_path = os.path.join(src_dir, split, "labels")
            dest_labels_path = os.path.join(dest_dir, "labels", split, new_folder_name)
            os.makedirs(dest_labels_path, exist_ok=True)
            print(f"Ensured destination labels folder exists: {dest_labels_path}")
            
            if not os.path.exists(source_labels_path):
                print(f"Warning: Source labels folder does not exist: {source_labels_path}")
                split_labels_count = 0
            else:
                for image_file_name in image_files_to_copy: # Use the same list of images
                    base_name = os.path.splitext(image_file_name)[0]
                    label_file_name = base_name + ".txt"
                    src_label_file_path = os.path.join(source_labels_path, label_file_name)
                    dest_label_file_path = os.path.join(dest_labels_path, label_file_name)
                    
                    if os.path.exists(src_label_file_path):
                        shutil.copy2(src_label_file_path, dest_label_file_path)
                        total_labels_copied += 1
                        split_labels_count += 1
                        if total_labels_copied % 200 == 0:
                            print(f"Total labels copied so far: {total_labels_copied}")
                    else:
                        print(f"Warning: Label file not found for {image_file_name} (expected: {label_file_name}) in {source_labels_path}")

        images_per_split[split] = split_images_count
        labels_per_split[split] = split_labels_count
        
        # Print counts for the current split
        print(f"For split '{split}': {split_images_count} images and {split_labels_count} labels copied.\n")
        
    print("Directory conversion completed.")
    print(f"Final Total Images Copied: {total_images_copied}")
    print(f"Final Total Labels Copied: {total_labels_copied}")
    print("Split-wise details:")
    for split_name in splits: # Iterate in defined order
        print(f"  {split_name}: {images_per_split.get(split_name, 0)} images, {labels_per_split.get(split_name, 0)} labels copied.")

# ============================
# Configuration Variables
# ============================

# Specify the absolute or relative path to the source directory containing the split folders (train, test, val).
# Example 1: Ends with _relabeled_split
src_directory = "/mnt/nas/TAmob/old_data/final_extracted_frames/07_05_2025 20_29_59 (UTC+03_00)_processed_fr20_10_197_21_24/_relabeled_split/"
# Example 2: Does not end with _relabeled_split
# src_directory = "/mnt/nas/TAmob/old_data/scooter_datasets/scooter_dataset_V6/" 
# Example 3: A simpler path
# src_directory = "/path/to/my_data_folder/"
# Example 4: A simpler path ending with the special suffix
# src_directory = "/path/to/another_project/_relabeled_split/"


# Specify the absolute or relative path to the destination directory.
dest_directory = "/mnt/nas/TAmob/data/"


# Specify number of files to transfer for each split.
# Set the number to a positive integer to transfer that many files randomly;
# set to 0 (or remove the key) to transfer all files.
files_limit = {
    "train": 9999999,  # Transfer all files (or up to this large number)
    "test": 9999999,   # Transfer all files
    "val": 9999999     # Transfer all files
    # Example: "train": 100, "test": 50, "val": 30 
}

# ============================
# Execute the conversion
# ============================
if __name__ == "__main__":
    if not os.path.isdir(src_directory):
        print(f"Error: Source directory '{src_directory}' not found or is not a directory.")
    elif not os.path.isdir(dest_directory):
        # Optionally create dest_directory if it doesn't exist, or raise an error
        try:
            os.makedirs(dest_directory, exist_ok=True)
            print(f"Destination directory '{dest_directory}' was created.")
        except OSError as e:
            print(f"Error: Could not create destination directory '{dest_directory}'. {e}")
    else:
        # Automatically generate the new_folder_name from src_directory
        auto_new_folder_name = generate_new_folder_name(src_directory)
        
        convert_directory(src_directory, dest_directory, auto_new_folder_name, files_limit)
