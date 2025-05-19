#!/usr/bin/env python3
import os
import shutil
import random

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

    for split in splits:
        split_images_count = 0
        split_labels_count = 0

        # Process images for the current split
        source_images = os.path.join(src_dir, split, "images")
        dest_images = os.path.join(dest_dir, "images", split, new_folder_name)
        os.makedirs(dest_images, exist_ok=True)
        print(f"Ensured destination folder exists: {dest_images}")
        
        if not os.path.exists(source_images):
            print(f"Warning: Source images folder does not exist: {source_images}")
            images_per_split[split] = 0
            labels_per_split[split] = 0
            continue
        
        # List all image files with valid extensions
        image_files = [f for f in os.listdir(source_images)
                       if os.path.splitext(f)[1].lower() in valid_image_extensions]
        
        # Determine file limit for this split (if provided)
        limit = files_limit.get(split, 0)
        if limit and limit < len(image_files):
            image_files = random.sample(image_files, limit)
        
        # Copy image files and update counters with aggregated prints
        for file in image_files:
            src_file = os.path.join(source_images, file)
            dest_file = os.path.join(dest_images, file)
            shutil.copy2(src_file, dest_file)
            total_images_copied += 1
            split_images_count += 1
            if total_images_copied % 200 == 0:
                print(f"Total images copied so far: {total_images_copied}")
        
        # Process labels for the current split using the same selection as for images
        source_labels = os.path.join(src_dir, split, "labels")
        dest_labels = os.path.join(dest_dir, "labels", split, new_folder_name)
        os.makedirs(dest_labels, exist_ok=True)
        print(f"Ensured destination folder exists: {dest_labels}")
        
        if not os.path.exists(source_labels):
            print(f"Warning: Source labels folder does not exist: {source_labels}")
            split_labels_count = 0
        else:
            for image_file in image_files:
                base = os.path.splitext(image_file)[0]
                label_file = base + ".txt"
                src_label_file = os.path.join(source_labels, label_file)
                dest_label_file = os.path.join(dest_labels, label_file)
                if os.path.exists(src_label_file):
                    shutil.copy2(src_label_file, dest_label_file)
                    total_labels_copied += 1
                    split_labels_count += 1
                    if total_labels_copied % 200 == 0:
                        print(f"Total labels copied so far: {total_labels_copied}")
                else:
                    print(f"Warning: Label file not found for {image_file} in {source_labels}")

        images_per_split[split] = split_images_count
        labels_per_split[split] = split_labels_count
        
        # Print counts for the current split
        print(f"For split '{split}': {split_images_count} images and {split_labels_count} labels copied.\n")
        
    print("Directory conversion completed.")
    print(f"Final Total Images Copied: {total_images_copied}")
    print(f"Final Total Labels Copied: {total_labels_copied}")
    print("Split-wise details:")
    for split in splits:
        print(f"  {split}: {images_per_split.get(split, 0)} images, {labels_per_split.get(split, 0)} labels copied.")

# ============================
# Configuration Variables
# ============================

# Specify the absolute or relative path to the source directory.
# src_directory = "/mnt/nas/TAmob/old_data/scooter_datasets/scooter_dataset_V6/"
src_directory = "/mnt/nas/TAmob/old_data/final_extracted_frames/05_05_2025 20_30_00 (UTC+03_00)_processed_fr20_10.197.21.23/_relabeled_split/"

# Specify the absolute or relative path to the destination directory.
dest_directory = "/mnt/nas/TAmob/data/"

# Specify the new folder name to be created inside each split folder at the destination.
new_folder_name = "05_05_2025 20_30_00 (UTC+03_00)_processed_fr20_10.197.21.23" \
"" + "_relabeled_split"

# Specify number of files to transfer for each split.
# Set the number to a positive integer to transfer that many files randomly;
# set to 0 (or remove the key) to transfer all files.
files_limit = {
    "train": 9999999,  # Transfer 100 files from the train split.
    "test": 9999999,    # Transfer 50 files from the test split.
    "val": 9999999      # Transfer 30 files from the val split.
}

# ============================
# Execute the conversion
# ============================
if __name__ == "__main__":
    convert_directory(src_directory, dest_directory, new_folder_name, files_limit)