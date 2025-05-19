#!/usr/bin/env python3
import os

# Set the target directory where files will be removed.
# Ensure that this directory contains the 'images/' and 'labels/' subdirectories.
# TARGET_DIR = "training_datasets/training_dataset6/"  # <-- change this to your target directory
TARGET_DIR = "training_datasets/training_dataset6/"  # <-- change this to your target directory

# Set the reference directory which holds the files whose corresponding images and labels should be removed.
REFERENCE_DIR = "scooter_datasets/scooter_dataset_V5/"  # <-- change this to your reference directory

def remove_files(target_dir, ref_dir):
    splits = ['train', 'val']
    total_images_removed = 0
    total_labels_removed = 0

    for split in splits:
        # Construct directory paths for images and labels in both reference and target directories.
        ref_images_dir = os.path.join(ref_dir, split, "images")
        ref_labels_dir = os.path.join(ref_dir, split, "labels")
        target_images_dir = os.path.join(target_dir, split, "images")
        target_labels_dir = os.path.join(target_dir, split, "labels")

        # Process images: For each .jpg file in the reference images folder, remove the corresponding file from the target.
        if os.path.exists(ref_images_dir):
            for file in os.listdir(ref_images_dir):
                if file.lower().endswith(".jpg"):
                    target_file_path = os.path.join(target_images_dir, file)
                    if os.path.exists(target_file_path):
                        os.remove(target_file_path)
                        print(f"Removed image: {target_file_path}")
                        total_images_removed += 1
                    else:
                        print(f"Image not found: {target_file_path}")
        else:
            print(f"Reference images directory not found: {ref_images_dir}")

        # Process labels: For each .txt file in the reference labels folder, remove the corresponding file from the target.
        if os.path.exists(ref_labels_dir):
            for file in os.listdir(ref_labels_dir):
                if file.lower().endswith(".txt"):
                    target_file_path = os.path.join(target_labels_dir, file)
                    if os.path.exists(target_file_path):
                        os.remove(target_file_path)
                        print(f"Removed label: {target_file_path}")
                        total_labels_removed += 1
                    else:
                        print(f"Label not found: {target_file_path}")
        else:
            print(f"Reference labels directory not found: {ref_labels_dir}")

    print(f"\nTotal images removed: {total_images_removed}")
    print(f"Total labels removed: {total_labels_removed}")

if __name__ == "__main__":
    remove_files(TARGET_DIR, REFERENCE_DIR)