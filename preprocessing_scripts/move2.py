import os
import random
import shutil

# Set seed for reproducibility
random.seed(42)  # Change 42 to any other integer if needed

"""PLEASE CHANGE THE VARIABLE BELOW TO ACCORDING TO YOUR NEEDS"""

# Set number of files to move / copy
number_files_to_move = 1130
operation = "move"  # Either 'move' or 'copy' from src

# Define source and destination directories
base_src_dir = "raw_datasets/vehicle_dataset_copy/"
base_dest_dir = "raw_datasets/vehicle_dataset_copy/"

# Choose whether to replace "R" or append "A" to the destination folder
choice = "A" 

# List of subsets to process ["train", "val", "test"]
subsets = ["1"]

"""END OF VARIABLES TO CHANGE"""


for subset in subsets:
    # Define source and destination directories for each subset
    src_img_dir = os.path.join(base_src_dir, "train", "images")
    src_labels_dir = os.path.join(base_src_dir, "train", "labels")
    destination_image_dir = os.path.join(base_dest_dir, "val", "images")
    destination_label_dir = os.path.join(base_dest_dir, "val", "labels")
    
    # Remove destination directories if user selected "R"
    if choice == "R":
        if os.path.exists(destination_image_dir):
            shutil.rmtree(destination_image_dir)
        if os.path.exists(destination_label_dir):
            shutil.rmtree(destination_label_dir)
        print(f"Destination directories for 'val' cleared.")
    elif choice == "A":
        print(f"Appending to destination directories for 'val'.")
    else:
        print("Invalid choice provided, defaulting to append.")

    # Ensure destination directories exist
    os.makedirs(destination_image_dir, exist_ok=True)
    os.makedirs(destination_label_dir, exist_ok=True)

    # Get all .jpg files from the source images folder
    image_files = [f for f in os.listdir(src_img_dir) if f.endswith('.jpg')]

    # Randomly select files (up to the number specified)
    random_files = random.sample(image_files, min(len(image_files), number_files_to_move))

    # Process each selected file along with its corresponding label file
    for file in random_files:
        image_path = os.path.join(src_img_dir, file)
        base_name = os.path.splitext(file)[0]
        label_file = base_name + ".txt"
        label_path = os.path.join(src_labels_dir, label_file)

        if operation == 'move':
            shutil.move(image_path, destination_image_dir)
            if os.path.exists(label_path):
                shutil.move(label_path, destination_label_dir)
        elif operation == 'copy':
            shutil.copy(image_path, destination_image_dir)
            if os.path.exists(label_path):
                shutil.copy(label_path, destination_label_dir)

    print(f"Processed {len(random_files)} .jpg files and their corresponding .txt files for 'train' subset.")