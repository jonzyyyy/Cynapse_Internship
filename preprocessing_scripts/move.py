import os
import random
import shutil

# Set seed for reproducibility
random.seed(42)  # Change 42 to any other integer if needed

"""PLEASE CHANGE THE VARIABLE BELOW TO ACCORDING TO YOUR NEEDS"""

# Set number of files to move / copy
number_files_to_move = 9999999
operation = "copy"  # Either 'move' or 'copy' from src

# Define source and destination directories
# base_src_dir = "raw_datasets/vehicle_dataset/"
# base_dest_dir = "training_datasets/training_dataset3/"

base_src_dir = "raw_datasets/vehicle_dataset/"
base_dest_dir = "../TAmob_box_detector/data/vehicle_dataset/"

# base_src_dir = "scooter_datasets/scooter_dataset_V5/"
# base_dest_dir = "training_datasets/training_dataset6/"

# base_src_dir = "box_datasets/box_dataset_final/"
# base_dest_dir = "box_datasets/box_dataset_V1/"

# base_src_dir = "scooter_datasets/scooter_rotated_datasets/rotated_scooter_dataset_split_relabelled/"
# base_dest_dir = "scooter_datasets/scooter_dataset_V5/"

# Choose whether to replace "R" or append "A" to the destination folder
choice = "A" 

add_postfix = None

# List of subsets to process ["train", "val", "test"]
subsets = ["train", "val"]

"""END OF VARIABLES TO CHANGE"""

for subset in subsets:
    # Define source and destination directories for each subset
    src_img_dir = os.path.join(base_src_dir, subset, "images")
    src_labels_dir = os.path.join(base_src_dir, subset, "labels")
    destination_image_dir = os.path.join(base_dest_dir, subset, "images")
    destination_label_dir = os.path.join(base_dest_dir, subset, "labels")
    
    # Remove destination directories if user selected "R"
    if choice == "R":
        if os.path.exists(destination_image_dir):
            shutil.rmtree(destination_image_dir)
        if os.path.exists(destination_label_dir):
            shutil.rmtree(destination_label_dir)
        print(f"Destination directories for '{subset}' cleared.")
    elif choice == "A":
        print(f"Appending to destination directories for '{subset}'.")
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

        if add_postfix:
            new_image_filename = base_name + add_postfix + ".jpg"  # Append the specified postfix
            new_label_filename = base_name + add_postfix + ".txt"  # Append the specified postfix to label file name
        else:
            new_image_filename = file
            new_label_filename = label_file
        dest_image_path = os.path.join(destination_image_dir, new_image_filename)
        dest_label_path = os.path.join(destination_label_dir, new_label_filename)

        if operation == 'move':
            shutil.move(image_path, dest_image_path)
            if os.path.exists(label_path):
                shutil.move(label_path, dest_label_path)
        elif operation == 'copy':
            shutil.copy(image_path, dest_image_path)
            if os.path.exists(label_path):
                shutil.copy(label_path, dest_label_path)

    print(f"Processed {len(random_files)} .jpg files and their corresponding .txt files for '{subset}' subset.")