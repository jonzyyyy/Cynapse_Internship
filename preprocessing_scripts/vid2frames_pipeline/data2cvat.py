import os
import shutil
import argparse # Import the argparse module

"""
This script move and rename image and labels files from source directory and prepare them in the correct format to be uploaded up to CVAT.

Parameters:
    -----------
    src_dir : Path to you source directory for image and labels. (Passed as a command-line argument)
        
    classes : List of class names representing each object categories. (Should correspond to YOLO's COCO.yaml file input)

    subset : Subset indicating if the files belong to the training set or validation set. Allowed values are 'Train' or 'Validation'.
        
Functionality:
    --------------
    1. Directory Set up:
       - The source directory expects to have 2 subfolders: images - containing image files, labels - corresponding files in .txt format.
       - The destination directory is generated based on the source directory name with and added "_cvat". 
       - Destination directory contains an additional subfolder depending on the subset (Train or Validation), e.g. obj_Train_data or obj_Validation_data 
    
    2. File Renaming and Moving:
        - Checks for subfolders in the images folder
        - If subfolders exists, the images and labels are copied and renamed using the format:
            {subfolders_name}_frame{i}.jpg and {subfolders_name}_frame{i}.txt 
        - If no subfolder exists, images and labels are copied without renaming.
    
    3. Creating .txt Files:
        - A Train.txt or Validation.txt file is created inside the desitnation folder, depending on the subset.
        - The file contains all image and text files, formatted to match the CVAT expected structure 
    
    4. CVAT Config Files:
        - obj.names: A file containing all class names, one per line.
        - obj.data: A file that specifies the number of classes, the path to the training or validation .txt file, the path to obj.names, and a backup directory.

    5. Creating a ZIP Archive:
        - After organizing all files and generating configuration files, the folder is zipped for uploading to CVAT 

Example Usage (from command line):
    --------------
    python your_script_name.py /path/to/your/source/directory --classes car motorcycle bus --subset Train

Notes:
    -------------
    - Source directory has to be in this format:

        src_dir/
        ├── images/
        │   ├── subfolder1/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        ├── labels/
        │   ├── subfolder1/
        │   │   ├── img1.txt
        │   │   ├── img2.txt
    
    - The output zip'd folder structure will be in this format suitable for CVAT uploading:

        dest_dir/
        ├── obj_Train_data/
        │   ├── image1.jpg
        │   ├── image1.txt
        ├── Train.txt
        ├── obj.names
        ├── obj.data
        ├── backup/
        └── dest_dir.zip
    
    - Input Assumptions:
        - The source directory contains images and labels subdirectories.
        - Each image in the images subdirectory has a corresponding annotation file in the labels subdirectory.

    -  Naming Convention:
        - The files are renamed if there are subfolders within the images folder, otherwise, they retain their original names.
"""

def move_and_rename_files(src_dir, classes, subset): # Removed dest_dir from parameters as it's derived
    images_subfolder = os.path.join(src_dir, 'images')
    labels_subfolder = os.path.join(src_dir, 'labels')

    # Extract the base name of the src_dir and append '_cvat'
    src_dir_name = os.path.basename(src_dir.rstrip('/'))
    dest_dir_name = f"{src_dir_name}_cvat"
    # Construct dest_dir relative to the parent of src_dir or current working directory if src_dir is a top-level dir
    parent_of_src_dir = os.path.dirname(src_dir.rstrip('/'))
    if not parent_of_src_dir: # Handles cases where src_dir might be like "my_data" (relative path with no parent specified)
        parent_of_src_dir = "." 
    dest_dir = os.path.join(parent_of_src_dir, dest_dir_name)


    # Create destination subfolder if it doesn't exist
    dest_subfolder = os.path.join(dest_dir, f"obj_{subset}_data")
    os.makedirs(dest_subfolder, exist_ok=True)
    
    # Create backup directory inside dest_dir
    backup_dir = os.path.join(dest_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)


    # Create Train.txt or Validation.txt file
    txt_file_path = os.path.join(dest_dir, f"{subset}.txt")
    with open(txt_file_path, "w") as txt_file:

        # Check if there are subfolders in the images_subfolder
        if not os.path.exists(images_subfolder):
            print(f"Error: 'images' subfolder not found in {src_dir}")
            return
        if not os.path.exists(labels_subfolder):
            print(f"Error: 'labels' subfolder not found in {src_dir}")
            return

        subfolders = [f for f in os.listdir(images_subfolder) if os.path.isdir(os.path.join(images_subfolder, f))]

        if subfolders:
            for folder_name in subfolders:
                src_images_folder = os.path.join(images_subfolder, folder_name)
                src_labels_folder = os.path.join(labels_subfolder, folder_name)

                if not os.path.exists(src_labels_folder):
                    print(f"Warning: Corresponding labels subfolder '{folder_name}' not found in {labels_subfolder}. Skipping.")
                    continue

                images = sorted([img for img in os.listdir(src_images_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                for i, image_name in enumerate(images, start=1):
                    base_name, _ = os.path.splitext(image_name)
                    label_name = f"{base_name}.txt" # Assuming label has same base name as image

                    src_image = os.path.join(src_images_folder, image_name)
                    src_label = os.path.join(src_labels_folder, label_name)

                    if not os.path.exists(src_label):
                        print(f"Warning: Label file {src_label} not found for image {src_image}. Skipping this image-label pair.")
                        continue

                    dest_image_name = f"{os.path.basename(folder_name)}_frame{i}{os.path.splitext(image_name)[1]}" # Preserve original extension
                    dest_label_name = f"{os.path.basename(folder_name)}_frame{i}.txt"
                    
                    dest_image = os.path.join(dest_subfolder, dest_image_name)
                    dest_label = os.path.join(dest_subfolder, dest_label_name)

                    # Copy and rename the image and label files
                    shutil.copy2(src_image, dest_image)
                    shutil.copy2(src_label, dest_label)

                    # Write the path of the image to Train.txt or Validation.txt
                    txt_file.write(f"data/obj_{subset}_data/{dest_image_name}\n")
        else:
            images = sorted([img for img in os.listdir(images_subfolder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            for image_name in images:
                base_name, _ = os.path.splitext(image_name)
                label_name = f"{base_name}.txt"

                src_image = os.path.join(images_subfolder, image_name)
                src_label = os.path.join(labels_subfolder, label_name)

                if not os.path.exists(src_label):
                    print(f"Warning: Label file {src_label} not found for image {src_image}. Skipping this image-label pair.")
                    continue

                dest_image = os.path.join(dest_subfolder, image_name)
                dest_label = os.path.join(dest_subfolder, label_name)

                # Copy the image and label files without renaming
                shutil.copy2(src_image, dest_image)
                shutil.copy2(src_label, dest_label)

                # Write the path of the image to Train.txt or Validation.txt
                txt_file.write(f"data/obj_{subset}_data/{image_name}\n")

    print(f"Successfully created {txt_file_path}")

    # Create obj.names file
    obj_names_path = os.path.join(dest_dir, "obj.names")
    with open(obj_names_path, "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"Successfully created {obj_names_path}")

    # Create obj.data file
    obj_data_path = os.path.join(dest_dir, "obj.data")
    with open(obj_data_path, "w") as f:
        f.write(f"classes = {len(classes)}\n")
        f.write(f"{subset.lower()} = data/{subset}.txt\n") # CVAT often expects 'train' or 'valid' in lowercase
        f.write(f"names = data/obj.names\n")
        f.write(f"backup = backup/\n") # Ensure backup directory is referenced correctly
    print(f"Successfully created {obj_data_path}")

    # Create a zip archive of the final product
    try:
        archive_name = shutil.make_archive(dest_dir, 'zip', root_dir=parent_of_src_dir, base_dir=dest_dir_name)
        print(f"Successfully created ZIP archive: {archive_name}")
        # Optional: Remove the original directory after zipping
        # shutil.rmtree(dest_dir)
        # print(f"Successfully removed original directory: {dest_dir}")
    except Exception as e:
        print(f"Error creating ZIP archive: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare image and label files for CVAT.")
    parser.add_argument("src_dir", type=str, help="Path to your source directory for images and labels.")
    parser.add_argument("--classes", nargs='+', required=True, help="List of class names (e.g., car motorcycle bus).")
    parser.add_argument("--subset", type=str, choices=['Train', 'Validation'], required=True, help="Subset type: 'Train' or 'Validation'.")
    
    args = parser.parse_args()


    # Check if the source directory exists
    if not os.path.exists(args.src_dir):
        print(f"Error: Source directory {args.src_dir} does not exist.")
        exit(1)

    
    move_and_rename_files(args.src_dir, args.classes, args.subset)

    # Hardcoded values:
    # src_dir = "/mnt/nas/TAmob/old_data/final_extracted_frames/07_05_2025 20_29_59 (UTC+03_00)_processed_fr20_10_197_21_23/_labeled/train"
    # classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 'scooter]
    # subset = 'Train' 
    # move_and_rename_files(src_dir, classes, subset)