import os
import shutil
import argparse

def split_images_and_labels_YOLO_format(source_dir):
    """
    IN-PLACE FUNCTION
    Organizes files in the source directory by moving .jpg files to an 'images'
    subdirectory and .txt files to a 'labels' subdirectory.

    Args:
        source_dir (str): The path to the source directory.
    """
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    # Define the destination directories
    image_dir = os.path.join(source_dir, "images")
    label_dir = os.path.join(source_dir, "labels")

    # Create directories if they do not exist
    try:
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        print(f"Created or confirmed existence of directories: '{image_dir}' and '{label_dir}'")
    except OSError as e:
        print(f"Error creating directories: {e}")
        return

    files_moved_count = 0
    # Iterate through files in the source directory
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)

        # Ensure it's a file (not a directory) and not one of the target directories
        if os.path.isfile(file_path):
            try:
                if file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg"):
                    shutil.move(file_path, os.path.join(image_dir, file_name))
                    print(f"Moved image: {file_name} to {image_dir}")
                    files_moved_count += 1
                elif file_name.lower().endswith(".txt"):
                    # Avoid moving files that might be part of other labeling systems if needed
                    # For now, we assume all .txt files are labels.
                    shutil.move(file_path, os.path.join(label_dir, file_name))
                    print(f"Moved label: {file_name} to {label_dir}")
                    files_moved_count += 1
            except shutil.Error as e:
                print(f"Error moving file {file_name}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred with file {file_name}: {e}")


    if files_moved_count > 0:
        print(f"\nSuccessfully moved {files_moved_count} files!")
    else:
        print("\nNo files were found to move in the specified source directory (that are not already in 'images' or 'labels' subdirectories).")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Organize image and label files from a source directory into 'images' and 'labels' subdirectories.")
    parser.add_argument("source_dir", type=str, help="The path to the source directory containing .jpg and .txt files.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the provided source directory
    split_images_and_labels_YOLO_format(args.source_dir)