import os
import logging

"""
Identifies and deletes PNG files in a specified folder that do not have corresponding text files in another specified folder
AND vice-versa (if there are no PNG files for text files).

Parameters:
    -----------
    txt_folder : Path to your text folder directory.
    jpg_folder : Path to your image folder directory.
    log_level  : Logging level to display script execution messages.

Functionality:
    --------------
    1. Retrieving Filenames:
       - Extracts base filenames for all .txt and .jpg files.
       
    2. Identifying Unmatched Files:
       - Computes the difference between PNG files and TXT filenames.
       - Identifies PNGs without corresponding text files and vice versa.

    3. Deleting Unmatched Files:
       - Iterates over unmatched PNG and TXT files and deletes them.
       - Logs the deletion process.

Example Usage:
    ------------
    >>> python rm_excess.py
"""

# Parameters (Modify as needed)
jpg_folder = "box_datasets/box_dataset_rm_veh_copy/train/images/"  # Replace with the path to image files
txt_folder = "box_datasets/box_dataset_rm_veh_copy/train/labels/"  # Replace with the path to text files
log_level = logging.INFO  # Change to DEBUG for more detailed logs
activateDelete = True # Set to False to check for number of files to be deleted without deleting any files. True initiates the deletion process.

# Logging setup
def setup_logger():
    logger = logging.getLogger("FileCleaner")
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger

logger = setup_logger()

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
    """Find and remove unmatched files between TXT and JPG folders."""
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
                    f.close()
                    if activateDelete:
                        os.remove(file_path) 
                        logger.info(f"Deleted empty file: {file_path}")
                    empty_count += 1
    logger.info(f"{empty_count} empty .txt files deleted!")

"""
Classes: An array of unwanted classes IDs
"""
def remove_unwanted_txt_files(classes):
    unwanted_files_count = 0
    for file_name in os.listdir(txt_folder):
        file_path = os.path.join(txt_folder, file_name)
        if file_name.endswith(".txt"):
            with open(file_path, 'r') as f:
                file_is_closed = False
                for line in f:
                    content = line.strip()
                    for id in classes:
                        if content.startswith(str(id)):
                            f.close()  # Close the file before deleting
                            if activateDelete:
                                os.remove(file_path)
                                logger.info(f"Deleted {file_path} because a line started with {id}.")
                            unwanted_files_count += 1
                            file_is_closed = True
                            break
                    if file_is_closed:
                            break
    logger.info(f"{unwanted_files_count} unwantned .txt files with invalid class IDs deleted!")


if __name__ == "__main__":
    remove_unwanted_txt_files([0, 1, 2, 3, 4]) # keep 5, 6 (box is inaccurate and scooter)
    check_empty_txt_files()
    remove_unmatched_files()
