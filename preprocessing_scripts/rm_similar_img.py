import os
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

"""
Identifies visually similar images using average hash techniques. If 2 images have a similar hash (within the hash threshold), one of the image will be removed.

Paramaters:
    ------------
    folder_paths: The path to the images folder.

    threshold: Maximum Hamming distance between 2 images hashes. The lower the value, the stricter the similarity comparison, lesser images will be considered similar.

    max_workers: Number of worker threads for parallel processing of large datasets. (Recommended not to change)

Functionality: 
    ------------
    1. Logger Set-up:
        - Sets up logger to provide informative messages about script's execution. Logs errors, warnings, and information about processed files.

    2. Compute Average Hash:
        - Use imagehash.average_hash() to compute the average hash.
        - Returns a tuple that contains the image hash and filename.
        - Returns "None" in case of an error.
    
    3. Removing Similar Images:
        - Scans specified 'folder_paths' and collects all image files.
        - For each image, compares its harsh to previously stored hash in tuple.
        - If similar image (by threshold) is found, file is marked for removal.
        - Once all images are processed, similar images are removed and shown through the logs.

Example Usage:
    ------------
    >>> remover_similar_images("/path/to/your/image.png")

Notes:
    ------------
    Only .jpg .png and .jpeg images are processed. If you have images in other formats, modify the file extension filter
"""

# Make changes to parameters here
import os
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

"""
Identifies visually similar images using average hash techniques. If 2 images have a similar hash (within the hash threshold), one of the image will be removed.

Paramaters:
    ------------
    folder_paths: The path to the images folder.

    threshold: Maximum Hamming distance between 2 images hashes. The lower the value, the stricter the similarity comparison, lesser images will be considered similar.

    max_workers: Number of worker threads for parallel processing of large datasets. (Recommended not to change)

Functionality: 
    ------------
    1. Logger Set-up:
        - Sets up logger to provide informative messages about script's execution. Logs errors, warnings, and information about processed files.

    2. Compute Average Hash:
        - Use imagehash.average_hash() to compute the average hash.
        - Returns a tuple that contains the image hash and filename.
        - Returns "None" in case of an error.
    
    3. Removing Similar Images:
        - Scans specified 'folder_paths' and collects all image files.
        - For each image, compares its harsh to previously stored hash in tuple.
        - If similar image (by threshold) is found, file is marked for removal.
        - Once all images are processed, similar images are removed and shown through the logs.

Example Usage:
    ------------
    >>> remover_similar_images("/path/to/your/image.png")

Notes:
    ------------
    Only .jpg .png and .jpeg images are processed. If you have images in other formats, modify the file extension filter
"""

# Make changes to parameters here

folder_paths = [
        "scooter_datasets/scooter_dataset_V3/val/images/" # Replace with folder path 
    ]
threshold = 1 # Replace with integer value 
max_workers = 4 # Replace with integer value 

# -------------------------------------
def setup_logger():
    logger = logging.getLogger("ImageCleaner")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger

logger = setup_logger()

def hash_image(image_path):
    try:
        with Image.open(image_path) as img:
            return imagehash.average_hash(img), os.path.basename(image_path)
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None

def remove_similar_images(folder_path, threshold=threshold, max_workers=max_workers):
    image_hashes = {}
    files_to_remove = []
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg"))] # file extension filter

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(hash_image, filename): filename for filename in image_files}

        for future in as_completed(futures):
            result = future.result()
            if result:
                img_hash, filename = result
                similar_image = None

                for existing_hash, existing_filename in image_hashes.items():
                    if img_hash - existing_hash < threshold:
                        similar_image = filename
                        break

                if similar_image:
                    files_to_remove.append(os.path.join(folder_path, similar_image))
                else:
                    image_hashes[img_hash] = filename

    for filepath in files_to_remove:
        try:
            # os.move(filepath, destination_path)
            # os.remove(filepath)
            logger.info(f"Removed {os.path.basename(filepath)}")
        except Exception as e:
            logger.error(f"Error removing {os.path.basename(filepath)}: {e}")
    
    logger.info(f"Removed {len(files_to_remove)} similar images in {folder_path}")

for folder in folder_paths:
        remove_similar_images(folder)
