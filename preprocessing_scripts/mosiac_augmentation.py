import cv2  # OpenCV is used for image reading, resizing, and writing
import numpy as np
import random
import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)

# Please READ before using the script: https://cynapseai.atlassian.net/wiki/x/eIJhK

"""
Key Configurations Available via Command-Line Arguments:

Required positional arguments:
  src_dir           Base source directory containing YOLO-formatted splits (e.g., train/images, train/labels).
  dest_dir          Base destination directory for augmented mosaic data.
  label_prefix      Prefix for output file names; results will be named {label_prefix}_mosaic_{index}.jpg/.txt.
  grid_size         Grid dimension for mosaic (e.g., 2 to create a 2×2 mosaic).

Optional arguments:
  --splits SPLITS   One or more dataset splits to process (default: ["train"]).
  --random_crop     Enable pre-letterbox random cropping (default: disabled).
  --jitter JITTER   Maximum fraction of width/height to crop randomly (default: 0.3).
  --min_length MIN_LENGTH
                    Minimum bounding-box side length in pixels; images with smaller objects are skipped (default: -1, no restriction).
  --enable_letterbox
                    Enable letterboxing to preserve aspect ratio with padding (default: enabled).
  --no_letterbox    Disable letterboxing (inverse of --enable_letterbox).

Examples:
  # 2×2 mosaic, train split only, with random crop and jitter up to 0.2
  python mosaic_augmentation.py data/raw data/augmented VEHICLE 2 --random_crop --jitter 0.2

  # 3×3 mosaic, train and val splits, letterboxing off
  python mosaic_augmentation.py data/raw data/augmented BOX 3 --splits train val --no_letterbox
"""

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def letterbox_image(image, target_size):
    """
    Resize image to fit in target_size (width, height) with unchanged aspect ratio using padding.
    Returns the padded image and the scaling factors used.
    """
    src_h, src_w = image.shape[:2]
    tgt_w, tgt_h = target_size
    logging.debug("src_h: %d, src_w: %d", src_h, src_w)
    scale = min(tgt_w / src_w, tgt_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = tgt_w - new_w
    pad_h = tgt_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, scale, left, top

def adjust_bbox_for_letterbox(label, orig_size, scale, pad_x, pad_y):
    """
    Adjust a YOLO-format bounding box for a letterboxed image.

    Parameters:
      label: list in format [class, x_center, y_center, w, h] normalized relative to the original image.
      orig_size: tuple (width, height) of the original image.
      scale: scaling factor used in letterboxing.
      pad_x: horizontal padding added to the left.
      pad_y: vertical padding added to the top.

    Returns:
      Adjusted bounding box in pixel coordinates [class, x1, y1, x2, y2].
    """
    cls, x_center, y_center, w, h = label
    orig_w, orig_h = orig_size
    abs_x_center = x_center * orig_w
    abs_y_center = y_center * orig_h
    abs_w = w * orig_w
    abs_h = h * orig_h
    x1 = abs_x_center - abs_w / 2
    y1 = abs_y_center - abs_h / 2
    x2 = abs_x_center + abs_w / 2
    y2 = abs_y_center + abs_h / 2
    # Apply scaling and add padding
    x1 = x1 * scale + pad_x
    y1 = y1 * scale + pad_y
    x2 = x2 * scale + pad_x
    y2 = y2 * scale + pad_y
    return [cls, x1, y1, x2, y2]

def random_crop_pre_letterbox(image, labels, jitter):
    """
    Perform a random jitter crop on the original image and adjust YOLO labels.
    - image: BGR NumPy array
    - labels: list of [cls, x_center, y_center, w, h] in normalized coords
    - jitter: fraction of width/height to allow as max crop on each side
    Returns cropped image and updated labels (normalized to the crop).
    """
    orig_h, orig_w = image.shape[:2]
    dw = int(orig_w * jitter)
    dh = int(orig_h * jitter)
    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    # Compute crop coordinates
    x1 = max(0, 0 + pleft)
    y1 = max(0, 0 + ptop)
    x2 = min(orig_w, orig_w + pright)
    y2 = min(orig_h, orig_h + pbot)

    # Perform crop
    cropped = image[y1:y2, x1:x2]
    crop_w, crop_h = x2 - x1, y2 - y1

    # Adjust labels
    new_labels = []
    for cls, xc, yc, w, h in labels:
        # Convert to absolute coords
        abs_xc = xc * orig_w
        abs_yc = yc * orig_h
        abs_w  = w  * orig_w
        abs_h  = h  * orig_h
        x_min = abs_xc - abs_w/2 - x1
        y_min = abs_yc - abs_h/2 - y1
        x_max = abs_xc + abs_w/2 - x1
        y_max = abs_yc + abs_h/2 - y1

        # Intersect with crop
        x_min = max(0, min(x_min, crop_w))
        y_min = max(0, min(y_min, crop_h))
        x_max = max(0, min(x_max, crop_w))
        y_max = max(0, min(y_max, crop_h))
        if x_max <= x_min or y_max <= y_min:
            continue

        # Re-normalize to cropped size
        new_w = x_max - x_min
        new_h = y_max - y_min
        new_xc = x_min + new_w/2
        new_yc = y_min + new_h/2
        new_labels.append([
            cls,
            new_xc / crop_w,
            new_yc / crop_h,
            new_w  / crop_w,
            new_h  / crop_h
        ])

    return cropped, new_labels

def mosaic_augmentation_letterbox(image_paths, label_paths, out_size=1280, grid_size=2, target_size=(640,640)):
    """
    Combines grid_size x grid_size images into one mosaic using letterboxed images
    that are resized with black padding without scaling up.
    Adjusts bounding boxes corresponding to the letterbox transformation.
    
    Parameters:
      image_paths: list of image file paths.
      label_paths: list of corresponding label file paths.
      out_size: size of the output mosaic image.
      grid_size: grid dimension for mosaic.
      target_size: target size for letterboxing (e.g., (640,640)).
    
    Returns:
      mosaic_img: the combined mosaic image.
      mosaic_labels: list of adjusted YOLO-format bounding boxes (normalized relative to the mosaic).
    """
    cell_size = out_size // grid_size
    mosaic_size = grid_size * cell_size
    mosaic_img = np.full((mosaic_size, mosaic_size, 3), 114, dtype=np.uint8)
    mosaic_labels = []
    
    for idx in range(len(image_paths)):
        row = idx // grid_size
        col = idx % grid_size
        mosaic_x1 = col * cell_size
        mosaic_y1 = row * cell_size
        mosaic_x2 = mosaic_x1 + cell_size
        mosaic_y2 = mosaic_y1 + cell_size
        
        img = cv2.imread(image_paths[idx])
        if img is None:
            raise ValueError(f"Image at {image_paths[idx]} could not be loaded.")
        orig_h, orig_w = img.shape[:2]
        labels = load_labels(label_path=label_paths[idx])

        if args.random_crop:
            img, labels = random_crop_pre_letterbox(img, labels, args.jitter)
            # Update dimensions after cropping so bounding-box adjustment uses correct size
            orig_h, orig_w = img.shape[:2]

        if args.enable_letterbox:
            img_processed, scale, pad_x, pad_y = letterbox_image(img, target_size)
        else:
            img_processed = cv2.resize(img, target_size)
            scale, pad_x, pad_y = 1.0, 0, 0

        patch = cv2.resize(img_processed, (cell_size, cell_size))
        factor = cell_size / target_size[0]
        for label in labels:
            bbox = adjust_bbox_for_letterbox(label, (orig_w, orig_h), scale, pad_x, pad_y)
            scaled_bbox = [bbox[0], bbox[1] * factor, bbox[2] * factor, bbox[3] * factor, bbox[4] * factor]
            new_bbox = [
                scaled_bbox[0],
                scaled_bbox[1] + mosaic_x1,
                scaled_bbox[2] + mosaic_y1,
                scaled_bbox[3] + mosaic_x1,
                scaled_bbox[4] + mosaic_y1
            ]
            x1c, y1c, x2c, y2c = clip_bbox(new_bbox[1:], mosaic_size)
            yolo_bbox = convert_bbox_to_yolo([new_bbox[0], x1c, y1c, x2c, y2c], mosaic_size)
            mosaic_labels.append(yolo_bbox)
        
        mosaic_img[mosaic_y1:mosaic_y2, mosaic_x1:mosaic_x2] = patch
    
    return mosaic_img, mosaic_labels

def load_labels(label_path):
    """
    Loads YOLO-format labels from a text file.
    Each line should be:
      <class> <x_center> <y_center> <width> <height>
    with normalised coordinates (0 to 1).
    """
    labels = []
    if not os.path.exists(label_path):
        return labels  # Return empty list if no label file is found.
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            labels.append([int(cls), float(x), float(y), float(w), float(h)])
    return labels

def clip_bbox(bbox, mosaic_size):
    """
    Clips a bounding box (x1, y1, x2, y2) to the mosaic boundaries.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, mosaic_size))
    y1 = max(0, min(y1, mosaic_size))
    x2 = max(0, min(x2, mosaic_size))
    y2 = max(0, min(y2, mosaic_size))
    return [x1, y1, x2, y2]

def convert_bbox_to_xyxy(label, img_size):
    """
    Convert a YOLO-format bbox from normalized format to pixel coordinates.
    Returns [class, x1, y1, x2, y2].
    
    This conversion is essential for adjusting bounding boxes during augmentation.
    """
    cls, x_center, y_center, w, h = label
    x_center *= img_size
    y_center *= img_size
    w *= img_size
    h *= img_size
    x1 = x_center - w/2
    y1 = y_center - h/2
    x2 = x_center + w/2
    y2 = y_center + h/2
    return [cls, x1, y1, x2, y2]

def convert_bbox_to_yolo(bbox, mosaic_size):
    """
    Convert a bbox from pixel coordinates (x1, y1, x2, y2) to YOLO normalised format.
    """
    cls, x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w/2
    y_center = y1 + h/2
    return [cls, x_center/mosaic_size, y_center/mosaic_size, w/mosaic_size, h/mosaic_size]

def resize_labels(mosaic_labels, old_size, new_size):
    """
    Optionally re-compute normalized bounding boxes for a resized mosaic image.
    Since the image is uniformly scaled, the normalized coordinates remain the same.
    This function demonstrates the resizing process for clarity.
    """
    resized_labels = []
    scale = new_size / old_size
    for label in mosaic_labels:
        cls, x, y, w, h = label
        # Convert normalized coordinates to absolute values for the old mosaic size
        abs_x = x * old_size
        abs_y = y * old_size
        abs_w = w * old_size
        abs_h = h * old_size
        # Scale the absolute coordinates to the new mosaic size
        abs_x *= scale
        abs_y *= scale
        abs_w *= scale
        abs_h *= scale
        # Convert back to normalized coordinates for the new size
        new_label = [cls, abs_x/new_size, abs_y/new_size, abs_w/new_size, abs_h/new_size]
        resized_labels.append(new_label)
    return resized_labels

def is_valid_for_mosaic(lbl_path, min_length, grid_size, img_size=64):
    """
    Checks if the image has the smallest bounding box dimensions (both width and height) in the original image
    large enough so that when the mosaic is resized, the effective bounding box dimensions
    do not drop below the user-defined min_length.
    If no labels are found, returns empty text file, with warning.
    """
    labels = load_labels(lbl_path)
    if not labels:
        logging.warning(f"No labels found for {lbl_path}. Image is treated as a background image.")
    min_box_width = float('inf')
    min_box_height = float('inf')
    for label in labels:
        # Convert the label to absolute pixel coordinates
        _, x1, y1, x2, y2 = convert_bbox_to_xyxy(label, img_size)
        box_width = x2 - x1
        box_height = y2 - y1
        min_box_width = min(min_box_width, box_width)
        min_box_height = min(min_box_height, box_height)
    # Both dimensions must be at or above the threshold: grid_size * min_length
    return (min_box_width >= grid_size * min_length) and (min_box_height >= grid_size * min_length)

def process_directory(src_base_directory, dest_base_directory, min_length=-1, grid_size=2):
    """
    Processes a directory containing images and their corresponding label files.
    Filters images based on the smallest bounding box size, groups them according to
    the grid_size, and applies mosaic augmentation to each group.
    
    The resulting mosaic image (of size 1280x1280) is then resized to 640x640.
    The augmented images and their label files are saved in subdirectories under dest_base_directory.
    
    Parameters:
      src_base_directory: The base directory containing 'images' and 'labels' folders.
      dest_base_directory: The directory where augmented images and labels will be saved.
      min_length: Minimum bounding box width (in pixels) required for an image to be used.
      grid_size: The grid dimension (e.g., 2 for 2x2 mosaic, 3 for 3x3 mosaic).
    """
    # Create destination subdirectories if they do not exist
    img_dest = os.path.join(dest_base_directory, "images")
    lbl_dest = os.path.join(dest_base_directory, "labels")
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(lbl_dest, exist_ok=True)

    images_dir = os.path.join(src_base_directory, "images")
    labels_dir = os.path.join(src_base_directory, "labels")

    # Get a sorted list of image files (supporting .jpg, .jpeg, and .png formats)

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(VALID_IMAGE_EXTENSIONS)])

    # Filter images based on the minimum bounding box length criterion
    valid_files = []
    for f in image_files:
        label_path = os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt")
        if is_valid_for_mosaic(label_path, min_length, grid_size, img_size=640):
            valid_files.append(f)
    logging.info(f"Found {len(image_files)} images, of which {len(valid_files)} pass the min_length criteria for mosaic augmentation.")

    group_size = grid_size * grid_size

    # Randomize the order of valid images
    random.shuffle(valid_files)

    num_groups = len(valid_files) // group_size
    logging.info(f"Processing {num_groups} groups of {group_size} images each for a {grid_size}x{grid_size} mosaic.")

    for i in range(num_groups):
        group_files = valid_files[i * group_size:(i + 1) * group_size]
        image_paths = [os.path.join(images_dir, f) for f in group_files]
        label_paths = [os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt") for f in group_files]

        # Create the mosaic using the mosaic_augmentation function
        mosaic_img, mosaic_labels = mosaic_augmentation_letterbox(image_paths, label_paths, out_size=1280, grid_size=grid_size)

        # Resize the mosaic to 640x640 and adjust the labels accordingly
        resized_img = cv2.resize(mosaic_img, (640, 640))
        # The old mosaic size is based on the grid configuration
        old_mosaic_size = grid_size * (1280 // grid_size)
        resized_labels = resize_labels(mosaic_labels, old_size=old_mosaic_size, new_size=640)

        # Save the resulting mosaic image and label file
        mosaic_img_path = os.path.join(img_dest, f"{args.label}_mosaic_{i+1}.jpg")
        mosaic_lbl_path = os.path.join(lbl_dest, f"{args.label}_mosaic_{i+1}.txt")

        cv2.imwrite(mosaic_img_path, resized_img)
        with open(mosaic_lbl_path, 'w') as f:
            for label in resized_labels:
                cls, x, y, w, h = label
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        logging.info(f"Saved mosaic image to {mosaic_img_path} and labels to {mosaic_lbl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mosaic Augmentation Script")
    parser.add_argument("--label", type=str, required=True, help="File prefix for the image and label files.")
    parser.add_argument("--src_dir", type=str, required=True, help="Source directory containing images and labels.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory for augmented data.")
    parser.add_argument("--grid_size", type=int, default=2, help="Grid size for mosaic (e.g., 2 for 2x2 mosaic).")
    parser.add_argument("--splits", type=str, nargs='+', default=["train"], help="Dataset splits to process (e.g., train, val, test).")
    parser.add_argument("--random_crop", default=False, help="Enable random cropping before letterboxing.")
    parser.add_argument("--jitter", type=float, default=np.random.uniform(0.0, 0.5), help="Jitter fraction for random cropping.")
    parser.add_argument("--min_length", type=int, default=-1, help="Minimum bounding box length required. -1 if there is no restriction")
    parser.add_argument("--enable_letterbox", default=True, help="Enable letterboxing to maintain aspect ratio.")

    args = parser.parse_args()
    # Log the configuration options
    logging.info("===== Mosaic Augmentation Configuration =====")
    for arg_name, arg_value in vars(args).items():
        logging.info("%s: %s", arg_name, arg_value)
    logging.info("=============================================")

    for split in args.splits:
        src_split_dir = os.path.join(args.src_dir, split)
        dest_split_dir = os.path.join(args.dest_dir, split)
        process_directory(
            src_split_dir,
            dest_split_dir,
            args.min_length,
            args.grid_size,
        )

