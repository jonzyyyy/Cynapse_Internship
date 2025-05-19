import os
import cv2
import glob
import shutil

def process_image(image_path, factor, max_width):
    """
    Process an image: if its width is greater than max_width, resize it by 'factor'
    and letterbox it to maintain the original dimensions.
    
    Returns:
        letterboxed (np.array): Processed image.
        transform (tuple): (offset_x, offset_y, factor, original_w, original_h) used for updating labels.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {image_path}")
        return None, None

    original_h, original_w = img.shape[:2]
    target_w, target_h = 640, 640

    # Compute new dimensions based on the scaling factor.
    new_w = int(original_w * factor)
    new_h = int(original_h * factor)

    # Resize the image.
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate padding to center the resized image in a 640x640 canvas.
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top

    letterboxed = cv2.copyMakeBorder(
        resized_img,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # Black padding
    )

    # Return the processed image and transformation parameters for label adjustment.
    return letterboxed, (pad_left, pad_top, factor, target_w, target_h, original_w, original_h)

def update_label(label_path, dest_label_path, transform):
    """
    Update a YOLO label file to account for the scaling and letterboxing.
    
    YOLO labels are assumed to be in the format:
        class_id x_center y_center width height
    with normalized coordinates relative to the original image dimensions.
    """
    if not os.path.exists(label_path):
        print(f"Label file {label_path} not found.")
        return

    offset_x, offset_y, factor, target_w, target_h, original_w, original_h = transform
    new_lines = []

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            # Skip malformed lines.
            continue
        class_id, x_center, y_center, width, height = parts
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width)
        height = float(height)

        # Convert from normalized to absolute coordinates.
        abs_x = x_center * original_w
        abs_y = y_center * original_h
        abs_w = width * original_w
        abs_h = height * original_h

        # Apply the transformation: scale and then offset.
        new_abs_x = offset_x + factor * abs_x
        new_abs_y = offset_y + factor * abs_y
        new_abs_w = factor * abs_w
        new_abs_h = factor * abs_h

        # Convert back to normalised coordinates relative to the original dimensions.
        new_x_center = new_abs_x / target_w
        new_y_center = new_abs_y / target_h
        new_width = new_abs_w / target_w
        new_height = new_abs_h / target_h

        new_line = f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n"
        new_lines.append(new_line)

    # Overwrite the label file with the updated content.
    with open(dest_label_path, 'w') as f:
        f.writelines(new_lines)

def process_dataset(src_dir, dest_dir, factor, max_width, suffix):
    """
    Loops through the base directory structure (train/val/test), processing images
    and updating labels where required.
    """
    for split in ['train', 'val', 'test']:
        src_images_dir = os.path.join(src_dir, split, 'images')
        src_labels_dir = os.path.join(src_dir, split, 'labels')
        dest_images_dir = os.path.join(dest_dir, split, 'images')
        dest_labels_dir = os.path.join(dest_dir, split, 'labels')

        # Create destination directories if they don't exist
        os.makedirs(dest_images_dir, exist_ok=True)
        os.makedirs(dest_labels_dir, exist_ok=True)

        # Get image files with common extensions.
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(src_images_dir, ext)))

        for image_path in image_files:
            print(f"Processing image: {image_path}")
            # Read the image to obtain its dimensions
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image {image_path}")
                continue
            original_h, original_w = img.shape[:2]
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_file = os.path.join(src_labels_dir, base_name + '.txt')
            scale_image = False

            # Check for the label file and compute the smallest bounding box width
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                bbox_widths = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            width_norm = float(parts[3])
                            abs_width = width_norm * original_w
                            bbox_widths.append(abs_width)
                        except ValueError:
                            continue
                if bbox_widths:
                    smallest_bbox = min(bbox_widths)
                    # If the smallest bounding box width is greater than the threshold, scale the image
                    if smallest_bbox > max_width:
                        scale_image = True
                else:
                    scale_image = False
            else:
                print(f"Warning: No label file found for image {image_path}")
                scale_image = False

            if scale_image:
                processed_image, transform = process_image(image_path, factor, max_width)
                if processed_image is not None:
                    orig_file_name = os.path.basename(image_path)
                    base_name, ext = os.path.splitext(orig_file_name)
                    new_file_name = base_name + suffix + ext
                    dest_image_path = os.path.join(dest_images_dir, new_file_name)
                    cv2.imwrite(dest_image_path, processed_image)

                    new_label_name = base_name + suffix + '.txt'
                    dest_label_file = os.path.join(dest_labels_dir, new_label_name)
                    update_label(label_file, dest_label_file, transform)
            # For files that do not meet the requirements, do not add them to the destination.

if __name__ == '__main__':
    # User-modifiable variables.
    factor = 0.3        # Scaling factor (e.g. 0.5 to reduce the content to 50%)
    max_width = 80       # Only process images with width greater than this value.
    src_dir = 'box_datasets/box_dataset_final/'  # Source directory containing train, val, test subfolders.
    dest_dir = 'box_datasets/box_dataset_scaled_0.3/'  # Destination directory for processed images and labels.
    suffix = f"_scale{factor}"

    process_dataset(src_dir, dest_dir, factor, max_width, suffix)
