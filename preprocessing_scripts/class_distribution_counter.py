import argparse
import os
import logging
from collections import Counter, defaultdict
from pathlib import Path

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

def count_class_distribution(data_dir, label_suffix=".txt", image_exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Count class distribution in YOLO-formatted dataset
    
    Args:
        data_dir: Root directory containing images and labels folders
        label_suffix: File extension for label files
        image_exts: Valid image file extensions
    
    Returns:
        Dictionary with statistics for train, val and test sets
    """
    data_dir = Path(data_dir)
    stats = {}
    
    # Check for dataset.yaml to understand directory structure
    dataset_yaml = data_dir / "dataset.yaml"
    if dataset_yaml.exists():
        logging.info(f"Found dataset.yaml at {dataset_yaml}")
    
    # Try multiple directory structures
    structures = [
        # Structure 1: Standard YOLO structure with images/ and labels/ at top level
        {"images_dir": data_dir / "images", "labels_dir": data_dir / "labels", "splits": ["train", "val", "test"]},
        
        # Structure 2: Flat structure with all images and labels directly in data_dir
        {"images_dir": data_dir, "labels_dir": data_dir, "splits": [""]},
        
        # Structure 3: train, val, test at top level, each with images/ and labels/ subdirectories
        {"images_dir": data_dir, "labels_dir": data_dir, "splits": ["train", "val", "test"],
         "structure": "split_top"}
    ]
    
    logging.info("Scanning for valid directory structures...")
    
    for structure in structures:
        images_dir = structure["images_dir"]
        labels_dir = structure["labels_dir"]
        splits = structure["splits"]
        structure_type = structure.get("structure", "standard")
        
        logging.info(f"Trying structure: images_dir={images_dir}, labels_dir={labels_dir}, type={structure_type}")
        
        for split in splits:
            # Determine directory paths based on structure type
            if structure_type == "standard":
                split_img_dir = images_dir / split if split else images_dir
                split_label_dir = labels_dir / split if split else labels_dir
            else:  # split_top structure
                split_dir = data_dir / split if split else data_dir
                split_img_dir = split_dir / "images"
                split_label_dir = split_dir / "labels"
            
            # Skip empty split names for standard structure
            if not split and structure_type == "standard":
                if not any(images_dir.glob(f"*{ext}") for ext in image_exts):
                    continue
            
            split_key = split if split else "all"
            
            logging.info(f"Checking split '{split_key}': {split_img_dir}")
            
            # Skip if directories don't exist
            if not split_img_dir.exists():
                logging.info(f"  Image directory not found: {split_img_dir}")
                continue
                
            if not split_label_dir.exists():
                logging.info(f"  Label directory not found: {split_label_dir}")
                continue
                
            logging.info(f"  Found valid directories for split '{split_key}'")
            
            # Count all valid images
            image_files = set()
            for ext in image_exts:
                image_files.update([f.stem for f in split_img_dir.glob(f"*{ext}")])
            
            logging.info(f"  Found {len(image_files)} images")
            
            # Initialize counters
            class_counts = Counter()
            images_with_class = defaultdict(set)
            total_annotations = 0
            
            # Process each label file
            label_count = 0
            valid_label_count = 0
            empty_label_count = 0
            non_yolo_count = 0
            missing_image_count = 0
            
            # Get ALL image files in image directory and show count for debugging
            all_image_files = []
            for ext in image_exts:
                found_files = list(split_img_dir.glob(f"**/*{ext}"))  # Use ** to search recursively
                all_image_files.extend(found_files)
                logging.info(f"  Found {len(found_files)} {ext} files in {split_img_dir}")
            
            # Get ALL label files in label directory
            all_label_files = list(split_label_dir.glob(f"**/*{label_suffix}"))  # Use ** to search recursively
            logging.info(f"  Found {len(all_label_files)} label files in {split_label_dir}")
            
            # No files found? Check subdirectories
            if not all_image_files and not all_label_files:
                logging.info(f"  No files found in expected directories. Checking subdirectories...")
                subdirs_img = [d for d in split_img_dir.glob("*") if d.is_dir()]
                subdirs_lbl = [d for d in split_label_dir.glob("*") if d.is_dir()]
                logging.info(f"  Image subdirectories: {[d.name for d in subdirs_img]}")
                logging.info(f"  Label subdirectories: {[d.name for d in subdirs_lbl]}")
                
                # Try listing a few files from each subdirectory
                for subdir in subdirs_img[:3]:  # First 3 subdirs only
                    files = list(subdir.glob(f"*{image_exts[0]}"))[:5]  # First 5 files only
                    logging.info(f"  Files in {subdir.name}: {[f.name for f in files]}")
            
            # Print first few label files for debugging (if any found)
            first_labels = all_label_files[:5]
            if first_labels:
                logging.info(f"  First few label files: {[l.name for l in first_labels]}")
                # Print content of first label file for debugging
                try:
                    with open(first_labels[0], 'r') as f:
                        content = f.read().strip()
                        logging.info(f"  Example label file content ({first_labels[0].name}):\n    {content[:200]}{'...' if len(content) > 200 else ''}")
                except (IndexError, FileNotFoundError, PermissionError) as e:
                    logging.info(f"  Could not read label file: {e}")
            
            # Try a more direct approach to locate files
            if not all_image_files or not all_label_files:
                logging.info("  Using direct file system listing to locate files...")
                import subprocess
                try:
                    # Use find command to locate files (works on Linux/Mac)
                    img_cmd = f"find {split_img_dir} -type f -name '*{image_exts[0]}' | head -5"
                    lbl_cmd = f"find {split_label_dir} -type f -name '*{label_suffix}' | head -5"
                    
                    img_result = subprocess.run(img_cmd, shell=True, capture_output=True, text=True)
                    lbl_result = subprocess.run(lbl_cmd, shell=True, capture_output=True, text=True)
                    
                    if img_result.stdout:
                        logging.info(f"  Found images with 'find':\n{img_result.stdout.strip()}")
                    if lbl_result.stdout:
                        logging.info(f"  Found labels with 'find':\n{lbl_result.stdout.strip()}")
                except Exception as e:
                    logging.info(f"  Error running find command: {e}")
            
            # Build a mapping from image filenames to full paths
            image_files = {}
            for img_file in all_image_files:
                image_files[img_file.stem] = img_file
            
            logging.info(f"  Processing {len(all_label_files)} label files...")
            
            # Fall back to flat file structure if no valid files found with expected structure
            if not all_label_files:
                logging.warning(f"  Couldn't find label files in expected locations. Please check paths.")
                logging.info(f"  You might need to specify a different path. Current path: {data_dir}")
                continue
                
            for label_file in all_label_files:
                base_name = label_file.stem
                label_count += 1
                
                # Skip if no corresponding image exists
                if base_name not in image_files:
                    missing_image_count += 1
                    if missing_image_count <= 5:  # Only log first few
                        logging.info(f"  No image for label: {base_name}")
                    continue
                
                # Parse the label file
                with open(label_file, "r") as f:
                    content = f.read().strip()
                    if not content:
                        empty_label_count += 1
                        continue
                        
                    file_classes = set()
                    lines = content.split('\n')
                    for line in lines:
                        parts = line.strip().split()
                        # Check for YOLO format (class_id x_center y_center width height)
                        if len(parts) >= 5:
                            try:
                                class_id = parts[0]
                                # Validate coordinates are float values
                                float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                class_counts[class_id] += 1
                                file_classes.add(class_id)
                                total_annotations += 1
                            except ValueError:
                                non_yolo_count += 1
                        else:
                            non_yolo_count += 1
                    
                    # Record which images have which classes
                    for class_id in file_classes:
                        images_with_class[class_id].add(base_name)
                    
                    if file_classes:
                        valid_label_count += 1
            
            logging.info(f"  Label files: {label_count} total, {valid_label_count} valid YOLO format")
            if empty_label_count > 0:
                logging.info(f"  Found {empty_label_count} empty label files")
            if non_yolo_count > 0:
                logging.info(f"  Found {non_yolo_count} non-YOLO format lines")
            if missing_image_count > 0:
                logging.info(f"  Found {missing_image_count} label files without matching images")
                
            # Check dataset.yaml for class names
            try:
                if dataset_yaml.exists():
                    import yaml
                    with open(dataset_yaml, 'r') as f:
                        yaml_content = yaml.safe_load(f)
                        if 'names' in yaml_content:
                            logging.info(f"  Class names from dataset.yaml: {yaml_content['names']}")
            except Exception as e:
                logging.info(f"  Error reading dataset.yaml: {e}")
            
            # Calculate statistics
            total_images = len(set().union(*images_with_class.values())) if images_with_class else 0
            
            # Group classes 0-4 together
            images_with_class_0_4 = set()
            for class_id in range(5):  # 0-4
                images_with_class_0_4.update(images_with_class.get(str(class_id), set()))
            
            stats[split_key] = {
                "total_images": total_images,
                "total_annotations": total_annotations,
                "class_counts": dict(class_counts),
                "images_per_class": {class_id: len(images) for class_id, images in images_with_class.items()},
                "class_0_4_images": len(images_with_class_0_4),
                "class_5_images": len(images_with_class.get("5", set()))
            }
            
            if total_images > 0:
                # We found valid data, so stop trying other structures
                break
    
    if not stats:
        logging.warning("No valid data found in any directory structure!")
        # Add empty stat to ensure something is returned
        stats["all"] = {
            "total_images": 0,
            "total_annotations": 0,
            "class_counts": {},
            "images_per_class": {},
            "class_0_4_images": 0,
            "class_5_images": 0
        }
    
    return stats

def format_stats(stats, detailed=False):
    """Format statistics into a readable string"""
    output = []
    
    if not stats:
        return "No data found!"
    
    for split, split_stats in stats.items():
        if not split_stats["total_images"]:
            output.append(f"{split.capitalize()}: No valid images found")
            continue
            
        output.append(f"{split.capitalize()}: (Total: {split_stats['total_images']})")
        
        # Group classes 0-4
        output.append(f"Class ID '0' - '4': {split_stats['class_0_4_images']} images")
        
        # Individual classes if detailed view requested
        if detailed:
            for i in range(5):
                class_id = str(i)
                count = split_stats['images_per_class'].get(class_id, 0)
                output.append(f"  Class ID '{class_id}': {count} images")
        
        output.append(f"Class ID '5': {split_stats['images_per_class'].get('5', 0)} images")
        output.append("")
        
        # Optional: detailed annotation counts
        if detailed:
            output.append(f"Total annotations: {split_stats['total_annotations']}")
            output.append("Annotations per class:")
            for class_id, count in sorted(split_stats['class_counts'].items()):
                output.append(f"  Class '{class_id}': {count} annotations")
            output.append("")
    
    if not any(split_stats["total_images"] > 0 for split_stats in stats.values()):
        output.append("WARNING: No valid data found in the specified directory!")
    
    return "\n".join(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate class distribution in YOLO dataset")
    parser.add_argument("--data_dir", required=True, help="Root directory containing the dataset")
    parser.add_argument("--output", help="Optional file to save statistics (otherwise print to console)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-class statistics")
    
    args = parser.parse_args()
    
    logging.info(f"Analyzing data in {args.data_dir}")
    stats = count_class_distribution(args.data_dir)
    
    # Format and output the statistics
    formatted_stats = format_stats(stats, args.detailed)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(formatted_stats)
        logging.info(f"Statistics saved to {args.output}")
    else:
        print("\n" + "-"*50)
        print("DATASET STATISTICS:")
        print("-"*50)
        print(formatted_stats)
        print("-"*50)
    
    logging.info("Analysis complete")
