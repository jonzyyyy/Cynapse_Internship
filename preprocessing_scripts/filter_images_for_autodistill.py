import os
import shutil
import json
import argparse
from pathlib import Path

def filter_and_flatten_images(src_root, dst_root, exclude_folders, allowed_exts=None, create_mapping=True, copy_labels=True, labels_dir=None, train_subfolder=True):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    allowed_exts = allowed_exts or {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # Set up destination directories according to the standard structure:
    # dst_root/train/images/ and dst_root/train/labels/
    dst_images_root = dst_root / "train" / "images"
    dst_images_root.mkdir(parents=True, exist_ok=True)
    if copy_labels:
        dst_labels_root = dst_root / "train" / "labels"
        dst_labels_root.mkdir(parents=True, exist_ok=True)
    
    # Set up labels directory paths for source
    if copy_labels:
        # If labels_dir not provided, try to infer it by replacing "images" with "labels" in path
        if not labels_dir and "images" in str(src_root):
            src_labels_root = Path(str(src_root).replace("images", "labels"))
        else:
            src_labels_root = Path(labels_dir) if labels_dir else src_root
            
        print(f"Labels will be copied from {src_labels_root} to {dst_labels_root}")
    
    # Dictionary to track original paths and new names
    file_mapping = {}
    
    for folder in src_root.iterdir():
        if folder.is_dir() and folder.name not in exclude_folders:
            print(f"Including: {folder.name}")
            for file in folder.iterdir():
                if file.is_file() and file.suffix.lower() in allowed_exts:
                    # Process image file
                    new_name = f"{folder.name}_{file.name}"
                    dst_file = dst_images_root / new_name
                    shutil.copy2(file, dst_file)
                    
                    # Record the mapping
                    if create_mapping:
                        file_mapping[str(dst_file)] = str(file)
                    
                    # Look for corresponding label file
                    if copy_labels:
                        label_file = src_labels_root / folder.name / f"{file.stem}.txt"
                        if label_file.exists():
                            new_label_name = f"{folder.name}_{file.stem}.txt"
                            dst_label_file = dst_labels_root / new_label_name
                            shutil.copy2(label_file, dst_label_file)
                            if create_mapping:
                                file_mapping[str(dst_label_file)] = str(label_file)
                        else:
                            print(f"  Warning: No label found for {file.name}")
        else:
            print(f"Excluding: {folder.name}")
    
    # Save mapping to a JSON file
    if create_mapping and file_mapping:
        # Always save mapping file at the top level directory
        mapping_file = dst_root / "file_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(file_mapping, f, indent=4)
        print(f"File mapping saved to {mapping_file}")
    
    return file_mapping

def restore_files_from_flat(flat_dir, output_dir, mapping_file=None, restore_to_original=False, train_subfolder=True):
    """Restore files from flattened directory to their original structure
    
    Args:
        flat_dir: Directory containing flattened files
        output_dir: Directory where the restored structure will be created
        mapping_file: Optional path to the JSON mapping file
        restore_to_original: If True and mapping_file provided, restore files to their original locations
        train_subfolder: Whether the flattened files are in a train subfolder
    """
    flat_dir = Path(flat_dir)
    output_dir = Path(output_dir)
    
    # Adjust flat_dir if using train subfolder structure
    if train_subfolder:
        # Check if flat_dir itself is the train directory
        if flat_dir.name == "train":
            flat_images_dir = flat_dir / "images"
            flat_labels_dir = flat_dir / "labels"
        else:
            # Assume flat_dir is the parent directory containing train/
            flat_images_dir = flat_dir / "train" / "images"
            flat_labels_dir = flat_dir / "train" / "labels"
    else:
        # Original structure
        flat_images_dir = flat_dir
        if "images" in str(flat_dir):
            flat_labels_dir = Path(str(flat_dir).replace("images", "labels"))
        else:
            flat_labels_dir = flat_dir / "labels"
    
    # Look for mapping file if not specified
    if mapping_file is None:
        mapping_file = flat_dir / "file_mapping.json"
        if not mapping_file.exists() and flat_dir.parent.exists():
            parent_mapping = flat_dir.parent / "file_mapping.json"
            if parent_mapping.exists():
                mapping_file = parent_mapping
                print(f"Using mapping file from parent directory: {mapping_file}")
    
    # Verify directories exist
    if not flat_images_dir.exists():
        print(f"Warning: Images directory not found: {flat_images_dir}")
        if train_subfolder:
            print("Try using --no-train-subfolder if your files aren't in a train/ subdirectory")
        return
    
    # Create output directories if not restoring to original locations
    if not restore_to_original:
        output_dir.mkdir(parents=True, exist_ok=True)
        if "images" in str(output_dir):
            output_labels_dir = Path(str(output_dir).replace("images", "labels"))
            output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # If mapping file is provided, use it for precise restoration
    if mapping_file and Path(mapping_file).exists():
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
            
        for dst_file, src_file in mapping.items():
            dst_path = Path(dst_file)
            src_path = Path(src_file)
            
            # If the dst_path in the mapping doesn't match our flat structure,
            # we need to adjust the path
            if train_subfolder and "train" not in str(dst_path) and dst_path.exists():
                # The mapping was created without train subfolder, but we're using one
                pass
            elif train_subfolder and "train" in str(dst_path) and not dst_path.exists():
                # The mapping was created with train subfolder, but the file is directly in flat_dir
                parts = dst_path.parts
                train_index = parts.index("train")
                adjusted_path = Path(*parts[:train_index]) / parts[train_index+2:]  # Skip train/images or train/labels
                if adjusted_path.exists():
                    dst_path = adjusted_path
            
            # Skip if file doesn't exist after adjustments
            if not dst_path.exists():
                print(f"Warning: Source file not found: {dst_path}")
                continue
                
            if restore_to_original:
                # Restore directly to original location
                output_path = src_path
            else:
                # Restore to specified output directory with original folder structure
                if ".txt" in str(dst_path) and "labels" not in str(output_dir):
                    # Handle label files going to a separate directory
                    if "images" in str(output_dir):
                        base_output_dir = Path(str(output_dir).replace("images", "labels"))
                    else:
                        base_output_dir = output_dir / "labels"
                else:
                    base_output_dir = output_dir
                    
                output_path = base_output_dir / src_path.parent.name / src_path.name
                
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                shutil.copy2(dst_path, output_path)
                print(f"Restored: {output_path}")
            except FileNotFoundError:
                print(f"Warning: Could not find {dst_path}")
        
        return
    
    # If no mapping file, try to infer original structure from filenames
    for directory in [flat_images_dir, flat_labels_dir]:
        if not directory.exists():
            continue
            
        for file in directory.iterdir():
            if file.is_file() and file.name != "file_mapping.json":
                # Expect filename format: "folder_filename.ext"
                parts = file.stem.split('_', 1)
                if len(parts) > 1:
                    folder_name, original_name = parts
                    original_name = original_name + file.suffix
                    
                    # Determine destination directory
                    if ".txt" in file.name and directory == flat_labels_dir:
                        if "images" in str(output_dir):
                            base_dir = str(output_dir).replace("images", "labels")
                        else:
                            base_dir = output_dir / "labels"
                        dest_folder = Path(base_dir) / folder_name
                    else:
                        dest_folder = output_dir / folder_name
                    
                    dest_folder.mkdir(parents=True, exist_ok=True)
                    
                    output_path = dest_folder / original_name
                    shutil.copy2(file, output_path)
                    print(f"Restored: {output_path}")
                else:
                    print(f"Warning: Cannot parse folder from {file.name}, skipping")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter, flatten and restore image directories")
    parser.add_argument('--mode', choices=['flatten', 'restore'], default='flatten', 
                        help='Operation mode: flatten or restore')
    parser.add_argument('--src', help='Source directory')
    parser.add_argument('--dst', help='Destination directory')
    parser.add_argument('--exclude', nargs='+', default=[], help='Folders to exclude')
    parser.add_argument('--mapping', help='Path to mapping file (for restore mode)')
    parser.add_argument('--original', action='store_true', 
                        help='Restore to original locations instead of destination directory')
    parser.add_argument('--labels-dir', help='Source directory for labels (if different from images path)')
    parser.add_argument('--no-labels', action='store_true', help='Skip processing label files')
    
    args = parser.parse_args()
    
    if not args.src or not args.dst:
        # Example usage when no arguments provided
        if args.mode == 'flatten':
            src_train = "/mnt/nas/TAmob/data/images/train"
            dst_train = "/mnt/nas/TAmob/old_data/recent_data"  # Creates train/images/ and train/labels/
            exclude = ["vehicle_dataset", "scooter_dataset_V7_relabeled_split"]
            
            # This will create:
            # /mnt/nas/TAmob/old_data/recent_data/train/images/
            # /mnt/nas/TAmob/old_data/recent_data/train/labels/
            filter_and_flatten_images(src_train, dst_train, exclude, copy_labels=True)
        else:
            # For restore mode
            src = "/mnt/nas/TAmob/old_data/recent_data"  # Contains train/images and train/labels
            dst = "/mnt/nas/TAmob/old_data/recent_data/restored"
            mapping = "/mnt/nas/TAmob/old_data/recent_data/file_mapping.json"
            original = True
            restore_files_from_flat(src, dst, mapping, original)
    else:
        if args.mode == 'flatten':
            filter_and_flatten_images(args.src, args.dst, args.exclude, 
                                    copy_labels=not args.no_labels,
                                    labels_dir=args.labels_dir)
        else:  # restore mode
            restore_files_from_flat(args.src, args.dst, args.mapping, args.original)