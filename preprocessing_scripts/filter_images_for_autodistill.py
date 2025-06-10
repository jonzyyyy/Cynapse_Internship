import os
import shutil
from pathlib import Path

def filter_and_flatten_images(src_root, dst_root, exclude_folders, allowed_exts=None):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    allowed_exts = allowed_exts or {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    for folder in src_root.iterdir():
        if folder.is_dir() and folder.name not in exclude_folders:
            print(f"Including: {folder.name}")
            for file in folder.iterdir():
                if file.is_file() and file.suffix.lower() in allowed_exts:
                    new_name = f"{folder.name}_{file.name}"
                    dst_file = dst_root / new_name
                    shutil.copy2(file, dst_file)
        else:
            print(f"Excluding: {folder.name}")

if __name__ == "__main__":
    # Example usage
    src_train = "/mnt/nas/TAmob/data/images/train"
    dst_train = "/mnt/nas/TAmob/old_data/stroller_checks/images/train_flat"
    exclude = ["vehicle_dataset"]

    filter_and_flatten_images(src_train, dst_train, exclude)