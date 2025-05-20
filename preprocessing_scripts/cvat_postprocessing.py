import os
import zipfile
import subprocess
import shutil
import glob # For finding the zip file

# ------------------------------------------------------------------------------
# Configuration for the Orchestrator Script
# ------------------------------------------------------------------------------

# --- Paths and File Names ---
# Directory that contains the '*_relabeled.zip' file.
# This is the primary input for this orchestrator.
# Example: "/path/to/your/raw_datasets/scooter_project_files/"
INPUT_DIR_CONTAINING_ZIP = "/mnt/nas/TAmob/old_data/final_extracted_frames/07_05_2025 20_29_59 (UTC+03_00)_processed_fr20_10_197_21_24" # PLEASE UPDATE THIS

# The suffix of the zip file to look for (e.g., "_relabeled.zip")
ZIP_FILE_SUFFIX = "_relabeled.zip"

# Optional: If the data inside the zip file is within a specific subfolder.
# Leave empty ("") if data is at the root of the unzipped folder.
# Example: "data/" or "obj_train_data/"
DATA_SUBDIR_WITHIN_ZIP = "obj_train_data" # PLEASE UPDATE THIS IF NEEDED

# Path to the data_processor.py script
# Assumes it's in the same directory as this orchestrator script.
# If not, provide the full path: e.g., "/path/to/scripts/data_processor.py"
DATA_PROCESSOR_SCRIPT_PATH = "/mnt/nas/TAmob/preprocessing_scripts/preprocess_CVAT_data.py"

# --- Output Directory for Processed Data ---
# This will be the '--dest' argument for data_processor.py
# Example: "/path/to/your/processed_datasets/scooter_output/"
OUTPUT_DIR_FOR_PROCESSOR = os.path.join(INPUT_DIR_CONTAINING_ZIP, "_relabeled_split")  # PLEASE UPDATE THIS

# --- Arguments for data_processor.py ---
# These will be passed as command-line arguments to data_processor.py
PROCESSOR_TRAIN_RATIO = 0.85
PROCESSOR_SEED = 42
PROCESSOR_ACTIVATE_DELETE = True # Set to True to enable actual deletion in data_processor.py

# --- Cleanup ---
# Set to True to remove the unzipped folder after data_processor.py finishes
CLEANUP_UNZIPPED_FOLDER = False


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def find_zip_file(directory, suffix):
    """Finds a file ending with the given suffix in the directory."""
    search_pattern = os.path.join(directory, f"*{suffix}")
    zip_files = glob.glob(search_pattern)
    if not zip_files:
        print(f"Error: No file found with suffix '{suffix}' in '{directory}'.")
        return None
    if len(zip_files) > 1:
        print(f"Warning: Multiple files found with suffix '{suffix}' in '{directory}'. Using the first one: {zip_files[0]}")
    return zip_files[0]

def unzip_file(zip_path, extract_to_dir):
    """Unzips a file to the specified directory."""
    if not os.path.exists(zip_path):
        print(f"Error: Zip file not found at '{zip_path}'.")
        return False
    
    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_to_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"Unzipping '{zip_path}' to '{extract_to_dir}'...")
            zip_ref.extractall(extract_to_dir)
        print("Unzipping completed successfully.")
        return True
    except zipfile.BadZipFile:
        print(f"Error: Invalid or corrupted zip file at '{zip_path}'.")
        return False
    except Exception as e:
        print(f"An error occurred during unzipping: {e}")
        return False

# ------------------------------------------------------------------------------
# Main Orchestration Logic
# ------------------------------------------------------------------------------

def main():
    print("Starting data processing orchestration...")

    # --- 1. Validate essential paths ---
    if not os.path.isdir(INPUT_DIR_CONTAINING_ZIP):
        print(f"Error: The specified input directory '{INPUT_DIR_CONTAINING_ZIP}' does not exist. Please update the configuration.")
        return

    if not os.path.exists(DATA_PROCESSOR_SCRIPT_PATH):
        print(f"Error: The data_processor.py script was not found at '{DATA_PROCESSOR_SCRIPT_PATH}'. Please check the path.")
        return

    # --- 2. Find the zip file ---
    zip_file_path = find_zip_file(INPUT_DIR_CONTAINING_ZIP, ZIP_FILE_SUFFIX)
    if not zip_file_path:
        return # Error message already printed by find_zip_file

    print(f"Found zip file: {zip_file_path}")

    # --- 3. Determine unzip location and unzip ---
    zip_file_basename = os.path.basename(zip_file_path)
    unzip_folder_name = os.path.splitext(zip_file_basename)[0]
    unzip_target_full_path = os.path.join(INPUT_DIR_CONTAINING_ZIP, unzip_folder_name)

    print(f"Target directory for unzipping: {unzip_target_full_path}")

    # Optional: Clean up old unzipped folder if it exists and we are re-running
    if os.path.exists(unzip_target_full_path):
        print(f"Removing existing unzipped folder: {unzip_target_full_path}")
        try:
            shutil.rmtree(unzip_target_full_path)
        except OSError as e:
            print(f"Error removing existing directory {unzip_target_full_path}: {e}. Please check permissions or remove manually.")
            return
            
    if not unzip_file(zip_file_path, unzip_target_full_path):
        print("Halting orchestration due to unzipping error.")
        return

    # --- 4. Determine the source directory for data_processor.py ---
    # This is the unzipped folder, potentially with a subdirectory
    actual_src_for_processor = os.path.join(unzip_target_full_path, DATA_SUBDIR_WITHIN_ZIP)
    # Normalize the path (e.g., remove trailing slashes if DATA_SUBDIR_WITHIN_ZIP is empty)
    actual_src_for_processor = os.path.normpath(actual_src_for_processor)

    if not os.path.isdir(actual_src_for_processor):
        print(f"Error: The determined source directory for the processor '{actual_src_for_processor}' does not exist after unzipping.")
        print(f"Please check 'DATA_SUBDIR_WITHIN_ZIP' configuration and the contents of your zip file.")
        if CLEANUP_UNZIPPED_FOLDER:
            print(f"Cleaning up unzipped folder: {unzip_target_full_path}")
            shutil.rmtree(unzip_target_full_path, ignore_errors=True)
        return
        
    print(f"Source directory for data_processor.py will be: {actual_src_for_processor}")
    print(f"Destination directory for data_processor.py will be: {OUTPUT_DIR_FOR_PROCESSOR}")

    # --- 5. Prepare arguments for data_processor.py ---
    # The variable 'processor_args' contains the necessary arguments
    processor_args = [
        "python3", DATA_PROCESSOR_SCRIPT_PATH,
        "--src", actual_src_for_processor,
        "--dest", OUTPUT_DIR_FOR_PROCESSOR,
        "--train_ratio", str(PROCESSOR_TRAIN_RATIO),
        "--seed", str(PROCESSOR_SEED)
    ]
    if PROCESSOR_ACTIVATE_DELETE:
        processor_args.append("--activate_delete")

    print(f"Running data_processor.py with arguments: {' '.join(processor_args)}")

    # --- 6. Execute data_processor.py ---
    try:
        # Ensure the output directory for the processor exists
        os.makedirs(OUTPUT_DIR_FOR_PROCESSOR, exist_ok=True)

        process_result = subprocess.run(processor_args, check=True, capture_output=True, text=True)
        print("data_processor.py executed successfully.")
        print("Output from data_processor.py:")
        print(process_result.stdout)
        if process_result.stderr:
            print("Errors/Warnings from data_processor.py:")
            print(process_result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error executing data_processor.py. Return code: {e.returncode}")
        print("Output (stdout):")
        print(e.stdout)
        print("Output (stderr):")
        print(e.stderr)
    except FileNotFoundError:
        print(f"Error: Could not find Python interpreter or the script '{DATA_PROCESSOR_SCRIPT_PATH}'.")
    except Exception as e:
        print(f"An unexpected error occurred while trying to run data_processor.py: {e}")
    finally:
        # --- 7. Cleanup (Optional) ---
        if CLEANUP_UNZIPPED_FOLDER and os.path.exists(unzip_target_full_path):
            print(f"Cleaning up unzipped folder: {unzip_target_full_path}")
            try:
                shutil.rmtree(unzip_target_full_path)
                print("Unzipped folder removed successfully.")
            except OSError as e:
                print(f"Error removing unzipped folder {unzip_target_full_path}: {e}")

    print("Orchestration script finished.")

if __name__ == "__main__":
    # --- IMPORTANT: UPDATE THESE PATHS BEFORE RUNNING ---
    # You MUST change these paths to match your directory structure.
    # For example:
    # INPUT_DIR_CONTAINING_ZIP = "/mnt/c/users/youruser/datasets/scooter_data_raw"
    # DATA_SUBDIR_WITHIN_ZIP = "obj_train_data" # if scooter_data_raw/some_relabeled.zip unzips to .../obj_train_data/
    # OUTPUT_DIR_FOR_PROCESSOR = "/mnt/c/users/youruser/datasets/scooter_data_processed"
    
    # Check if placeholder paths are still there and warn user
    if "path/to/your" in INPUT_DIR_CONTAINING_ZIP or \
       "path/to/your" in OUTPUT_DIR_FOR_PROCESSOR:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! IMPORTANT: You need to update the placeholder paths in this script:    !!!")
        print("!!! 'INPUT_DIR_CONTAINING_ZIP' and 'OUTPUT_DIR_FOR_PROCESSOR'              !!!")
        print("!!! Please edit the script before running.                                 !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main()
