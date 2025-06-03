#!/usr/bin/env python3
import subprocess
import sys
import os
import yaml # For modifying docker-compose.yml
import re   # For regular expression matching (IP address extraction)
import logging # For logging

# Common video extensions to look for
COMMON_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

# --- USER CONFIGURATION SECTION ---
# Please update the variables below before running the script.

# Base directory containing raw input videos. Also used for IP extraction for output folder naming.
CONFIG_RAW_INPUT_DIR = "/Volumes/Model_Center/Tel-aviv/NewVideos2025/230525/Export 23-05-2025 20-30-00/Media player format/AXIS P5655-E PTZ Dome Network Camera (10.197.21.23) - Camera 1"

# Base directory where the video processing script will save all its FINAL (Stage 2) output (including subfolders).
CONFIG_FINAL_FRAMES_BASE_DIR = "/mnt/nas/TAmob/old_data/final_extracted_frames/" # <<< EDIT THIS

# Factor by which to reduce frames in the target video processing script.
CONFIG_FRAMES_FACTOR = 10

# Temporary directory for the video processing script to store ITS OWN INTERMEDIATE (Stage 1) outputs.
# The video processing script (CONFIG_VIDEO_SCRIPT_PATH) is responsible for using and cleaning up this directory.
CONFIG_TEMP_DIR = "/mnt/nas/video_processing_temp_stage1/" 

# Path to the video processing script (Stage 1 & 2: reduce frames, extract frames)
CONFIG_VIDEO_SCRIPT_PATH = "/mnt/nas/TAmob/preprocessing_scripts/vid2frames_pipeline/reduce_and_extract_frames.py"

# --- CVAT Preparation Script Configuration ---
# Path to your script that prepares data for CVAT.
CONFIG_CVAT_PREP_SCRIPT_PATH = "data2cvat.py"
# List of class names for CVAT (e.g., ['car', 'person', 'truck'])
CONFIG_CVAT_CLASSES = ["car", "motorcycle", "bus", "truck", "bicycle", "scooter"] 
# Subset for CVAT: 'Train' or 'Validation'
CONFIG_CVAT_SUBSET = "Train" 
# Set to True to run the CVAT preparation step, False to skip it.
CONFIG_RUN_CVAT_PREP = True 


# --- Docker Configuration ---
# Optional: Path to the directory containing 'docker-compose.yml'.
# Set to None if you don't want to run docker-compose. 
CONFIG_DOCKER_COMPOSE_DIR = "/mnt/nas/TAmob/preprocessing_scripts/autodistill_main_JON" 

# Optional: If CONFIG_DOCKER_COMPOSE_DIR is specified, set to True to run 'docker-compose up -d' (detached mode).
CONFIG_DOCKER_DETACHED_MODE = False # <<< EDIT THIS (if needed)
# --- END OF USER CONFIGURATION SECTION ---

# --- Helper Functions ---
def run_script_via_subprocess(command_list, script_friendly_name):
    """
    Generic function to run a script using subprocess, stream output, and return success.
    """
    logging.info(f"--- Executing {script_friendly_name} ---")
    logging.debug(f"Full command for {script_friendly_name}: {' '.join(command_list)}")
    logging.info(f"--- Real-time output from {script_friendly_name}: ---") # Info for the section header

    process = None
    try:
        process = subprocess.Popen(
            command_list, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            bufsize=1 # Line-buffered
        )

        # Stream stdout directly
        if process is not None and process.stdout:
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line) # Use sys.stdout.write for direct pass-through
                sys.stdout.flush()
        
        process.wait() 

        # Capture any remaining stderr after process completion
        stderr_output = ""
        if process.stderr:
            stderr_output = process.stderr.read()

        if process.returncode != 0:
            logging.error(f"{script_friendly_name} failed with exit code {process.returncode}.")
            if stderr_output:
                logging.error(f"--- {script_friendly_name} STDERR ---:\n{stderr_output.strip()}")
            return False
        else:
            if stderr_output: 
                logging.warning(f"--- {script_friendly_name} STDERR (non-fatal warnings/messages): ---:\n{stderr_output.strip()}")
            logging.info(f"--- {script_friendly_name} execution completed successfully. ---")
            return True
        
    except FileNotFoundError:
        logging.error(f"Could not find the interpreter ('{command_list[0]}') or the script ('{command_list[1]}').")
        logging.error("Please check the paths.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while trying to run {script_friendly_name}: {e}", exc_info=True)
        return False
    finally:
        if 'process' in locals() and process is not None:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()

def run_video_processing_script(
    script_to_run_path,
    raw_input_dir,
    final_frames_base_dir, 
    frames_factor,
    temp_dir 
):
    command = [
        sys.executable,
        script_to_run_path,
        raw_input_dir,
        final_frames_base_dir,
        "--frames-factor", str(frames_factor),
        "--temp-dir", temp_dir
    ]
    logging.info(f"Preparing to run Video Processing Script:")
    logging.debug(f"  Script: {script_to_run_path}")
    logging.debug(f"  Raw Input: {raw_input_dir}")
    logging.debug(f"  Final Output Base (Stage 2): {final_frames_base_dir}") 
    logging.debug(f"  Frames Factor: {frames_factor}")
    logging.debug(f"  Temp Dir (for video script's Stage 1): {temp_dir}")
    return run_script_via_subprocess(command, "Video Processing Script")

def run_cvat_preparation_script(
    script_to_run_path,
    src_dir_for_cvat, 
    cvat_classes,
    cvat_subset
):
    if not os.path.isdir(src_dir_for_cvat):
        logging.error(f"Source directory for CVAT script '{src_dir_for_cvat}' does not exist. Skipping CVAT preparation.")
        if not os.path.isdir(os.path.join(src_dir_for_cvat, "images")):
             logging.warning(f"'images' subfolder not found in '{src_dir_for_cvat}'. "
                   "The video processing script might not have created it. CVAT prep will likely fail.")
        return False

    command = [
        sys.executable,
        script_to_run_path,
        src_dir_for_cvat, 
        "--classes"
    ] + cvat_classes + [ 
        "--subset", cvat_subset
    ]
    logging.info(f"Preparing to run CVAT Preparation Script:")
    logging.debug(f"  Script: {script_to_run_path}")
    logging.debug(f"  Source Directory for CVAT: {src_dir_for_cvat}")
    logging.debug(f"  Classes: {cvat_classes}")
    logging.debug(f"  Subset: {cvat_subset}")
    return run_script_via_subprocess(command, "CVAT Preparation Script")

def determine_dynamic_app_data_path(
    raw_input_dir_for_ip_extraction, 
    final_frames_output_base_dir,          
    frames_factor
    ):
    logging.info(f"--- Determining dynamic path for Stage 2 output (and /app/data) ---")
    video_files = []
    for root, _, files in os.walk(raw_input_dir_for_ip_extraction):
        for f in files:
            if os.path.splitext(f)[1].lower() in COMMON_VIDEO_EXTENSIONS:
                video_files.append(os.path.join(root, f))
    
    if not video_files:
        logging.warning(f"No video files found in '{raw_input_dir_for_ip_extraction}'. Cannot determine dynamic Stage 2 output path.")
        return None
        
    video_files.sort() 
    first_video_full_path = video_files[0]
    logging.debug(f"  First video file found for path generation: {first_video_full_path}")

    original_video_dirname = os.path.dirname(first_video_full_path)
    original_video_basename = os.path.basename(first_video_full_path)
    original_video_name_no_ext = os.path.splitext(original_video_basename)[0]

    try:
        rel_path = os.path.relpath(original_video_dirname, raw_input_dir_for_ip_extraction)
    except ValueError: 
        rel_path = "."
    if rel_path == ".": 
        rel_path = "" 

    ip_for_folder = "unknown_camera" 
    ip_address_match = re.search(r'\((\d{1,3}(?:\.\d{1,3}){3})\)', raw_input_dir_for_ip_extraction)
    if ip_address_match:
        ip_for_folder = ip_address_match.group(1).replace(".", "_")
    logging.debug(f"  Extracted IP for folder name: {ip_for_folder}")

    base_reduced_video_name = f"{original_video_name_no_ext}_processed_fr{frames_factor}"
    frame_output_subfolder_name = f"{base_reduced_video_name}_{ip_for_folder}" 
    logging.debug(f"  Generated frame output subfolder name (for Stage 2): {frame_output_subfolder_name}")

    dynamic_path = os.path.join(final_frames_output_base_dir, rel_path, frame_output_subfolder_name)
    normalized_dynamic_path = os.path.normpath(dynamic_path)
    logging.info(f"  Dynamically determined Stage 2 output / /app/data host path: {normalized_dynamic_path}")
    return normalized_dynamic_path


def modify_docker_compose_app_data_volume(docker_compose_file_path, new_app_data_host_path):
    logging.info(f"--- Modifying Docker Compose File: {docker_compose_file_path} ---")
    logging.debug(f"Setting host path for '/app/data' to: {new_app_data_host_path}")

    try:
        with open(docker_compose_file_path, 'r') as f:
            compose_config = yaml.safe_load(f)

        if not compose_config:
            logging.error(f"Could not parse {docker_compose_file_path}. Is it a valid YAML file?")
            return False

        pytorch_service = compose_config.get('services', {}).get('pytorch')
        if not pytorch_service:
            logging.error("'services.pytorch' not found in docker-compose.yml.")
            return False

        volumes = pytorch_service.get('volumes', [])
        if not isinstance(volumes, list):
            logging.error("'services.pytorch.volumes' is not a list in docker-compose.yml.")
            return False

        volume_updated = False
        new_volumes = []
        target_container_path = "/app/data" 

        for i, volume_entry in enumerate(volumes):
            if isinstance(volume_entry, str):
                parts = volume_entry.split(':')
                if len(parts) >= 2 and parts[1].startswith(target_container_path):
                    mode = ""
                    if len(parts) > 2: 
                        mode = ":" + ":".join(parts[2:])
                    new_volume_string = f"{new_app_data_host_path}:{parts[1]}{mode}"
                    logging.debug(f"  Updating volume: '{volume_entry}' to '{new_volume_string}'")
                    new_volumes.append(new_volume_string)
                    volume_updated = True
                else:
                    new_volumes.append(volume_entry)
            else:
                new_volumes.append(volume_entry)
                logging.debug(f"  Skipping non-string volume entry: {volume_entry}")

        if not volume_updated:
            logging.warning(f"Volume mapping to '{target_container_path}' not found. Adding it.")
            new_volume_string = f"{new_app_data_host_path}:{target_container_path}"
            new_volumes.append(new_volume_string)
            logging.debug(f"  Added volume: '{new_volume_string}'")

        pytorch_service['volumes'] = new_volumes

        with open(docker_compose_file_path, 'w') as f:
            yaml.dump(compose_config, f, sort_keys=False, indent=2)
        
        logging.info(f"Docker Compose file updated successfully.")
        return True

    except FileNotFoundError:
        logging.error(f"Docker Compose file '{docker_compose_file_path}' not found.")
        return False
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML in '{docker_compose_file_path}': {e}", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while modifying Docker Compose file: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # --- Logging Configuration ---
    # Change level to logging.DEBUG to see more verbose output
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout) # Ensure logs go to stdout
        ]
    )

    logging.info("--- Orchestrator Script Starting ---")
    logging.info("Using the following configuration (defined within the script):")
    logging.info(f"  Video Processing Script Path: {CONFIG_VIDEO_SCRIPT_PATH}")
    logging.info(f"  Raw Input Directory: {CONFIG_RAW_INPUT_DIR}")
    logging.info(f"  Base Dir for Final Frames (Stage 2): {CONFIG_FINAL_FRAMES_BASE_DIR}")
    logging.info(f"  Frames Factor: {CONFIG_FRAMES_FACTOR}")
    logging.info(f"  Temporary Directory (for video script's Stage 1): {CONFIG_TEMP_DIR}")
    if CONFIG_RUN_CVAT_PREP:
        logging.info(f"  CVAT Preparation Script Path: {CONFIG_CVAT_PREP_SCRIPT_PATH}")
        logging.info(f"  CVAT Classes: {CONFIG_CVAT_CLASSES}")
        logging.info(f"  CVAT Subset: {CONFIG_CVAT_SUBSET}")
    else:
        logging.info("  CVAT Preparation step: Skipped (CONFIG_RUN_CVAT_PREP is False)")
    
    path_for_docker_app_data = None 
    video_processing_successful = False 
    docker_compose_step_flow_successful = True 
    overall_success = False 

    path_for_docker_app_data = determine_dynamic_app_data_path(
        CONFIG_RAW_INPUT_DIR, 
        CONFIG_FINAL_FRAMES_BASE_DIR, 
        CONFIG_FRAMES_FACTOR 
    )
    if not path_for_docker_app_data:
        logging.critical("Could not determine dynamic path for Stage 2 output / Docker /app/data. Further steps requiring this path will be skipped.")
        if CONFIG_RUN_CVAT_PREP or CONFIG_DOCKER_COMPOSE_DIR:
            logging.error("Orchestrator: Pipeline execution encountered critical errors.")
            sys.exit(1) 
    
    if CONFIG_DOCKER_COMPOSE_DIR: 
        logging.info(f"  Docker Compose Directory: {CONFIG_DOCKER_COMPOSE_DIR}")
        logging.info(f"  Docker Detached Mode: {CONFIG_DOCKER_DETACHED_MODE}")
    else:
        logging.info("  Docker Compose step: Skipped (no directory configured)")
    
    # --- Video Processing Step (Original Step 1 & 2 combined) ---
    expected_final_output_dir_exists = False
    expected_images_subfolder_exists = False

    if path_for_docker_app_data:
        if os.path.isdir(path_for_docker_app_data):
            expected_final_output_dir_exists = True

    if expected_final_output_dir_exists:
        logging.info(f"Expected final output directory '{path_for_docker_app_data}' for video processing already exist.")
        logging.info("Skipping video processing script execution.")
        video_processing_successful = True 
    else:
        if path_for_docker_app_data: 
            if not expected_final_output_dir_exists:
                logging.info(f"Expected final output directory '{path_for_docker_app_data}' not found. Proceeding with video processing.")
            elif not expected_images_subfolder_exists:
                logging.info(f"Expected final output directory '{path_for_docker_app_data}' exists, but its 'images' subfolder is missing. Proceeding with video processing.")
        
        video_processing_successful = run_video_processing_script(
            CONFIG_VIDEO_SCRIPT_PATH,
            CONFIG_RAW_INPUT_DIR, 
            CONFIG_FINAL_FRAMES_BASE_DIR, 
            CONFIG_FRAMES_FACTOR, 
            CONFIG_TEMP_DIR 
        )

    overall_success = video_processing_successful 

    # --- Docker Compose Step ---
    if overall_success and CONFIG_DOCKER_COMPOSE_DIR and path_for_docker_app_data:
        labeled_dir_path = os.path.join(path_for_docker_app_data, "_labeled")
        if os.path.isdir(labeled_dir_path):
            logging.info(f"'_labeled' directory found at '{labeled_dir_path}'. Skipping Docker Compose steps.")
        else:
            logging.info(f"'_labeled' directory not found in '{path_for_docker_app_data}'. Proceeding with Docker Compose steps.")
            docker_operations_were_successful = True 

            docker_compose_file_full_path = os.path.join(CONFIG_DOCKER_COMPOSE_DIR, "docker-compose.yml")
            if not os.path.isfile(docker_compose_file_full_path):
                docker_compose_file_full_path_alt = os.path.join(CONFIG_DOCKER_COMPOSE_DIR, "docker-compose.yaml")
                if os.path.isfile(docker_compose_file_full_path_alt):
                    docker_compose_file_full_path = docker_compose_file_full_path_alt
                else:
                    logging.error(f"Neither 'docker-compose.yml' nor 'docker-compose.yaml' found in '{CONFIG_DOCKER_COMPOSE_DIR}'.")
                    docker_operations_were_successful = False

            if docker_operations_were_successful: 
                yaml_modification_successful = modify_docker_compose_app_data_volume(
                    docker_compose_file_full_path,
                    path_for_docker_app_data 
                )
                if not yaml_modification_successful:
                    logging.error("Failed to modify docker-compose.yml. Skipping 'docker compose up'.")
                    docker_operations_were_successful = False
                else:
                    if not os.path.isdir(path_for_docker_app_data): 
                        logging.critical(f"The host path for Docker's /app/data "
                              f"('{path_for_docker_app_data}') does not exist or is not a directory. "
                              "Docker Compose will likely fail to mount the volume.")
                        logging.critical("Please ensure the video processing script created this directory or it exists from a previous run.")
                        docker_operations_were_successful = False
                    
                    if docker_operations_were_successful: 
                        docker_compose_command = ["docker", "compose", "up"]
                        if CONFIG_DOCKER_DETACHED_MODE:
                            docker_compose_command.append("-d")
                        
                        if not os.path.isdir(CONFIG_DOCKER_COMPOSE_DIR): 
                            logging.error(f"Docker Compose directory '{CONFIG_DOCKER_COMPOSE_DIR}' not found or is not a directory.")
                            docker_operations_were_successful = False
                        else:
                            logging.info(f"--- Attempting to run '{' '.join(docker_compose_command)}' in {CONFIG_DOCKER_COMPOSE_DIR} ---")
                            try:
                                logging.debug(f"Executing command: {' '.join(docker_compose_command)} in directory {CONFIG_DOCKER_COMPOSE_DIR}")
                                process_docker = subprocess.run(docker_compose_command, cwd=CONFIG_DOCKER_COMPOSE_DIR, check=True, capture_output=True, text=True) 
                                logging.info(f"--- '{' '.join(docker_compose_command)}' completed with exit code {process_docker.returncode}. ---")
                                if process_docker.stdout: logging.debug(f"Docker Compose STDOUT:\n{process_docker.stdout}")
                                if process_docker.stderr: logging.warning(f"Docker Compose STDERR:\n{process_docker.stderr}") # Warnings from compose can go to stderr
                            except subprocess.CalledProcessError as e:
                                logging.error(f"'{' '.join(docker_compose_command)}' failed with exit code {e.returncode}.")
                                if e.stdout: logging.error(f"Docker Compose STDOUT on error:\n{e.stdout}")
                                if e.stderr: logging.error(f"Docker Compose STDERR on error:\n{e.stderr}")
                                docker_operations_were_successful = False
                            except FileNotFoundError:
                                logging.error("'docker compose' command not found. Is Docker (with Compose V2 plugin) installed and in your PATH?")
                                docker_operations_were_successful = False
                            except Exception as e:
                                logging.error(f"An unexpected error occurred while trying to run '{' '.join(docker_compose_command)}': {e}", exc_info=True)
                                docker_operations_were_successful = False
            
            if not docker_operations_were_successful:
                overall_success = False 
                docker_compose_step_flow_successful = False


    elif CONFIG_DOCKER_COMPOSE_DIR: 
        if not overall_success: 
            logging.info("Skipping Docker Compose step because a preceding step failed.")
        elif not path_for_docker_app_data: 
             logging.info("Could not determine dynamic path for Docker /app/data. Skipping Docker Compose step.")
        docker_compose_step_flow_successful = False 
        if overall_success : 
            overall_success = False


    # --- CVAT Preparation Step (After Docker) ---
    if overall_success and docker_compose_step_flow_successful and CONFIG_RUN_CVAT_PREP: 
        if not CONFIG_CVAT_PREP_SCRIPT_PATH or not os.path.isfile(CONFIG_CVAT_PREP_SCRIPT_PATH):
            logging.error(f"CVAT Preparation script path '{CONFIG_CVAT_PREP_SCRIPT_PATH}' not configured correctly or file not found. Skipping CVAT prep.")
            overall_success = False 
        elif not path_for_docker_app_data: 
            logging.error("Source directory for CVAT preparation (Stage 2 output) not determined. Skipping CVAT prep.")
            overall_success = False
        else:
            labelled_dir_path = os.path.join(path_for_docker_app_data, "_labeled", CONFIG_CVAT_SUBSET.lower())
            expected_images_subfolder_for_cvat = os.path.join(labelled_dir_path, "images")
            if not os.path.isdir(expected_images_subfolder_for_cvat):
                logging.error(f"The 'images' subfolder is missing in the CVAT source directory: '{expected_images_subfolder_for_cvat}'.")
                logging.error("The video processing script (Stage 2) should have created this. Skipping CVAT preparation.")
                overall_success = False
            else:
                cvat_prep_successful_run = run_cvat_preparation_script( 
                    CONFIG_CVAT_PREP_SCRIPT_PATH,
                    labelled_dir_path, 
                    CONFIG_CVAT_CLASSES,
                    CONFIG_CVAT_SUBSET
                )
                if not cvat_prep_successful_run: 
                    overall_success = False 
    elif CONFIG_RUN_CVAT_PREP: 
         logging.info("Skipping CVAT preparation because a preceding step (video processing or Docker) failed or was skipped in a way that prevents CVAT prep.")


    if overall_success:
        logging.info("Orchestrator: All specified steps executed successfully.")
        sys.exit(0)
    else:
        logging.error("Orchestrator: Pipeline execution encountered errors or critical pre-flight checks failed.")
        sys.exit(1)
