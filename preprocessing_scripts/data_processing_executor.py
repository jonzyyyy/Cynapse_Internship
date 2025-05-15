#!/usr/bin/env python3
import subprocess
import sys
import os
import yaml # For modifying docker-compose.yml
import re   # For regular expression matching (IP address extraction)
import argparse # For parsing command-line arguments

# Common video extensions to look for
COMMON_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

# --- Helper Functions ---
def run_video_processing_script(
    script_to_run_path,
    raw_input_dir,
    final_frames_base_dir, 
    frames_factor,
    temp_dir
):
    """
    Constructs and executes the command to run the video processing script.
    The video script will save its output within final_frames_base_dir.
    Streams output in real-time.
    """
    # Ensure the target script exists
    if not os.path.exists(script_to_run_path):
        print(f"Error: Target script '{script_to_run_path}' not found.")
        return False
    # Ensure the target script is a file
    if not os.path.isfile(script_to_run_path):
        print(f"Error: Target script path '{script_to_run_path}' is not a file.")
        return False

    # Construct the command as a list of arguments
    command = [
        sys.executable,  # Path to the current Python interpreter
        script_to_run_path,
        raw_input_dir,
        final_frames_base_dir, 
        "--frames-factor", str(frames_factor),
        "--temp-dir", temp_dir
    ]

    print(f"\n--- Executing Video Processing Script ---")
    print(f"Script: {script_to_run_path}")
    print(f"Arguments:")
    print(f"  Raw Input: {raw_input_dir}")
    print(f"  Final Output Base: {final_frames_base_dir}") 
    print(f"  Frames Factor: {frames_factor}")
    print(f"  Temp Dir: {temp_dir}")
    print(f"Full command: {' '.join(command)}")
    print("----------------------------------------")
    print("--- Real-time output from video processing script: ---")

    try:
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            bufsize=1 # Line-buffered
        )

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='') 
        
        process.wait() 

        if process.returncode != 0:
            print(f"\nError: Video processing script '{script_to_run_path}' failed with exit code {process.returncode}.")
            if process.stderr:
                remaining_stderr = process.stderr.read()
                if remaining_stderr:
                    print("\n--- Video Processing Script STDERR ---")
                    print(remaining_stderr, end='')
            return False
        else:
            if process.stderr:
                remaining_stderr = process.stderr.read()
                if remaining_stderr: 
                    print("\n--- Video Processing Script STDERR (non-fatal warnings/messages): ---")
                    print(remaining_stderr, end='')
            print("\n--- Video Processing Script execution completed successfully. ---")
            return True
        
    except FileNotFoundError:
        print(f"Error: Could not find the Python interpreter ('{sys.executable}') or the script '{script_to_run_path}'.")
        print("Please check the paths.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while trying to run the video script: {e}")
        return False
    finally:
        if 'process' in locals() and process.stdout:
            process.stdout.close()
        if 'process' in locals() and process.stderr:
            process.stderr.close()

def predict_first_stage1_output_path(
    raw_input_base_dir, 
    temp_output_base_dir, 
    frames_factor
):
    """
    Predicts the output path of the first processed video from Stage 1
    (the frame-reduced video file).
    """
    print(f"\n--- Predicting Stage 1 output path for skip check ---")
    video_files = []
    for root, _, files in os.walk(raw_input_base_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in COMMON_VIDEO_EXTENSIONS:
                video_files.append(os.path.join(root, f))
    
    if not video_files:
        print(f"  No video files found in '{raw_input_base_dir}'. Cannot predict Stage 1 output.")
        return None
        
    video_files.sort() 
    first_video_full_path = video_files[0]
    print(f"  First video file for prediction: {first_video_full_path}")

    original_video_dirname = os.path.dirname(first_video_full_path)
    original_video_basename = os.path.basename(first_video_full_path)
    original_video_name_no_ext = os.path.splitext(original_video_basename)[0]

    try:
        # Relative path from the main input directory to the current video's directory
        rel_path = os.path.relpath(original_video_dirname, raw_input_base_dir)
    except ValueError:
        # This can happen if raw_input_base_dir is not a parent of original_video_dirname
        # (e.g. different drives on Windows, or if paths are tricky)
        # For simplicity, assume direct subfolder or same folder if relpath fails robustly
        rel_path = "." # Fallback, implies the video is effectively at the root for output structure
    
    if rel_path == ".": # os.path.relpath might return "." if they are the same path
        rel_path = "" # Ensure it's an empty string for os.path.join if video is in raw_input_base_dir itself

    # Stage 1 output name for the reduced video
    reduced_video_filename = f"{original_video_name_no_ext}_processed_fr{frames_factor}.avi"
    
    # Full path to the directory where the reduced video (Stage 1 output) for this specific video would be saved
    predicted_path = os.path.join(temp_output_base_dir, rel_path, reduced_video_filename)
    
    normalized_predicted_path = os.path.normpath(predicted_path)
    print(f"  Predicted Stage 1 output file path: {normalized_predicted_path}")
    return normalized_predicted_path

def determine_dynamic_app_data_path(
    raw_input_dir_for_ip_extraction, 
    base_output_dir,                 
    frames_factor
    ):
    """
    Determines the specific output subdirectory for the first video found (Stage 2 output),
    mimicking the logic of the video processing script. This path will be
    used for the /app/data Docker volume.
    """
    print(f"\n--- Determining dynamic path for /app/data (Stage 2 output) ---")
    video_files = []
    for root, _, files in os.walk(raw_input_dir_for_ip_extraction):
        for f in files:
            if os.path.splitext(f)[1].lower() in COMMON_VIDEO_EXTENSIONS:
                video_files.append(os.path.join(root, f))
    
    if not video_files:
        print(f"Warning: No video files found in '{raw_input_dir_for_ip_extraction}'. Cannot determine dynamic /app/data path.")
        return None
        
    video_files.sort() 
    first_video_full_path = video_files[0]
    print(f"  First video file found for /app/data path generation: {first_video_full_path}")

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
    print(f"  Extracted IP for folder name: {ip_for_folder}")

    base_reduced_video_name = f"{original_video_name_no_ext}_processed_fr{frames_factor}"
    frame_output_subfolder_name = f"{base_reduced_video_name}_{ip_for_folder}"
    print(f"  Generated frame output subfolder name (for Stage 2): {frame_output_subfolder_name}")

    dynamic_path = os.path.join(base_output_dir, rel_path, frame_output_subfolder_name)
    normalized_dynamic_path = os.path.normpath(dynamic_path)
    print(f"  Dynamically determined /app/data host path (Stage 2 output): {normalized_dynamic_path}")
    return normalized_dynamic_path


def modify_docker_compose_app_data_volume(docker_compose_file_path, new_app_data_host_path):
    """
    Modifies the docker-compose.yml file to update the host path for the /app/data volume
    in the 'pytorch' service.
    """
    print(f"\n--- Modifying Docker Compose File: {docker_compose_file_path} ---")
    print(f"Setting host path for '/app/data' to: {new_app_data_host_path}")

    try:
        with open(docker_compose_file_path, 'r') as f:
            compose_config = yaml.safe_load(f)

        if not compose_config:
            print(f"Error: Could not parse {docker_compose_file_path}. Is it a valid YAML file?")
            return False

        pytorch_service = compose_config.get('services', {}).get('pytorch')
        if not pytorch_service:
            print("Error: 'services.pytorch' not found in docker-compose.yml.")
            return False

        volumes = pytorch_service.get('volumes', [])
        if not isinstance(volumes, list):
            print("Error: 'services.pytorch.volumes' is not a list in docker-compose.yml.")
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
                    print(f"  Updating volume: '{volume_entry}' to '{new_volume_string}'")
                    new_volumes.append(new_volume_string)
                    volume_updated = True
                else:
                    new_volumes.append(volume_entry)
            else:
                new_volumes.append(volume_entry)
                print(f"  Skipping non-string volume entry: {volume_entry}")

        if not volume_updated:
            print(f"Warning: Volume mapping to '{target_container_path}' not found. Adding it.")
            new_volume_string = f"{new_app_data_host_path}:{target_container_path}"
            new_volumes.append(new_volume_string)
            print(f"  Added volume: '{new_volume_string}'")

        pytorch_service['volumes'] = new_volumes

        with open(docker_compose_file_path, 'w') as f:
            yaml.dump(compose_config, f, sort_keys=False, indent=2)
        
        print(f"✔ Docker Compose file updated successfully.")
        return True

    except FileNotFoundError:
        print(f"Error: Docker Compose file '{docker_compose_file_path}' not found.")
        return False
    except yaml.YAMLError as e:
        print(f"Error parsing YAML in '{docker_compose_file_path}': {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while modifying Docker Compose file: {e}")
        return False

if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Orchestrates video processing and Docker Compose setup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "raw_input_dir",
        help="Base directory containing raw input videos. Also used for IP extraction for output folder naming."
    )
    parser.add_argument(
        "final_frames_base_dir",
        help="Base directory where the video processing script will save all its output (including subfolders)."
    )
    parser.add_argument(
        "-ff", "--frames-factor",
        type=int,
        default=20,
        help="Factor by which to reduce frames in the target video processing script."
    )
    parser.add_argument(
        "-t", "--temp-dir",
        default="/mnt/nas/", 
        help="Temporary directory for the target video processing script to store Stage 1 outputs."
    )
    args = parser.parse_args()

    # --- Static Configuration (can be edited directly in the script if needed) ---
    CONFIG_VIDEO_SCRIPT_PATH = "/mnt/nas/TAmob/preprocessing_scripts/reduce_and_extract_frames.py"
    CONFIG_DOCKER_COMPOSE_DIR = "/mnt/nas/ML_Engineering/autodistill_main_RAM/" 
    CONFIG_DOCKER_DETACHED_MODE = False
    # --- End of Static Configuration ---


    print("--- Orchestrator Script Starting ---")
    print("Using the following configuration:")
    print(f"  Video Script Path (static): {CONFIG_VIDEO_SCRIPT_PATH}")
    print(f"  Raw Input Directory (from arg): {args.raw_input_dir}")
    print(f"  Base Dir for Final Frames (from arg): {args.final_frames_base_dir}")
    print(f"  Frames Factor (from arg/default): {args.frames_factor}")
    print(f"  Temporary Directory (from arg/default): {args.temp_dir}")
    
    path_for_docker_app_data = None 
    video_processing_successful = False # Initialize
    overall_success = False # Initialize

    if CONFIG_DOCKER_COMPOSE_DIR:
        print(f"  Docker Compose Directory (static): {CONFIG_DOCKER_COMPOSE_DIR}")
        print(f"  Docker Detached Mode (static): {CONFIG_DOCKER_DETACHED_MODE}")
        path_for_docker_app_data = determine_dynamic_app_data_path(
            args.raw_input_dir, 
            args.final_frames_base_dir, 
            args.frames_factor 
        )
        if not path_for_docker_app_data:
            print("Critical Error: Could not determine dynamic path for Docker /app/data. Docker steps will be skipped.")
    else:
        print("  Docker Compose step: Skipped (no directory configured)")
    print("------------------------------------")

    # Predict Stage 1 output path to check if we can skip video processing
    predicted_stage1_file = predict_first_stage1_output_path(
        args.raw_input_dir,
        args.temp_dir,
        args.frames_factor
    )

    if predicted_stage1_file and os.path.isfile(predicted_stage1_file):
        print(f"\nINFO: Stage 1 output file '{predicted_stage1_file}' already exists.")
        print("Skipping video processing script execution.")
        video_processing_successful = True 
        # If we skip, we assume the corresponding Stage 2 output (needed for Docker) also exists.
        # A warning if it doesn't exist will be helpful.
        if path_for_docker_app_data and not os.path.isdir(path_for_docker_app_data):
            print(f"WARNING: Video processing script was skipped based on Stage 1 output, "
                  f"but the expected Stage 2 output directory '{path_for_docker_app_data}' "
                  f"for Docker is missing or not a directory. Docker Compose may fail.")
            # Depending on strictness, you could set overall_success = False here
    else:
        if predicted_stage1_file:
             print(f"\nINFO: Stage 1 output file '{predicted_stage1_file}' not found. Proceeding with video processing.")
        else:
             print(f"\nINFO: Could not predict Stage 1 output file. Proceeding with video processing.")
        
        video_processing_successful = run_video_processing_script(
            CONFIG_VIDEO_SCRIPT_PATH,
            args.raw_input_dir, 
            args.final_frames_base_dir, 
            args.frames_factor, 
            args.temp_dir 
        )

    overall_success = video_processing_successful # Initial status based on video processing

    # Proceed to Docker steps if video processing was (or was assumed to be) successful
    if video_processing_successful and CONFIG_DOCKER_COMPOSE_DIR and path_for_docker_app_data:
        print(f"\nVideo processing considered successful. Proceeding to Docker Compose step.")
        
        docker_compose_file_full_path = os.path.join(CONFIG_DOCKER_COMPOSE_DIR, "docker-compose.yml")
        if not os.path.isfile(docker_compose_file_full_path):
            docker_compose_file_full_path_alt = os.path.join(CONFIG_DOCKER_COMPOSE_DIR, "docker-compose.yaml")
            if os.path.isfile(docker_compose_file_full_path_alt):
                docker_compose_file_full_path = docker_compose_file_full_path_alt
            else:
                print(f"Error: Neither 'docker-compose.yml' nor 'docker-compose.yaml' found in '{CONFIG_DOCKER_COMPOSE_DIR}'.")
                overall_success = False # Update overall success

        if overall_success: # Continue only if docker-compose file found and video processing was okay
            yaml_modification_successful = modify_docker_compose_app_data_volume(
                docker_compose_file_full_path,
                path_for_docker_app_data 
            )

            if not yaml_modification_successful:
                print("Error: Failed to modify docker-compose.yml. Skipping 'docker compose up'.")
                overall_success = False # Update overall success
            else:
                # Before running docker-compose up, ensure the host path for /app/data actually exists
                if not os.path.isdir(path_for_docker_app_data):
                    print(f"CRITICAL ERROR: The dynamically determined host path for Docker's /app/data "
                          f"('{path_for_docker_app_data}') does not exist or is not a directory. "
                          "Docker Compose will likely fail to mount the volume.")
                    print("Please ensure the video processing script created this directory or it exists from a previous run.")
                    overall_success = False # This is critical for Docker
                
                if overall_success: # Proceed only if path_for_docker_app_data exists
                    docker_compose_command = ["docker", "compose", "up"]
                    if CONFIG_DOCKER_DETACHED_MODE:
                        docker_compose_command.append("-d")
                    
                    if not os.path.isdir(CONFIG_DOCKER_COMPOSE_DIR): # Should be redundant but safe
                        print(f"Error: Docker Compose directory '{CONFIG_DOCKER_COMPOSE_DIR}' not found or is not a directory.")
                        overall_success = False
                    else:
                        print(f"\n--- Attempting to run '{' '.join(docker_compose_command)}' in {CONFIG_DOCKER_COMPOSE_DIR} ---")
                        try:
                            print(f"Executing command: {' '.join(docker_compose_command)} in directory {CONFIG_DOCKER_COMPOSE_DIR}")
                            process = subprocess.run(docker_compose_command, cwd=CONFIG_DOCKER_COMPOSE_DIR, check=True)
                            print(f"--- '{' '.join(docker_compose_command)}' completed with exit code {process.returncode}. ---")
                        except subprocess.CalledProcessError as e:
                            print(f"Error: '{' '.join(docker_compose_command)}' failed with exit code {e.returncode}.")
                            overall_success = False
                        except FileNotFoundError:
                            print("Error: 'docker compose' command not found. Is Docker Compose installed and in your PATH?")
                            overall_success = False
                        except Exception as e:
                            print(f"An unexpected error occurred while trying to run '{' '.join(docker_compose_command)}': {e}")
                            overall_success = False
                    
    elif CONFIG_DOCKER_COMPOSE_DIR: 
        if not video_processing_successful: # This implies it actually failed, not skipped
            print("\nVideo processing failed. Skipping Docker Compose modification and execution.")
        elif not path_for_docker_app_data: # This implies dynamic path determination failed
             print("\nCould not determine dynamic path for Docker /app/data. Skipping Docker Compose modification and execution.")
        # If video_processing_successful is True due to skip, but other conditions fail, it's handled inside the main 'if'

    if overall_success:
        print("\nOrchestrator: All specified steps executed successfully. 🎉")
        sys.exit(0)
    else:
        print("\nOrchestrator: Pipeline execution encountered errors or critical pre-flight checks failed. 😔")
        sys.exit(1)
