#!/usr/bin/env python3
import os
# turn off OpenCV’s internal codec warnings
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
from tqdm import tqdm
import shutil
import re
import argparse

# --- Configuration ---
COMMON_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

# RAW_VIDEOS_INPUT_BASE_DIRECTORY will now be passed as a parameter where needed,
# specifically for the IP extraction logic.

def get_video_duration(video_path):
    """Get duration in minutes (not used in main pipeline but kept for completeness)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: cannot open {video_path} to get duration.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps == 0:
        return 0.0
    return (total / fps) / 60


def process_video_for_frame_reduction(
    input_video_path,
    stage1_output_base_dir,
    # This is the top-level directory from which relative paths for output are calculated
    raw_videos_input_base_dir_for_relpath, 
    frames_reduction_factor
):
    """
    Stage 1: downsample frames by skipping 'frames_reduction_factor' frames,
    write out as AVI with XVID.
    """
    print(f"\n--- Stage 1: Processing {input_video_path} ---")
    base = os.path.splitext(os.path.basename(input_video_path))[0]
    try:
        # Calculate relative path from the main input directory to the current video's directory
        rel = os.path.relpath(
            os.path.dirname(input_video_path),
            raw_videos_input_base_dir_for_relpath 
        )
    except ValueError:
        # Fallback if input_video_path is not under raw_videos_input_base_dir_for_relpath
        # (e.g., if they are on different drives on Windows)
        rel = "." 

    out_dir = os.path.join(stage1_output_base_dir, rel)
    os.makedirs(out_dir, exist_ok=True)

    # change extension to .avi
    out_path = os.path.join(out_dir, f"{base}_processed_fr{frames_reduction_factor}.avi")
    print(f" → saving reduced video to: {out_path}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_video_path}")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if w == 0 or h == 0 or fps == 0 or total == 0:
        print(f"Error: invalid video properties for {input_video_path} (w:{w},h:{h},fps:{fps},total:{total})")
        cap.release()
        return None

    if frames_reduction_factor <= 0:
        print(f"Error: frames reduction factor must be >0, got {frames_reduction_factor}")
        cap.release()
        return None

    # XVID in an AVI container
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not out.isOpened():
        print(f"Error: cannot open VideoWriter for {out_path}")
        cap.release()
        return None

    num_to_write = total // frames_reduction_factor
    if num_to_write == 0:
        print(f"Warning: video too short (total frames: {total}) for skip factor {frames_reduction_factor}. No frames will be written.")
        out.release()
        cap.release()
        # Clean up empty output file if created
        if os.path.exists(out_path) and os.path.getsize(out_path) == 0:
            try:
                os.remove(out_path)
                print(f"Removed empty file: {out_path}")
            except OSError as e:
                print(f"Warning: could not remove empty file {out_path}: {e}")
        return None

    for i in tqdm(range(num_to_write), desc=f"Reducing {base} (factor {frames_reduction_factor})"):
        idx = i * frames_reduction_factor
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: could not read frame {idx} from {input_video_path}. Stopping reduction for this video.")
            break
        out.write(frame)

    out.release()
    cap.release()
    print(f"✔ Stage 1 done: {out_path}")
    return out_path


def reduce_frames_in_all_subdirectories(
    raw_videos_input_base_dir, # Main input directory for videos
    stage1_output_base_dir,    # Where to save reduced videos
    frames_reduction_factor
):
    """
    Walks through raw_videos_input_base_dir, finds videos, and processes them
    using process_video_for_frame_reduction.
    """
    print("\n=== Stage 1: Frame Reduction ===")
    os.makedirs(stage1_output_base_dir, exist_ok=True)
    found_any = False
    processed_video_paths = []

    for root, _, files in os.walk(raw_videos_input_base_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in COMMON_VIDEO_EXTENSIONS:
                found_any = True
                full_video_path = os.path.join(root, fn)
                # Pass raw_videos_input_base_dir for correct relative path calculation
                output_path = process_video_for_frame_reduction(
                    full_video_path, 
                    stage1_output_base_dir,
                    raw_videos_input_base_dir, # This is the base for relpath
                    frames_reduction_factor
                )
                if output_path:
                    processed_video_paths.append(output_path)

    if not found_any:
        print(f"No videos found under {raw_videos_input_base_dir}")
    print(f"=== Stage 1 complete: {len(processed_video_paths)} videos processed ===\n")
    return processed_video_paths


def extract_frames_from_single_video(
    reduced_video_path,
    stage2_output_base_dir, # Base directory for all extracted frames
    stage1_output_base_dir_for_relpath, # Base of reduced videos, for relpath calc
    original_raw_input_dir_for_ip_extraction # Original raw input dir for IP extraction
):
    """
    Stage 2: extract each frame from a reduced video into
      <basename_of_reduced_video>_<camera_ip_or_unknown_camera>/
    The IP is extracted from the 'original_raw_input_dir_for_ip_extraction' path.
    """
    print(f"\n--- Stage 2: Extracting frames from {reduced_video_path} ---")
    base_reduced_video_name = os.path.splitext(os.path.basename(reduced_video_path))[0]
    try:
        # Relative path from the base of reduced videos (temp_dir)
        rel = os.path.relpath(
            os.path.dirname(reduced_video_path),
            stage1_output_base_dir_for_relpath
        )
    except ValueError:
        rel = "."

    # Attempt to grab IP from the original raw input directory path string
    # This logic assumes the IP address is embedded in the path like "...(10.197.21.23)..."
    ip_address = "unknown_camera" # Default
    if original_raw_input_dir_for_ip_extraction:
        match = re.search(r'\((\d{1,3}(?:\.\d{1,3}){3})\)', original_raw_input_dir_for_ip_extraction)
        if match:
            ip_address = match.group(1).replace(".","_") # Replace dots for folder name
    
    # Create a unique folder name for frames from this video
    # e.g., video1_processed_fr20_10_197_21_23
    frame_output_subfolder_name = f"{base_reduced_video_name}_{ip_address}"

    # Full path to the directory where frames for this specific video will be saved
    final_frames_output_dir_for_video = os.path.join(stage2_output_base_dir, rel, frame_output_subfolder_name)
    os.makedirs(final_frames_output_dir_for_video, exist_ok=True)
    print(f" → saving frames to: {final_frames_output_dir_for_video}")

    cap = cv2.VideoCapture(reduced_video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {reduced_video_path} for frame extraction.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames == 0:
        print(f"Warning: {reduced_video_path} has no frames to extract.")
        cap.release()
        return

    for i in tqdm(range(total_frames), desc=f"Extracting {base_reduced_video_name}"):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: could not read frame {i} from {reduced_video_path}. Stopping extraction for this video.")
            break
        # Frame filename: e.g., video1_processed_fr20_10_197_21_23_frame_000000.jpg
        frame_filename = f"{frame_output_subfolder_name}_frame_{i:06d}.jpg"
        cv2.imwrite(os.path.join(final_frames_output_dir_for_video, frame_filename), frame)

    cap.release()
    print(f"✔ Extracted {total_frames} frames into: {final_frames_output_dir_for_video}")


def extract_frames_from_all_reduced_videos(
    stage1_output_base_dir, # Directory containing processed (reduced) videos
    stage2_output_base_dir, # Top-level directory for final extracted frames
    original_raw_input_dir    # Original raw input dir, passed for IP extraction
):
    """
    Walks through stage1_output_base_dir, finds reduced videos, and extracts frames.
    """
    print("\n=== Stage 2: Frame Extraction ===")
    os.makedirs(stage2_output_base_dir, exist_ok=True)
    found_any_reduced_videos = False
    processed_count = 0

    for root, _, files in os.walk(stage1_output_base_dir):
        # Look for videos processed by stage 1 (they have "_processed_fr" in their name)
        reduced_videos_in_dir = [f for f in files if "_processed_fr" in f and f.lower().endswith(".avi")]
        
        if not reduced_videos_in_dir:
            continue
        
        found_any_reduced_videos = True
        for reduced_video_filename in reduced_videos_in_dir:
            full_reduced_video_path = os.path.join(root, reduced_video_filename)
            extract_frames_from_single_video(
                full_reduced_video_path,
                stage2_output_base_dir,
                stage1_output_base_dir, # This is the base for relpath calculation
                original_raw_input_dir    # Pass for IP extraction
            )
            processed_count +=1

    if not found_any_reduced_videos:
        print(f"No reduced videos found under {stage1_output_base_dir} to extract frames from.")
    print(f"=== Stage 2 complete: attempted frame extraction for {processed_count} videos ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-stage video processing: 1. Reduce frame rate, 2. Extract frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    
    # Positional arguments
    parser.add_argument(
        "raw_input_dir", 
        help="Base directory containing raw input videos. Subdirectories will be scanned."
    )
    parser.add_argument(
        "final_frames_dir", 
        help="Base directory where final extracted frames will be saved, maintaining relative structure."
    )
    
    # Optional arguments
    parser.add_argument(
        "-f", "--frames-factor", 
        type=int, 
        default=20,
        help="Factor by which to reduce frames. E.g., 20 means 1 out of every 20 frames is kept."
    )
    parser.add_argument(
        "-t", "--temp-dir", 
        default="/mnt/nas/temp_frame_reduced_videos/", # Default temporary directory
        help="Temporary directory to store intermediate frame-reduced videos. This directory will be deleted after processing if cleanup is successful."
    )

    args = parser.parse_args()

    # Use parsed arguments
    # Ensure paths are absolute for robustness, especially if script changes CWD
    # Though os.path.join and os.walk generally handle relative paths fine.
    # For this script, relative paths from CWD should work as long as they are correct.
    cfg_raw_input_dir = args.raw_input_dir
    cfg_final_frames_dir = args.final_frames_dir
    cfg_frames_factor = args.frames_factor
    cfg_temp_reduced_dir = args.temp_dir

    print("--- Starting Two-Stage Video Processing ---")
    print(f"Configuration:")
    print(f"  Raw Input Directory: {cfg_raw_input_dir}")
    print(f"  Final Frames Directory: {cfg_final_frames_dir}")
    print(f"  Frames Reduction Factor: {cfg_frames_factor}")
    print(f"  Temporary Directory: {cfg_temp_reduced_dir}")
    print("-------------------------------------------")


    # Stage 1: Reduce frames
    # reduce_frames_in_all_subdirectories returns a list of paths to processed videos,
    # but it's not strictly needed for the next stage as it re-scans the temp_dir.
    # We keep it for potential future use or logging.
    processed_reduced_videos = reduce_frames_in_all_subdirectories(
        cfg_raw_input_dir,
        cfg_temp_reduced_dir,
        cfg_frames_factor
    )

    # Stage 2: Extract frames from the reduced videos
    if not processed_reduced_videos and not os.path.exists(cfg_temp_reduced_dir):
         print("No videos were processed in Stage 1, or temporary directory does not exist. Skipping Stage 2 and cleanup.")
    else:
        extract_frames_from_all_reduced_videos(
            cfg_temp_reduced_dir,
            cfg_final_frames_dir,
            cfg_raw_input_dir # Pass the original raw input dir for IP extraction
        )

        # Cleanup temporary directory
        if os.path.exists(cfg_temp_reduced_dir):
            try:
                print(f"\nAttempting to delete temporary folder: {cfg_temp_reduced_dir}")
                shutil.rmtree(cfg_temp_reduced_dir)
                print(f"Successfully deleted temp folder: {cfg_temp_reduced_dir}")
            except OSError as e:
                print(f"Warning: Could not delete temp folder {cfg_temp_reduced_dir}. Error: {e}")
                print("You may need to delete it manually.")
        else:
            print(f"Temporary directory {cfg_temp_reduced_dir} not found, skipping cleanup.")


    print("\n--- Video processing pipeline finished! ---")
