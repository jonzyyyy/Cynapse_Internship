import os
import subprocess
import csv
from collections import defaultdict
import shutil
import json
import argparse

POSSIBLE_IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def main_pipeline(
    input_and_label_dir,
    class_map_json,
    diversified_csvs,
    output_folder,
    batch_size=8,
    subset="Train"
):
    # Ensure output folder and embeddings subdir exist
    os.makedirs(output_folder, exist_ok=True)
    embeddings_dir = os.path.join(output_folder, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    input_dir = os.path.join(input_and_label_dir, "train", "images")
    label_dir = os.path.join(input_and_label_dir, "train", "labels")
    confidence_csv = os.path.join(input_and_label_dir, "object_confidences.csv")
    # Standard output file names (now as CSVs)
    priority_list_csv = os.path.join(output_folder, "priority_review.csv")
    outlier_output_csv = os.path.join(output_folder, "outliers.csv")
    merged_priority_csv = os.path.join(os.path.dirname(priority_list_csv), "merged_priority_review.csv")
    data_review_dir = os.path.join(output_folder, "data_for_review")

    with open(class_map_json) as f:
        class_id_to_name = json.load(f)
        class_list = [class_id_to_name[str(i)] for i in range(len(class_id_to_name))]

    # 1. Hybrid Embedding Extraction - Save to output_folder/embeddings
    subprocess.run([
        "python3", "hybrid_embedding_extractor.py",
        "--image_dir", input_dir,
        "--label_dir", label_dir,
        "--class_map", class_map_json,
        "--dest_dir", embeddings_dir,
        "--batch_size", str(batch_size)
    ], check=True)

    # 2. Outlier/Noise Detection - Use embeddings_dir in output_folder
    subprocess.run([
        "python3", "outlier_detection.py",
        "--embeddings_dir", embeddings_dir,
        "--label_dir", label_dir,
        "--class_map", class_map_json,
        "--output_path", outlier_output_csv
    ], check=True)

    # 3 & 4. Committee Disagreement & Uncertainty/Diversity Sampling - Use embeddings_dir in output_folder
    # diversified_args = [
    #     "python3", "diversified_sampling.py",
    #     "--embeddings_dir", embeddings_dir,
    #     "--label_dir", label_dir,
    #     "--conf_csv", confidence_csv,
    #     "--output_path", priority_list_csv,
    #     "--uncertainty_percentile", "10.0",
    #     "--diversity_k", "50"
    # ]
    # if diversified_csvs:
    #     diversified_args += ["--committee_csvs"] + diversified_csvs
    # subprocess.run(diversified_args, check=True)

    # 5. Construct Combined Priority List for Review (merge CSVs with grouped indices)
    priority_indices = defaultdict(set)
    files_to_merge = []
    for fpath in [outlier_output_csv, priority_list_csv]:
        if os.path.exists(fpath):
            files_to_merge.append(fpath)

    if not files_to_merge:
        print("No priority or outlier CSVs found, skipping merging step.")
    else:
        for fpath in files_to_merge:
            with open(fpath, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)
                for row in reader:
                    if row and row[0]:
                        image_path = row[0]
                        if len(row) > 1 and row[1]:
                            indices = [idx.strip() for idx in row[1].split(",") if idx.strip()]
                            priority_indices[image_path].update(indices)

        with open(merged_priority_csv, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["image_path", "bbox_indices"])
            for image_path, idx_set in sorted(priority_indices.items()):
                idx_list = sorted(idx_set, key=int)
                writer.writerow([image_path, ",".join(idx_list)])
        print(f"Saved merged priority list to {merged_priority_csv}")

    # 8. Human Review (CVAT)  Copy required files to data_for_review/images and data_for_review/labels
    images_review_dir = os.path.join(data_review_dir, "images")
    labels_review_dir = os.path.join(data_review_dir, "labels")
    os.makedirs(images_review_dir, exist_ok=True)
    os.makedirs(labels_review_dir, exist_ok=True)

    # Use the merged_priority_csv to find which images and labels to copy
    with open(merged_priority_csv, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        for row in reader:
            if row and row[0]:
                image_name = row[0]  # This is the base filename without extension
                # Find image and label file (assume .jpg and .txt; change as needed)
                # If your original images have a fixed extension, change ".jpg" below as appropriate.
                src_img_path = None
                for ext in POSSIBLE_IMG_EXTENSIONS:
                    candidate = os.path.join(input_dir, image_name + ext)
                    if os.path.exists(candidate):
                        src_img_path = candidate
                        break
                if not src_img_path:
                    print(f"Image file for {image_name} not found in supported formats.")
                    continue
                src_label_path = os.path.join(label_dir, image_name + ".txt")
                if not os.path.exists(src_label_path):
                    print(f"Label file for {image_name} not found.")
                    continue

                # Copy to review dirs
                dst_img_path = os.path.join(images_review_dir, os.path.basename(src_img_path))
                dst_label_path = os.path.join(labels_review_dir, os.path.basename(src_label_path))
                shutil.copy2(src_img_path, dst_img_path)
                shutil.copy2(src_label_path, dst_label_path)
    print(f"Copied all review data to {data_review_dir}")

    # 9. Convert to CVAT format (consistent argument passing)
    data2cvat_script = os.path.abspath("./data2cvat.py")
    data2cvat_args = ["python3", data2cvat_script, data_review_dir, "--classes"] + class_list + ["--subset", subset]

    print(f"Running data2cvat.py for review set: {data2cvat_args}")
    subprocess.run(data2cvat_args, check=True)
    print("CVAT package preparation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full review pipeline with arguments.")
    parser.add_argument("--input_and_label_dir", type=str, required=True, help="Input directory containing labeled data")
    parser.add_argument("--class_map_json", type=str, required=True, help="Path to class_map.json")
    parser.add_argument("--diversified_csvs", type=str, nargs="*", default=[], help="List of CSV files with model predictions")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to store outputs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding extraction")
    parser.add_argument("--subset", type=str, default="Train", help="Subset type (e.g., Train or Validation)")

    args = parser.parse_args()

    main_pipeline(
        input_and_label_dir=args.input_and_label_dir,
        class_map_json=args.class_map_json,
        diversified_csvs=args.diversified_csvs,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        subset=args.subset
    )