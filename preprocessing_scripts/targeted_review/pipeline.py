import os
import subprocess

def main_pipeline(
    input_dir,
    label_dir,
    class_map_json,
    confidence_csv,
    diversified_csvs,
    output_folder,
    batch_size=8
):
    # Ensure output folder and embeddings subdir exist
    os.makedirs(output_folder, exist_ok=True)
    embeddings_dir = os.path.join(output_folder, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    # Standard output file names
    priority_list_txt = os.path.join(output_folder, "priority_review.txt")
    outlier_output_txt = os.path.join(output_folder, "outliers.txt")

    # 1. Data Ingestion: Assume images are placed in input_dir

    # 2. Auto-Labelling: Run GroundingDINO or equivalent externally to create YOLO .txts in label_dir
    auto_labeled_image_path = "/mnt/nas/TAmob/old_data/final_extracted_frames/11_05_2025 19_59_59 (UTC+03_00)_processed_fr20_10_197_21_24/_labeled/train/images"
    auto_labeled_label_path = "/mnt/nas/TAmob/old_data/final_extracted_frames/11_05_2025 19_59_59 (UTC+03_00)_processed_fr20_10_197_21_24/_labeled/train/labels"

    # 3. Hybrid Embedding Extraction - Save to output_folder/embeddings
    subprocess.run([
        "python3", "hybrid_embedding_extractor.py",
        "--image_dir", auto_labeled_image_path,
        "--label_dir", auto_labeled_label_path,
        "--class_map", class_map_json,
        "--dest_dir", embeddings_dir,
        "--batch_size", str(batch_size)
    ], check=True)

    # 4. Outlier/Noise Detection - Use embeddings_dir in output_folder
    subprocess.run([
        "python3", "outlier_detection.py",
        "--embeddings_dir", embeddings_dir,
        "--label_dir", auto_labeled_label_path,
        "--class_map", class_map_json,
        "--output_path", outlier_output_txt
    ], check=True)

    # 5 & 6. Committee Disagreement & Uncertainty/Diversity Sampling - Use embeddings_dir in output_folder
    diversified_args = [
        "python3", "diversified_sampling.py",
        "--embeddings_dir", embeddings_dir,
        "--label_dir", auto_labeled_label_path,
        "--conf_csv", confidence_csv,
        "--output_path", priority_list_txt,
        "--uncertainty_percentile", "10.0",
        "--diversity_k", "50"
    ]
    if diversified_csvs:
        diversified_args += ["--committee_csvs"] + diversified_csvs
    subprocess.run(diversified_args, check=True)

    # # 7. Construct Combined Priority List for Review
    # priority_set = set()
    # for fpath in [outlier_output_txt, priority_list_txt]:
    #     with open(fpath, 'r') as f:
    #         priority_set.update(line.strip() for line in f)
    # merged_priority_txt = os.path.join(os.path.dirname(priority_list_txt), "merged_priority_review.txt")
    # with open(merged_priority_txt, "w") as f:
    #     for obj_id in sorted(priority_set):
    #         f.write(obj_id + "\n")
    # print(f"Saved merged priority list to {merged_priority_txt}")

    # # 8. Human Review (CVAT) — plug in your CVAT integration here

    # # 9. Update Database (after review export) — plug in your integration here

    # # 10. Retrain Models (periodic) — plug in your retraining logic

    # # 11. Maintenance Clean (periodic) — plug in your maintenance logic


if __name__ == "__main__":
    main_pipeline(
        input_dir="/mnt/nas/TAmob/",
        label_dir="/mnt/nas/TAmob/",
        class_map_json="/mnt/nas/TAmob/preprocessing_scripts/targeted_review/class_map.json",
        confidence_csv="/mnt/nas/TAmob/old_data/final_extracted_frames/11_05_2025 19_59_59 (UTC+03_00)_processed_fr20_10_197_21_24/object_confidences.csv",
        diversified_csvs=["/mnt/nas/TAmob/modelA_preds.csv", "/mnt/nas/TAmob/modelB_preds.csv"],
        output_folder="/mnt/nas/TAmob/old_data/final_extracted_frames/11_05_2025 19_59_59 (UTC+03_00)_processed_fr20_10_197_21_24/pipeline_output",
        batch_size=8
    )