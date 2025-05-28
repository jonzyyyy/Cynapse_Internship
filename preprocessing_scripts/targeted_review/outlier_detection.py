
import argparse
import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Tuple, Dict
import csv

def load_embeddings_and_labels(
    emb_dir: str,
    label_dir: str,
    class_map: Dict[str, str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads hybrid embeddings from .npy files and corresponding YOLO labels.

    Args:
        emb_dir: Directory containing .npy embedding files (one per image).
        label_dir: Directory containing YOLO .txt label files (one per image).
        class_map: Mapping from class ID (string) to class name.

    Returns:
        embeddings: N×D array of embeddings.
        labels:     N-length array of class names.
        object_ids: List of object identifiers, e.g., "imageID_0", "imageID_1", …
    """
    embeddings_list = []
    labels_list = []
    object_ids = []

    emb_files = sorted([f for f in os.listdir(emb_dir) if f.endswith('.npy')])
    for fname in emb_files:
        image_id = os.path.splitext(fname)[0]
        emb_path = os.path.join(emb_dir, fname)
        embs = np.load(emb_path)  # shape (num_boxes, emb_dim)

        # Skip empty embedding arrays
        if embs.size == 0 or embs.ndim != 2:
            print(f"Skipping {fname}: empty or invalid embedding array.")
            continue

        label_path = os.path.join(label_dir, f"{image_id}.txt")
        if not os.path.exists(label_path):
            print(f"Warning: No label file for {image_id}, skipping.")
            continue

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) != embs.shape[0]:
            print(f"Warning: {image_id} has {len(lines)} labels vs {embs.shape[0]} embeddings. Skipping {emb_path}.")
            continue

        for idx, emb in enumerate(embs):
            parts = lines[idx].split()
            class_id = parts[0]
            class_name = class_map.get(str(class_id), str(class_id))
            embeddings_list.append(emb)
            labels_list.append(class_name)
            object_ids.append(f"{image_id}_{idx}")

    if not embeddings_list:
        raise RuntimeError("No valid embeddings/labels found!")

    embeddings = np.vstack(embeddings_list)
    labels = np.array(labels_list)
    return embeddings, labels, object_ids

def main():
    parser = argparse.ArgumentParser(
        description="Advanced outlier/noise detection on hybrid embeddings."
    )
    parser.add_argument(
        "--embeddings_dir", required=True,
        help="Directory with .npy hybrid embeddings (one file per image)."
    )
    parser.add_argument(
        "--label_dir", required=True,
        help="Directory with YOLO .txt labels (matching .npy files)."
    )
    parser.add_argument(
        "--class_map", required=True,
        help="JSON file mapping class IDs to human-readable names."
    )
    parser.add_argument(
        "--n_clusters", type=int, default=10,
        help="Number of clusters for KMeans (optional, for analysis)."
    )
    parser.add_argument(
        "--lof_neighbors", type=int, default=20,
        help="Number of neighbors for Local Outlier Factor."
    )
    parser.add_argument(
        "--distance_percentile", type=float, default=95.0,
        help="Percentile threshold for distance-based outlier flagging."
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Path to save flagged object IDs (one per line)."
    )
    parser.add_argument(
        "--visualisation", type=bool, default=False,
        help="Whether to run KMeans for visualisation purposes (optional)."
    )
    

    args = parser.parse_args()

    # Load class map
    with open(args.class_map) as f:
        class_map = json.load(f)

    # Load embeddings, labels, and object IDs
    print("Loading all embeddings and labels...")
    embeddings, labels, object_ids = load_embeddings_and_labels(
        args.embeddings_dir,
        args.label_dir,
        class_map
    )
    print(f"Loaded {embeddings.shape[0]} object embeddings.")

    # 1) Distance-to-centroid
    print("Computing distance-to-class-centroid outliers...")
    class_centroids = {}
    for cls in np.unique(labels):
        class_centroids[cls] = embeddings[labels == cls].mean(axis=0)
    distances = np.array([
        np.linalg.norm(embeddings[i] - class_centroids[labels[i]])
        for i in range(len(embeddings))
    ])
    dist_thresh = np.percentile(distances, args.distance_percentile)
    dist_flags = set(np.where(distances > dist_thresh)[0])

    # 2) Local Outlier Factor
    print(f"Running Local Outlier Factor with {args.lof_neighbors} neighbors...")
    lof = LocalOutlierFactor(n_neighbors=args.lof_neighbors)
    lof_pred = lof.fit_predict(embeddings)  # -1 => outlier
    lof_flags = set(np.where(lof_pred == -1)[0])

    # 3) (Optional) KMeans clustering - Can be used for analysis
    if args.visualisation:
        print(f"Running KMeans with n_clusters={args.n_clusters}...")
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(embeddings)


    # Combine and map back to object IDs
    flagged_indices = sorted(dist_flags | lof_flags)
    flagged_objects = [object_ids[i] for i in flagged_indices]

    # Collect flagged objects by image
    flagged_dict = {}
    for obj_id in flagged_objects:
        # obj_id is "imageID_index"
        if "_" not in obj_id:
            continue
        image_id, idx = obj_id.rsplit("_", 1)
        flagged_dict.setdefault(image_id, []).append(int(idx))

    # Write as CSV: image_path, bbox_indices
    image_dir = args.embeddings_dir.replace("embeddings", "images")
    out_csv = args.output_path.replace(".txt", ".csv")

    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "bbox_indices"])
        for image_id, indices in flagged_dict.items():
            # Try multiple extensions for the image file
            image_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = os.path.join(image_dir, image_id + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if image_path is None:
                image_path = image_id  # fallback, or raise/log
            # Sort indices and convert to comma-separated string
            index_str = ",".join(map(str, sorted(indices)))
            writer.writerow([image_path, index_str])

    print(f"CSV with flagged image paths and bbox indices written to {out_csv}")

if __name__ == "__main__":
    main()