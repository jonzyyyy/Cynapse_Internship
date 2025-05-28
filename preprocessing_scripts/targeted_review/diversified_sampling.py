import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict
import csv

def load_embeddings_labels_confidences(
    emb_dir: str,
    label_dir: str,
    conf_csv: str
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Loads hybrid embeddings, labels, object IDs, and confidences for all objects.
    """
    embeddings_list = []
    labels_list = []
    object_ids = []
    conf_df = pd.read_csv(conf_csv, index_col=0)  # Assumes first col is object_id, columns: ['confidence', ...]
    conf_lookup = conf_df['confidence'].to_dict()

    emb_files = sorted([f for f in os.listdir(emb_dir) if f.endswith('.npy')])
    for fname in emb_files:
        image_id = os.path.splitext(fname)[0]
        emb_path = os.path.join(emb_dir, fname)
        embs = np.load(emb_path)
        if embs.size == 0 or embs.ndim != 2:
            continue

        label_path = os.path.join(label_dir, f"{image_id}.txt")
        if not os.path.exists(label_path):
            continue
        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) != embs.shape[0]:
            continue

        for idx, emb in enumerate(embs):
            obj_id = f"{image_id}_{idx}"
            confidence = conf_lookup.get(obj_id, None)
            if confidence is None:
                continue
            label = lines[idx].split()[0]
            embeddings_list.append(emb)
            labels_list.append(label)
            object_ids.append(obj_id)

    confidences = np.array([conf_lookup[oid] for oid in object_ids])
    return np.vstack(embeddings_list), np.array(labels_list), object_ids, confidences

def load_committee_predictions(pred_csvs: List[str], object_ids: List[str]) -> np.ndarray:
    """
    Loads committee predictions from multiple CSV files and aligns them by object_id.
    Each CSV should have columns: object_id, pred_label.
    Returns a (num_objects, num_models) array of predictions.
    """
    preds = []
    for csv_path in pred_csvs:
        df = pd.read_csv(csv_path, index_col=0)
        pred_map = df['pred_label'].to_dict()
        preds.append([pred_map.get(oid, None) for oid in object_ids])
    return np.array(preds).T  # shape: (num_objects, num_models)

def main():
    parser = argparse.ArgumentParser(description="Diversified Uncertainty & Committee Sampling")
    parser.add_argument("--embeddings_dir", required=True, help="Directory with .npy hybrid embeddings.")
    parser.add_argument("--label_dir", required=True, help="Directory with YOLO .txt labels.")
    parser.add_argument("--conf_csv", required=True, help="CSV file with columns [object_id, confidence].")
    parser.add_argument("--output_path", required=True, help="Output txt file for selected object_ids.")
    parser.add_argument("--uncertainty_percentile", type=float, default=10.0, help="Bottom X%% confidence to consider.")
    parser.add_argument("--diversity_k", type=int, default=50, help="If you set --diversity_k 50, you’ll get 50 diverse, uncertain samples chosen from the bottom X% of the model’s confidence distribution.")
    parser.add_argument("--committee_csvs", nargs='*', help="List of CSVs with [object_id, pred_label] for committee disagreement.")
    parser.add_argument("--disagreement_only", action="store_true", help="If set, only samples with model disagreement are selected.")
    args = parser.parse_args()

    # 1. Load data
    embs, labels, object_ids, confidences = load_embeddings_labels_confidences(
        args.embeddings_dir, args.label_dir, args.conf_csv
    )

    print(f"Loaded {len(object_ids)} objects with confidences.")

    # 2. Uncertainty Sampling: Get indices of lowest confidence objects
    n_select_uncertain = int(len(confidences) * args.uncertainty_percentile / 100)
    idx_uncertain = np.argsort(confidences)[:n_select_uncertain]

    # 3. Diversity Sampling: cluster the uncertain set and pick representative samples
    if len(idx_uncertain) > args.diversity_k:
        print(f"Clustering {len(idx_uncertain)} uncertain samples into {args.diversity_k} clusters for diversity.")
        uncertain_embs = embs[idx_uncertain]
        kmeans = KMeans(n_clusters=args.diversity_k, random_state=42).fit(uncertain_embs)
        centers = kmeans.cluster_centers_
        selected_indices = []
        for center in centers:
            # Pick the closest embedding in the batch to each cluster center
            dists = np.linalg.norm(uncertain_embs - center, axis=1)
            closest = np.argmin(dists)
            selected_indices.append(idx_uncertain[closest])
        diverse_indices = sorted(set(selected_indices))
    else:
        diverse_indices = list(idx_uncertain)

    selected_obj_ids = set(object_ids[i] for i in diverse_indices)

    # # 4. Committee/Disagreement Sampling
    # if args.committee_csvs:
    #     print("Loading committee predictions for disagreement sampling...")
    #     preds = load_committee_predictions(args.committee_csvs, object_ids)
    #     disagreements = []
    #     for i, row in enumerate(preds):
    #         if len(set(row)) > 1:
    #             disagreements.append(i)
    #     if args.disagreement_only:
    #         selected_obj_ids = set(object_ids[i] for i in disagreements)
    #     else:
    #         selected_obj_ids |= set(object_ids[i] for i in disagreements)

    # print(f"Selected {len(selected_obj_ids)} samples for review.")


    # 5. Output for review (CSV in grouped format)
    # Map image_path to list of bbox indices
    image_to_indices = defaultdict(list)
    for obj_id in sorted(selected_obj_ids):
        # obj_id format: "image_id_index"
        if "_" not in obj_id:
            continue  # skip malformed ids
        image_path, idx = obj_id.rsplit("_", 1)
        image_to_indices[image_path].append(idx)

    with open(args.output_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "bbox_indices"])
        for image_path, idx_list in image_to_indices.items():
            indices_str = ",".join(sorted(idx_list, key=int))
            writer.writerow([image_path, indices_str])
    print(f"Saved grouped list to {args.output_path} as CSV")


if __name__ == "__main__":
    main()