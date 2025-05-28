import argparse
import json
from typing import List, Dict
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import logging  
from tqdm import tqdm
import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_TQDM"] = "1"

from contextlib import contextmanager
import tqdm as _tqdm_module

@contextmanager
def silence_inner_tqdm():
    """
    Disable any tqdm bars started inside this context
    (e.g. by transformers / sentence-transformers) while
    leaving the outer progress bar intact.
    """
    orig_tqdm = _tqdm_module.tqdm
    _tqdm_module.tqdm = lambda *a, **k: a[0] if a else None
    try:
        yield
    finally:
        _tqdm_module.tqdm = orig_tqdm

# Add basic logging config at the top
logging.basicConfig(
    level=logging.INFO,  # Default level; set to DEBUG for more verbosity
    format='[%(levelname)s] %(message)s'
)

def yolo_to_bbox(xc, yc, w, h, img_w, img_h):
    left = int((xc - w/2) * img_w)
    upper = int((yc - h/2) * img_h)
    right = int((xc + w/2) * img_w)
    lower = int((yc + h/2) * img_h)
    return left, upper, right, lower

def extract_hybrid_embeddings_from_yolo(
    image_path: str,
    label_path: str,
    class_id_to_name: Dict,
    vision_model,
    processor,
    text_model
) -> List[np.ndarray]:
    image = Image.open(image_path)
    img_w, img_h = image.size
    hybrid_embeddings = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = parts[0]
            xc, yc, w, h = map(float, parts[1:])
            bbox = yolo_to_bbox(xc, yc, w, h, img_w, img_h)
            class_name = class_id_to_name.get(str(class_id), str(class_id))
            crop = image.crop(bbox)
            with silence_inner_tqdm():
                inputs = processor(images=crop, return_tensors="pt")
                with torch.no_grad():
                    visual_emb = vision_model.get_image_features(**inputs).squeeze().cpu().numpy()
            with silence_inner_tqdm():
                text_emb = text_model.encode(class_name)
            hybrid_embeddings.append(np.concatenate([visual_emb, text_emb]))
    return hybrid_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch extract hybrid embeddings from YOLO-labelled images.")
    parser.add_argument("--image_dir", required=True, help="Directory containing image files.")
    parser.add_argument("--label_dir", required=True, help="Directory containing YOLO .txt label files.")
    parser.add_argument("--class_map", required=True, help="JSON mapping class IDs to names.")
    parser.add_argument("--dest_dir", required=True, help="Directory to save per-image .npy embeddings.")
    parser.add_argument("--vision_model_name", default="openai/clip-vit-base-patch16", help="HuggingFace CLIP vision model.")
    parser.add_argument("--processor_name", default="openai/clip-vit-base-patch16", help="HuggingFace CLIP processor.")
    parser.add_argument("--text_model_name", default="all-MiniLM-L6-v2", help="SentenceTransformer model for text embeddings.")
    parser.add_argument("--batch_size", type=int, default=8, help="How many images to process at once (RAM control).")

    args = parser.parse_args()

    with open(args.class_map) as f:
        class_id_to_name = json.load(f)

    os.makedirs(args.dest_dir, exist_ok=True)

    vision_model = CLIPModel.from_pretrained(args.vision_model_name)
    processor = CLIPProcessor.from_pretrained(args.processor_name)
    text_model = SentenceTransformer(args.text_model_name)

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(valid_exts)])

    already_done = set(os.path.splitext(f)[0] for f in os.listdir(args.dest_dir) if f.endswith('.npy'))

    num_to_process = sum(
        1 for f in image_files if os.path.splitext(f)[0] not in already_done
    )
    progress_bar = tqdm(total=num_to_process, desc="Images processed", unit="img")

    batch = []
    batch_num = 0
    for fname in image_files:
        base = os.path.splitext(fname)[0]
        if base in already_done:
            continue
        image_path = os.path.join(args.image_dir, fname)
        label_path = os.path.join(args.label_dir, base + ".txt")
        if not os.path.exists(label_path):
            logging.info(f"Skipping {fname}: label file missing.")
            continue
        batch.append((fname, image_path, label_path))
        # When batch full or last file
        if len(batch) == args.batch_size or fname == image_files[-1]:
            batch_num += 1
            batch_fnames = [b[0] for b in batch]
            logging.info(f"Processing batch {batch_num} ({len(batch)} images): {batch_fnames}")
            for fname_, image_path_, label_path_ in batch:
                try:
                    embeddings = extract_hybrid_embeddings_from_yolo(
                        image_path_, label_path_, class_id_to_name,
                        vision_model, processor, text_model
                    )
                    out_file = os.path.join(args.dest_dir, f"{os.path.splitext(fname_)[0]}.npy")
                    emb_array = np.stack(embeddings) if embeddings else np.zeros((0,))
                    np.save(out_file, emb_array)
                    logging.debug(f"Saved {emb_array.shape[0]} embeddings to {out_file}")
                except Exception as e:
                    logging.info(f"Failed processing {fname_}: {e}")
                progress_bar.update(1)  # Update for every image processed in batch
            batch = []

    progress_bar.close()
    logging.info(f"All batches processed. {num_to_process} images in total.")