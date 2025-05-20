import os
import sys
import argparse

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png')

def die(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)

def validate_structure(root, split, batches):
    """Ensure split/batch/images + split/batch/labels exist."""
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        die(f"Missing split directory: {split_dir}")
    if not batches:
        die(f"No batches specified for split '{split}'")
    for batch in batches:
        batch_dir = os.path.join(split_dir, batch)
        if not os.path.isdir(batch_dir):
            die(f"Batch '{batch}' not found under {split_dir}")
        for sub in ('images', 'labels'):
            if not os.path.isdir(os.path.join(batch_dir, sub)):
                die(f"Missing '{sub}/' in {batch_dir}")

def collect_batch_paths(root, split, batch):
    """Return (valid_paths, missing_label_relpaths)."""
    valid, missing = [], []
    imgs_base = os.path.join(root, split, batch, 'images')
    lbls_base = os.path.join(root, split, batch, 'labels')
    for dirpath, _, files in os.walk(imgs_base):
        rel_folder = os.path.relpath(dirpath, imgs_base)
        for fname in sorted(files):
            if not fname.lower().endswith(SUPPORTED_EXTS):
                continue
            img_full = os.path.join(dirpath, fname)
            rel_img = os.path.relpath(img_full, root)
            label_dir = os.path.join(lbls_base, rel_folder) if rel_folder != '.' else lbls_base
            label_file = os.path.join(label_dir, os.path.splitext(fname)[0] + '.txt')
            if os.path.isfile(label_file):
                valid.append(img_full)
            else:
                missing.append(rel_img)
    return valid, missing

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-split .txt lists for a YOLO dataset."
    )
    parser.add_argument('-r', '--root', required=True,
                        help="Path to dataset root")
    parser.add_argument('--train', nargs='+',
                        help="Batches for TRAIN split, or 'all'")
    parser.add_argument('--valid',   nargs='+',
                        help="Batches for VAL split, or 'all'")
    parser.add_argument('--test',  nargs='+',
                        help="Batches for TEST split, or 'all' (optional)")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        die(f"Dataset root not found: {root}")

    # Build split → batch_list mapping
    split_args = {'train': args.train, 'valid': args.valid, 'test': args.test}
    split_batches = {}
    for split, batches in split_args.items():
        if batches is None:
            continue  # user doesn’t want this split
        # handle “all”
        if len(batches) == 1 and batches[0].lower() == 'all':
            candidate = [
                d for d in sorted(os.listdir(os.path.join(root, split)))
                if os.path.isdir(os.path.join(root, split, d))
            ]
            if not candidate:
                die(f"No batch subfolders under {split}")
            batches = candidate
        split_batches[split] = batches

    if not split_batches:
        die("No splits specified. Use --train, --valid, and/or --test.")

    out_base = os.getcwd()
    for split, batches in split_batches.items():
        # 1) validate
        validate_structure(root, split, batches)
        # 2) collect
        all_valid, all_missing = [], []
        for batch in batches:
            v, m = collect_batch_paths(root, split, batch)
            all_valid.extend(v)
            all_missing.extend(m)
        # 3) report missing
        for rel in all_missing:
            print(f"[{split}] Missing label for image: {rel}")
        # 4) write file
        txt_dir = os.path.join(out_base, 'data_txt')
        os.makedirs(txt_dir, exist_ok=True)
        out_path = os.path.join(txt_dir, f"{split}.txt")
        with open(out_path, 'w') as f:
            written = 0
            skip_set = set(all_missing)
            for img in all_valid:
                # write the absolute or rel paths you prefer:
                if os.path.relpath(img, root) in skip_set:
                    continue
                f.write(img + '\n')
                written += 1
        print(f"Wrote {written} entries to {out_path} (skipped {len(all_missing)})")

if __name__ == '__main__':
    main()