#!/usr/local/bin/python3
"""
Fiducial Binary Dataset Splitter (v2 — Balanced Subset)
=========================================================
Converts the 4-class fiducial dataset into 4 binary datasets, each with
a balanced subset targeting approximately:

  - 50% positive (target corner type)
  - 35% hard negatives (other corner types)
  - 15% background negatives (no corner at all)

Total: ~4,500 training images, ~1,125 validation images per model.

Images are COPIED (not symlinked) for portability.

Usage:
    /usr/local/bin/python3 split_binary_datasets.py --seed 42
    /usr/local/bin/python3 split_binary_datasets.py --target-train 4500 --target-val 1125

Author: Photo Pose Detector Project
"""

import os
import sys
import random
import shutil
from pathlib import Path
from collections import defaultdict

# Class ID mapping (from the original 4-class fiducial dataset)
CORNER_CLASS_IDS = {
    'ul': 0,
    'ur': 1,
    'll': 2,
    'lr': 3,
}

# Target ratios
POSITIVE_RATIO = 0.50
HARD_NEG_RATIO = 0.35
BG_NEG_RATIO = 0.15  # = 1.0 - POSITIVE_RATIO - HARD_NEG_RATIO

DEFAULT_TARGET_TRAIN = 4500
DEFAULT_TARGET_VAL = 1125

# Source data directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATA = PROJECT_ROOT / "data_fiducial"


def scan_source_dataset(split='train'):
    """Scan the source dataset and classify each image by its label type.

    Returns dict: {class_id: [list of (img_path, label_path)],
                   'background': [list of (img_path, None)]}
    """
    img_dir = SOURCE_DATA / "images" / split
    lbl_dir = SOURCE_DATA / "labels" / split
    removed_img_dir = SOURCE_DATA / "REMOVED" / "images" / split
    removed_lbl_dir = SOURCE_DATA / "REMOVED" / "labels" / split

    by_class = defaultdict(list)  # class_id -> [(img_path, label_path)]
    backgrounds = []               # [(img_path, None)]

    # Process active dataset
    active_imgs = sorted(img_dir.glob("*.jpg"))
    for img_file in active_imgs:
        stem = img_file.stem
        label_file = lbl_dir / f"{stem}.txt"

        if label_file.exists():
            # Read the class ID from the label
            with open(label_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    by_class[class_id].append((img_file, label_file))
                else:
                    backgrounds.append((img_file, None))
        else:
            backgrounds.append((img_file, None))

    # Process REMOVED dataset (all labeled)
    if removed_img_dir.exists():
        removed_imgs = sorted(removed_img_dir.glob("*.jpg"))
        for img_file in removed_imgs:
            stem = img_file.stem
            label_file = removed_lbl_dir / f"{stem}.txt"

            if label_file.exists():
                with open(label_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        class_id = int(first_line.split()[0])
                        by_class[class_id].append((img_file, label_file))

    return by_class, backgrounds


def create_binary_dataset(corner_name, class_id, seed=42,
                          target_train=DEFAULT_TARGET_TRAIN,
                          target_val=DEFAULT_TARGET_VAL):
    """Create a balanced binary dataset for one corner type."""

    rng = random.Random(seed)

    dataset_dir = PROJECT_ROOT / "data_fiducial_binary" / corner_name
    dirs = {}
    for split in ['train', 'val']:
        dirs[f'img_{split}'] = dataset_dir / "images" / split
        dirs[f'lbl_{split}'] = dataset_dir / "labels" / split

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    target = {'train': target_train, 'val': target_val}
    stats = {}

    for split in ['train', 'val']:
        by_class, backgrounds = scan_source_dataset(split)

        # Positive samples: target class
        positives = list(by_class[class_id])

        # Hard negatives: other corner classes
        hard_negatives = []
        for other_id in CORNER_CLASS_IDS.values():
            if other_id != class_id:
                hard_negatives.extend(by_class[other_id])

        # Background negatives: no label
        bg_negatives = list(backgrounds)

        # Shuffle
        rng.shuffle(positives)
        rng.shuffle(hard_negatives)
        rng.shuffle(bg_negatives)

        # Calculate target counts
        n_total = target[split]
        n_pos = int(n_total * POSITIVE_RATIO)
        n_hard = int(n_total * HARD_NEG_RATIO)
        n_bg = n_total - n_pos - n_hard

        # Cap at available data
        n_pos = min(n_pos, len(positives))
        n_hard = min(n_hard, len(hard_negatives))
        n_bg = min(n_bg, len(bg_negatives))

        # If any category is short, redistribute the remainder
        remaining = n_total - n_pos - n_hard - n_bg
        if remaining > 0:
            # Try to fill from categories that have surplus
            for cat_name, cat_list, cat_count in [
                ('positive', positives, n_pos),
                ('hard_neg', hard_negatives, n_hard),
                ('bg_neg', bg_negatives, n_bg),
            ]:
                available = len(cat_list) - cat_count
                fill = min(remaining, available)
                if cat_name == 'positive':
                    n_pos += fill
                elif cat_name == 'hard_neg':
                    n_hard += fill
                else:
                    n_bg += fill
                remaining -= fill
                if remaining <= 0:
                    break

        actual_total = n_pos + n_hard + n_bg

        selected_pos = positives[:n_pos]
        selected_hard = hard_negatives[:n_hard]
        selected_bg = bg_negatives[:n_bg]

        # Write positive samples (class 0)
        for idx, (img_path, label_path) in enumerate(selected_pos):
            prefix = f"{split}_pos_{idx:04d}"

            # Copy image
            shutil.copy2(str(img_path), str(dirs[f'img_{split}'] / f"{prefix}.jpg"))

            # Rewrite label: replace class_id with 0
            with open(label_path, 'r') as f:
                original = f.read().strip()
            parts = original.split()
            parts[0] = '0'
            with open(dirs[f'lbl_{split}'] / f"{prefix}.txt", 'w') as f:
                f.write(' '.join(parts) + '\n')

        # Write hard negative samples (other corner types → empty label)
        for idx, (img_path, label_path) in enumerate(selected_hard):
            prefix = f"{split}_hard_{idx:04d}"

            # Copy image
            shutil.copy2(str(img_path), str(dirs[f'img_{split}'] / f"{prefix}.jpg"))

            # Empty label file — the model should NOT detect anything
            (dirs[f'lbl_{split}'] / f"{prefix}.txt").touch()

        # Write background negative samples (no corner → empty label)
        for idx, (img_path, _) in enumerate(selected_bg):
            prefix = f"{split}_bg_{idx:04d}"

            # Copy image
            shutil.copy2(str(img_path), str(dirs[f'img_{split}'] / f"{prefix}.jpg"))

            # Empty label file
            (dirs[f'lbl_{split}'] / f"{prefix}.txt").touch()

        stats[split] = {
            'positive': n_pos,
            'hard_neg': n_hard,
            'bg_neg': n_bg,
            'total': actual_total,
            'available_pos': len(positives),
            'available_hard': len(hard_negatives),
            'available_bg': len(bg_negatives),
        }

    return stats


def create_yaml_files():
    """Create dataset YAML config files for each binary model."""
    for corner_name in CORNER_CLASS_IDS:
        dataset_dir = PROJECT_ROOT / "data_fiducial_binary" / corner_name
        yaml_path = dataset_dir / f"dataset_{corner_name}.yaml"

        yaml_content = f"""# YOLO Binary Detection Dataset — {corner_name.upper()} Corner
# =========================================================
# Single-class detection: "corner" (class 0) vs. background
# The model detects whether this specific corner type is present
# in a 640x640 crop centered near a potential corner location.
#
# Composition:
#   - 50% positive (target corner type)
#   - 35% hard negatives (other corner types)
#   - 15% background (no corner at all)
#
# NO flip augmentation! Flipping changes corner orientation.

path: {dataset_dir}
train: images/train
val: images/val

nc: 1
names:
  0: corner
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"  Created {yaml_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Create balanced binary fiducial datasets')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--target-train', type=int, default=DEFAULT_TARGET_TRAIN,
                        help=f'Target number of training images (default: {DEFAULT_TARGET_TRAIN})')
    parser.add_argument('--target-val', type=int, default=DEFAULT_TARGET_VAL,
                        help=f'Target number of validation images (default: {DEFAULT_TARGET_VAL})')
    args = parser.parse_args()

    print("=" * 70)
    print("FIDUCIAL BINARY DATASET SPLITTER v2 (Balanced Subset)")
    print("=" * 70)
    print(f"Target ratios: {POSITIVE_RATIO:.0%} positive, {HARD_NEG_RATIO:.0%} hard neg, {BG_NEG_RATIO:.0%} background")
    print(f"Target per model: {args.target_train} train, {args.target_val} val")
    print(f"Source: {SOURCE_DATA}")
    print(f"Output: {PROJECT_ROOT / 'data_fiducial_binary'}")
    print(f"Seed: {args.seed}")
    print()

    # Check source data
    if not SOURCE_DATA.exists():
        print(f"ERROR: Source data not found at {SOURCE_DATA}")
        sys.exit(1)

    # First, scan to show available data
    class_names = {v: k for k, v in CORNER_CLASS_IDS.items()}  # {0: 'ul', 1: 'ur', ...}
    print("Scanning available data...")
    all_stats = {}
    for split in ['train', 'val']:
        by_class, backgrounds = scan_source_dataset(split)
        total_labeled = sum(len(v) for v in by_class.values())
        class_summary = ', '.join(f'{class_names.get(k, k)}={len(v)}' for k, v in sorted(by_class.items()))
        print(f"  {split}: {total_labeled} labeled ({class_summary}), "
              f"{len(backgrounds)} background, {total_labeled + len(backgrounds)} total")
    print()

    # Create binary datasets
    total_stats = {}
    for corner_name, class_id in CORNER_CLASS_IDS.items():
        print(f"Creating dataset for {corner_name.upper()} (class {class_id})...")
        stats = create_binary_dataset(
            corner_name, class_id,
            seed=args.seed,
            target_train=args.target_train,
            target_val=args.target_val,
        )
        total_stats[corner_name] = stats

        for split in ['train', 'val']:
            s = stats[split]
            pos_pct = s['positive'] / s['total'] * 100 if s['total'] > 0 else 0
            hard_pct = s['hard_neg'] / s['total'] * 100 if s['total'] > 0 else 0
            bg_pct = s['bg_neg'] / s['total'] * 100 if s['total'] > 0 else 0
            print(f"  {split}: {s['total']} images "
                  f"({s['positive']} pos [{pos_pct:.0f}%], "
                  f"{s['hard_neg']} hard [{hard_pct:.0f}%], "
                  f"{s['bg_neg']} bg [{bg_pct:.0f}%]) "
                  f"[of {s['available_pos']}+{s['available_hard']}+{s['available_bg']} available]")
        print()

    # Create YAML config files
    print("Creating YAML config files...")
    create_yaml_files()

    # Compute disk usage
    total_size = 0
    for corner_name in CORNER_CLASS_IDS:
        dataset_dir = PROJECT_ROOT / "data_fiducial_binary" / corner_name
        for f in dataset_dir.rglob("*.jpg"):
            total_size += f.stat().st_size

    print()
    print("=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)
    print(f"Total disk usage: {total_size / 1e9:.1f} GB")
    print()

    # Summary table
    print(f"{'Corner':<6} {'Train':>8} {'Val':>7} {'Train%':>8} {'Val%':>7}")
    print("-" * 40)
    for corner_name in CORNER_CLASS_IDS:
        s = total_stats[corner_name]
        print(f"{corner_name.upper():<6} {s['train']['total']:>8} {s['val']['total']:>7} "
              f"  {s['train']['positive']/s['train']['total']*100:.0f}P/{s['train']['hard_neg']/s['train']['total']*100:.0f}H/{s['train']['bg_neg']/s['train']['total']*100:.0f}B "
              f"  {s['val']['positive']/s['val']['total']*100:.0f}P/{s['val']['hard_neg']/s['val']['total']*100:.0f}H/{s['val']['bg_neg']/s['val']['total']*100:.0f}B")

    print()
    print("Next step: Train 4 binary models using train_fiducial_binary.py")


if __name__ == "__main__":
    main()