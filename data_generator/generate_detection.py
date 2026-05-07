#!/usr/bin/env python3
"""
Photo Pose Detector — Detection Data Generator
================================================

Generates synthetic training images for the **detection model only**.
Each image contains 1–4 photographs placed on a random background,
with axis-aligned bounding box labels in standard YOLO detection format.

This is a simplified version of the original `generate.py` with all
pose/keypoint code removed.  The detection dataset is self-contained
(no symlinks needed).

Output format per label line:
    class_id x_center y_center width height

Usage:
    python generate_detection.py --mode examples --count 10 --source ./images
    python generate_detection.py --mode batch --train-count 4000 --val-count 1000 --source ./images --output ../data_detection
"""

import cv2
import numpy as np
from pathlib import Path
import random
import sys
import time
import math
import argparse

from generate_common import (
    CANVAS_SIZE, PHOTO_SIZE_MIN, PHOTO_SIZE_MAX, ROTATION_RANGE, BOUND_MARGIN,
    get_rotated_polygon, compute_rotated_bbox, rotate_photo,
    random_base_background, apply_texture_overlay, fast_glare,
    apply_photo_shadow, composite_photo_at_center,
    apply_perspective_safe, transform_corners,
    load_and_prepare_photo, create_debug_image,
    check_bounds, check_overlaps,
)

# Configuration
NUM_PHOTOS_MIN = 1
NUM_PHOTOS_MAX = 4
MAX_PACK_ATTEMPTS = 50


# =============================================================================
# PLACEMENT ENGINE (multi-photo)
# =============================================================================

def _shrink_to_fit(width, height, cx, cy, rotation, canvas_size,
                   margin=BOUND_MARGIN, min_side=150):
    """Iteratively shrink a photo until its rotated corners fit within bounds."""
    w, h = width, height
    for _ in range(30):
        corners = get_rotated_polygon(w, h, cx, cy, rotation)
        x_min, x_max = corners[:, 0].min(), corners[:, 0].max()
        y_min, y_max = corners[:, 1].min(), corners[:, 1].max()

        if (x_min >= margin and x_max <= canvas_size - margin and
                y_min >= margin and y_max <= canvas_size - margin):
            return w, h

        x_overflow = max(0, margin - x_min, x_max - (canvas_size - margin))
        y_overflow = max(0, margin - y_min, y_max - (canvas_size - margin))
        max_overflow = max(x_overflow, y_overflow)
        bbox_diag = max(x_max - x_min, y_max - y_min, 1)
        shrink = max(1.0 - (max_overflow / bbox_diag) * 1.1, 0.7)

        new_w = max(int(w * shrink), min_side)
        new_h = max(int(h * shrink), min_side)

        if new_w == w and new_h == h:
            return w, h
        w, h = new_w, new_h

    return w, h


def _generate_placements(num_photos, canvas_size):
    """Generate candidate placements for num_photos photos."""
    placements = []

    if num_photos == 1:
        rot_range = ROTATION_RANGE
    else:
        rot_range = 5

    photo_min = PHOTO_SIZE_MIN

    if num_photos == 1:
        rotation = random.uniform(-rot_range, rot_range)
        cx = canvas_size / 2 + random.uniform(-30, 30)
        cy = canvas_size / 2 + random.uniform(-30, 30)
        size = random.randint(int(canvas_size * 0.55), int(canvas_size * 0.85))
        aspect = random.uniform(0.8, 1.2)
        width, height = size, int(size * aspect)
        width, height = _shrink_to_fit(width, height, cx, cy, rotation,
                                        canvas_size, BOUND_MARGIN, photo_min)
        placements.append({
            'width': width, 'height': height,
            'center_x': cx, 'center_y': cy,
            'rotation': rotation,
        })

    elif num_photos == 2:
        rotation1 = random.uniform(-rot_range, rot_range)
        rotation2 = random.uniform(-rot_range, rot_range)

        if random.random() < 0.5:
            cx1 = canvas_size * 0.27 + random.uniform(-8, 8)
            cy1 = canvas_size * 0.50 + random.uniform(-8, 8)
            cx2 = canvas_size * 0.73 + random.uniform(-8, 8)
            cy2 = canvas_size * 0.50 + random.uniform(-8, 8)
            sep = abs(cx2 - cx1)
            layout = 'horizontal'
        else:
            cx1 = canvas_size * 0.50 + random.uniform(-8, 8)
            cy1 = canvas_size * 0.27 + random.uniform(-8, 8)
            cx2 = canvas_size * 0.50 + random.uniform(-8, 8)
            cy2 = canvas_size * 0.73 + random.uniform(-8, 8)
            sep = abs(cy2 - cy1)
            layout = 'vertical'

        expansion = abs(math.cos(math.radians(rot_range))) + abs(math.sin(math.radians(rot_range)))
        max_from_sep = int(sep / expansion)

        for cx, cy, rot in [(cx1, cy1, rotation1), (cx2, cy2, rotation2)]:
            aspect = random.uniform(0.9, 1.05)
            dim_max = max(max_from_sep, PHOTO_SIZE_MIN)
            dim_min = max(PHOTO_SIZE_MIN, int(dim_max * 0.9))
            if dim_min > dim_max:
                dim_min = dim_max
            if layout == 'horizontal':
                width = random.randint(dim_min, dim_max)
                height = int(width * aspect)
            else:
                height = random.randint(dim_min, dim_max)
                width = int(height * aspect)
            width, height = _shrink_to_fit(width, height, cx, cy, rot,
                                            canvas_size, BOUND_MARGIN, photo_min)
            placements.append({
                'width': width, 'height': height,
                'center_x': cx, 'center_y': cy,
                'rotation': rot,
            })

    else:  # 3 or 4 photos — grid layout
        usable = canvas_size - 2 * BOUND_MARGIN
        cols, rows = 2, 2
        cell_w = usable / cols
        cell_h = usable / rows

        positions = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(positions)
        positions = positions[:num_photos]

        for row, col in positions:
            rotation = random.uniform(-rot_range, rot_range)
            cx = BOUND_MARGIN + (col + 0.5) * cell_w + random.uniform(-cell_w * 0.03, cell_w * 0.03)
            cy = BOUND_MARGIN + (row + 0.5) * cell_h + random.uniform(-cell_h * 0.03, cell_h * 0.03)

            size = random.randint(int(min(cell_w, cell_h) * 0.88), int(min(cell_w, cell_h) * 0.98))
            aspect = random.uniform(0.88, 1.05)
            width, height = size, int(size * aspect)
            width, height = _shrink_to_fit(width, height, cx, cy, rotation,
                                            canvas_size, BOUND_MARGIN, photo_min)
            placements.append({
                'width': width, 'height': height,
                'center_x': cx, 'center_y': cy,
                'rotation': rotation,
            })

    return placements


def pack_photos_validated(canvas_size):
    """Pack 1–4 photos with no overlaps and all corners in bounds."""
    for _ in range(MAX_PACK_ATTEMPTS):
        target_count = random.randint(NUM_PHOTOS_MIN, NUM_PHOTOS_MAX)
        placements = _generate_placements(target_count, canvas_size)
        if check_bounds(placements, canvas_size) and check_overlaps(placements):
            return placements

    # Fallback: single centered photo
    size = random.randint(PHOTO_SIZE_MIN, min(PHOTO_SIZE_MAX, int(canvas_size * 0.85)))
    aspect = random.uniform(0.8, 1.2)
    w, h = size, int(size * aspect)
    cx, cy = canvas_size / 2, canvas_size / 2
    rot = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
    w, h = _shrink_to_fit(w, h, cx, cy, rot, canvas_size, BOUND_MARGIN, PHOTO_SIZE_MIN)
    return [{'width': w, 'height': h, 'center_x': cx, 'center_y': cy, 'rotation': rot}]


# =============================================================================
# IMAGE GENERATION
# =============================================================================

def generate_image(source_dir):
    """Generate a single composite image with detection labels.

    Returns:
        image (BGR numpy array)
        det_labels (list of str): YOLO detection label lines
        corners_list (list of (4,2) arrays): for debug visualization
    """
    # Create textured background
    canvas = random_base_background(CANVAS_SIZE, CANVAS_SIZE)
    canvas = apply_texture_overlay(canvas)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
    canvas[:, :, 3] = 255

    # Place photos
    placements = pack_photos_validated(CANVAS_SIZE)
    photos_data = []

    for placement in placements:
        photo, orig_w, orig_h, new_w, new_h = load_and_prepare_photo(
            source_dir, placement['width'], placement['height'])

        photo = rotate_photo(photo, placement['rotation'])

        # Shadow
        shadow_offset = random.randint(2, 5)
        angle = random.uniform(0, 2 * math.pi)
        offset_x = int(shadow_offset * math.cos(angle))
        offset_y = int(shadow_offset * math.sin(angle))
        canvas = apply_photo_shadow(
            canvas, photo, placement['center_x'], placement['center_y'],
            offset_x, offset_y,
            random.uniform(1.5, 3.0), random.uniform(0.15, 0.35),
            new_w, new_h, placement['rotation'],
        )

        canvas = composite_photo_at_center(
            canvas, photo, placement['center_x'], placement['center_y'])

        corners = get_rotated_polygon(
            new_w, new_h, placement['center_x'], placement['center_y'],
            placement['rotation'])
        photos_data.append({'corners': corners})

    # Perspective warp
    corners_list = [p['corners'] for p in photos_data]
    canvas, persp_M, _ = apply_perspective_safe(canvas, corners_list)

    # Transform corners and generate labels
    final_corners_list = []
    det_labels = []

    for photo in photos_data:
        warped = transform_corners(photo['corners'], persp_M)
        final_corners_list.append(warped)

        min_x = warped[:, 0].min()
        max_x = warped[:, 0].max()
        min_y = warped[:, 1].min()
        max_y = warped[:, 1].max()

        x_center = ((min_x + max_x) / 2) / CANVAS_SIZE
        y_center = ((min_y + max_y) / 2) / CANVAS_SIZE
        width = (max_x - min_x) / CANVAS_SIZE
        height = (max_y - min_y) / CANVAS_SIZE

        det_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return canvas, det_labels, final_corners_list


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate detection training data (bounding boxes only)')
    parser.add_argument('--source', default='./images',
                        help='Directory containing source photos')
    parser.add_argument('--output', default='./data/examples',
                        help='Output directory')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of example images')
    parser.add_argument('--mode', choices=['examples', 'batch'], default='examples',
                        help='Examples: debug images. Batch: train/val split.')
    parser.add_argument('--train-count', type=int, default=4000,
                        help='Training images (batch mode)')
    parser.add_argument('--val-count', type=int, default=1000,
                        help='Validation images (batch mode)')
    args = parser.parse_args()

    if args.mode == 'batch':
        _batch_generate(args)
    else:
        _example_generate(args)


def _example_generate(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} detection example images...")
    start = time.time()

    for i in range(args.count):
        img, det_labels, corners_list = generate_image(args.source)
        cv2.imwrite(str(output_dir / f"det_example_{i + 1:02d}.jpg"), img)
        cv2.imwrite(str(output_dir / f"det_example_{i + 1:02d}_debug.jpg"),
                    create_debug_image(img, corners_list))
        with open(output_dir / f"det_example_{i + 1:02d}.txt", 'w') as f:
            f.write('\n'.join(det_labels))
        print(f"  {i + 1:2d}/{args.count}")

    print(f"\n✅ Done in {time.time() - start:.1f}s → {output_dir.absolute()}")


def _batch_generate(args):
    from datetime import datetime

    total = args.train_count + args.val_count
    base_dir = Path(args.output)

    dirs = {
        'img_train': base_dir / "images" / "train",
        'img_val': base_dir / "images" / "val",
        'lbl_train': base_dir / "labels" / "train",
        'lbl_val': base_dir / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"Detection batch: {args.train_count} train + {args.val_count} val")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    for i in range(total):
        is_train = i < args.train_count
        img, det_labels, _ = generate_image(args.source)

        if is_train:
            prefix = f"train_{i + 1:06d}"
            img_dir, lbl_dir = dirs['img_train'], dirs['lbl_train']
        else:
            idx = i - args.train_count + 1
            prefix = f"val_{idx:06d}"
            img_dir, lbl_dir = dirs['img_val'], dirs['lbl_val']

        cv2.imwrite(str(img_dir / f"{prefix}.jpg"), img)
        with open(lbl_dir / f"{prefix}.txt", 'w') as f:
            f.write('\n'.join(det_labels))

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i + 1:5d}/{total}] {elapsed / 60:.1f}m | {rate:.1f}/s | ETA {eta:.0f}m")

    elapsed = time.time() - start
    print(f"\n✅ Complete! {total} images ({elapsed / 60:.1f}m) → {base_dir.absolute()}")


if __name__ == "__main__":
    main()