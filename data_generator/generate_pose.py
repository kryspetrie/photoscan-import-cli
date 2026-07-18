#!/usr/bin/env python3
"""
PhotoScan Import CLI — Pose Data Generator
==========================================

Generates synthetic training images for the **pose model only**.
Each image contains exactly 1 photograph, tightly framed with a small
margin, simulating the kind of crop that the detection model would
produce at inference time.

This eliminates the train-inference distribution mismatch that caused
poor keypoint localization in the previous architecture, where the
pose model trained on full-scene images (tiny photos) but deployed on
tightly cropped regions (photos filling the frame).

High-Resolution Render → Scale → Crop Pipeline:
  1. Render at 2× resolution (1280×1280)
  2. Place 1 photo with rotation, shadow, glare, perspective warp
  3. Find the bounding box of the 4 warped corners
  4. Scale the entire image so the bbox fills ~600px of 640px
  5. Crop a 640×640 window with random padding (15–40px)
  6. Generate YOLO-pose labels with corner keypoints

Output format per label line (17 columns):
    class_id x_center y_center width height kp0x kp0y kp0v kp1x kp1y kp1v kp2x kp2y kp2v kp3x kp3y kp3v

Keypoint order: LL (kp0), UL (kp1), UR (kp2), LR (kp3)

Usage:
    python generate_pose.py --mode examples --count 10 --source ./images
    python generate_pose.py --mode batch --train-count 4000 --val-count 1000 --source ./images --output ../data_pose
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
)

# =============================================================================
# Pose-specific configuration
# =============================================================================

POSE_CANVAS_INIT_SIZE = 1280   # High-res render canvas
POSE_FINAL_SIZE = 640           # Output image size
POSE_FIT_SIZE = 600             # Target bbox fill size
POSE_PADDING_MIN = 15           # Min padding around photo
POSE_PADDING_MAX = 40           # Max padding around photo
POSE_CORNER_MARGIN = 5          # Min pixels from corner to image edge (safety)

# Single photo: allow full rotation range
POSE_ROTATION_RANGE = ROTATION_RANGE  # 30°


# =============================================================================
# IMAGE GENERATION
# =============================================================================

def generate_pose_image(source_dir):
    """Generate a single tightly-cropped pose training image.

    Returns:
        final_image (BGR, 640×640)
        pose_label (str): YOLO-pose format label line
        final_corners (4,2): corner coordinates in final image space (for debug)
    """
    canvas_size = POSE_CANVAS_INIT_SIZE

    # Step 1: Create high-resolution background
    canvas = random_base_background(canvas_size, canvas_size)
    canvas = apply_texture_overlay(canvas)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
    canvas[:, :, 3] = 255

    # Step 2: Place 1 photo with random rotation
    photo_size = random.randint(
        int(canvas_size * 0.35),
        int(canvas_size * 0.75),
    )
    aspect = random.uniform(0.7, 1.3)
    target_w = photo_size
    target_h = int(photo_size * aspect)

    photo, orig_w, orig_h, new_w, new_h = load_and_prepare_photo(
        source_dir, target_w, target_h)

    rotation = random.uniform(-POSE_ROTATION_RANGE, POSE_ROTATION_RANGE)
    photo = rotate_photo(photo, rotation)

    # Center the photo on the canvas (enough room for any rotation + warp)
    cx = canvas_size / 2 + random.uniform(-50, 50)
    cy = canvas_size / 2 + random.uniform(-50, 50)

    # Shadow
    shadow_offset = random.randint(2, 5)
    angle = random.uniform(0, 2 * math.pi)
    offset_x = int(shadow_offset * math.cos(angle))
    offset_y = int(shadow_offset * math.sin(angle))
    canvas = apply_photo_shadow(
        canvas, photo, cx, cy,
        offset_x, offset_y,
        random.uniform(1.5, 3.0), random.uniform(0.15, 0.35),
        new_w, new_h, rotation,
    )

    canvas = composite_photo_at_center(canvas, photo, cx, cy)

    # Step 3: Get corners of the placed photo
    corners = get_rotated_polygon(new_w, new_h, cx, cy, rotation)
    corners_list = [corners]

    # Step 4: Apply perspective warp (keeping corners in bounds)
    canvas, persp_M, _ = apply_perspective_safe(canvas, corners_list)
    warped_corners = transform_corners(corners, persp_M)

    # Step 5: Calculate bounding box of the warped photo
    bbox_x1 = warped_corners[:, 0].min()
    bbox_y1 = warped_corners[:, 1].min()
    bbox_x2 = warped_corners[:, 0].max()
    bbox_y2 = warped_corners[:, 1].max()
    bbox_w = bbox_x2 - bbox_x1
    bbox_h = bbox_y2 - bbox_y1

    if bbox_w < 10 or bbox_h < 10:
        # Degenerate — retry with a new image
        return generate_pose_image(source_dir)

    # Step 6: Calculate scale to make bbox fill POSE_FIT_SIZE
    scale = min(POSE_FIT_SIZE / bbox_w, POSE_FIT_SIZE / bbox_h)

    # Step 7: Scale the entire image
    new_canvas_w = int(canvas_size * scale)
    new_canvas_h = int(canvas_size * scale)

    # Convert to BGR first (resize doesn't handle BGRA well)
    canvas_bgr = canvas if canvas.shape[2] == 3 else cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    scaled = cv2.resize(canvas_bgr, (new_canvas_w, new_canvas_h),
                        interpolation=cv2.INTER_LINEAR)

    # Step 8: Scale all corner coordinates
    scaled_corners = warped_corners * scale
    scaled_bbox_x1 = bbox_x1 * scale
    scaled_bbox_y1 = bbox_y1 * scale

    # Step 9: Compute crop position with random padding
    padding = random.randint(POSE_PADDING_MIN, POSE_PADDING_MAX)
    crop_x = int(scaled_bbox_x1) - padding
    crop_y = int(scaled_bbox_y1) - padding

    # Step 10: Pad image if crop goes out of bounds
    pad_top = max(0, -crop_y)
    pad_left = max(0, -crop_x)
    pad_bottom = max(0, (crop_y + POSE_FINAL_SIZE) - new_canvas_h)
    pad_right = max(0, (crop_x + POSE_FINAL_SIZE) - new_canvas_w)

    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
        # Use average border color for padding
        scaled = cv2.copyMakeBorder(
            scaled, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REPLICATE,
        )
        # Adjust crop offset for the padding
        crop_x += pad_left
        crop_y += pad_top

    # Step 11: Crop 640×640 window
    final = scaled[crop_y:crop_y + POSE_FINAL_SIZE,
                   crop_x:crop_x + POSE_FINAL_SIZE]

    # Handle edge case where crop is slightly too small
    if final.shape[0] < POSE_FINAL_SIZE or final.shape[1] < POSE_FINAL_SIZE:
        padded = np.full((POSE_FINAL_SIZE, POSE_FINAL_SIZE, 3), 128, dtype=np.uint8)
        padded[:final.shape[0], :final.shape[1]] = final
        final = padded

    # Step 12: Shift corner coordinates into final image space
    final_corners = scaled_corners - np.array([crop_x, crop_y])

    # Step 13: Verify all corners have sufficient margin from edges
    for i in range(4):
        x, y = final_corners[i]
        if (x < POSE_CORNER_MARGIN or x > POSE_FINAL_SIZE - POSE_CORNER_MARGIN or
                y < POSE_CORNER_MARGIN or y > POSE_FINAL_SIZE - POSE_CORNER_MARGIN):
            # Corner too close to edge — reject and regenerate
            return generate_pose_image(source_dir)

    # Step 14: Generate YOLO-pose label
    # Bounding box from the final corners
    fc = final_corners
    min_x = fc[:, 0].min()
    max_x = fc[:, 0].max()
    min_y = fc[:, 1].min()
    max_y = fc[:, 1].max()

    x_center = ((min_x + max_x) / 2) / POSE_FINAL_SIZE
    y_center = ((min_y + max_y) / 2) / POSE_FINAL_SIZE
    width = (max_x - min_x) / POSE_FINAL_SIZE
    height = (max_y - min_y) / POSE_FINAL_SIZE

    # Keypoints in LL, UL, UR, LR order (visibility=2 means visible)
    kp_parts = []
    for i in range(4):
        kp_parts.append(f"{fc[i, 0] / POSE_FINAL_SIZE:.6f}")
        kp_parts.append(f"{fc[i, 1] / POSE_FINAL_SIZE:.6f}")
        kp_parts.append("2")

    pose_label = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " + " ".join(kp_parts)

    return final, pose_label, final_corners


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate pose training data (corner keypoints only)')
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

    print(f"Generating {args.count} pose example images...")
    start = time.time()

    for i in range(args.count):
        img, pose_label, corners = generate_pose_image(args.source)
        cv2.imwrite(str(output_dir / f"pose_example_{i + 1:02d}.jpg"), img)
        cv2.imwrite(str(output_dir / f"pose_example_{i + 1:02d}_debug.jpg"),
                    create_debug_image(img, [corners]))
        with open(output_dir / f"pose_example_{i + 1:02d}.txt", 'w') as f:
            f.write(pose_label + '\n')
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

    print(f"Pose batch: {args.train_count} train + {args.val_count} val")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()
    rejected = 0

    for i in range(total):
        is_train = i < args.train_count

        img, pose_label, _ = generate_pose_image(args.source)

        if is_train:
            prefix = f"train_{i + 1:06d}"
            img_dir, lbl_dir = dirs['img_train'], dirs['lbl_train']
        else:
            idx = i - args.train_count + 1
            prefix = f"val_{idx:06d}"
            img_dir, lbl_dir = dirs['img_val'], dirs['lbl_val']

        cv2.imwrite(str(img_dir / f"{prefix}.jpg"), img)
        with open(lbl_dir / f"{prefix}.txt", 'w') as f:
            f.write(pose_label + '\n')

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i + 1:5d}/{total}] {elapsed / 60:.1f}m | {rate:.1f}/s | ETA {eta:.0f}m")

    elapsed = time.time() - start
    print(f"\n✅ Complete! {total} images ({elapsed / 60:.1f}m) → {base_dir.absolute()}")


if __name__ == "__main__":
    main()