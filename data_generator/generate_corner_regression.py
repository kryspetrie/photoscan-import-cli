#!/usr/bin/env python3
"""
Photo Pose Detector — Corner Regression Data Generator
========================================================

Generates corner crop training images for a lightweight regression model
that predicts the precise (x, y) position of a photo corner.

Pipeline: detect → pose (approximate) → crop around corner → regression head → precise corner

Approach: Reuse generate_fiducial_pose_image() to produce full 640×640 scenes
with visible photo segments and corner keypoints. Then extract 320×320 crops
around each visible corner. The corner position within the crop is the keypoint label.

Label format (8 columns per instance):
  class_id x_center y_center width height kpx kpy kpv

  - class 0 = "corner"
  - (cx, cy, w, h) = bounding box of visible edges in the 320×320 crop
  - (kpx, kpy) = corner position (normalized to crop coords)
  - kpv = 2 (visible) or 0 (off-screen — edge-only samples)

Usage:
    python generate_corner_regression.py --mode examples --count 20 --source ./images
    python generate_corner_regression.py --mode batch --train-count 5000 --val-count 1000 --source ./images
"""

import cv2
import numpy as np
from pathlib import Path
import random
import time
import math
import argparse

from generate_common import (
    CANVAS_SIZE, BOUND_MARGIN,
    get_rotated_polygon, compute_rotated_bbox, rotate_photo,
    random_base_background, apply_texture_overlay, fast_glare,
    apply_photo_shadow, composite_photo_at_center,
    load_and_prepare_photo, create_debug_image,
    polygon_intersection_area, polygon_area,
)

from generate_fiducial_pose import (
    generate_fiducial_pose_image, extract_segments_from_photo,
    count_unique_visible_corners, compute_bbox_coverage,
    clip_line_to_rect, segment_to_yolo_label,
    _place_photo_on_canvas, build_photo_grid,
    choose_crop_offset, apply_perspective_mild, _transform_corners,
    CORNER_VISIBILITY_MARGIN, MIN_SEGMENT_LENGTH,
)

# =============================================================================
# Corner Regression Configuration
# =============================================================================

CROP_SIZE_DEFAULT = 320    # Default corner crop size (pixels)
CROP_SIZE_MIN = 224        # Minimum crop size
CROP_SIZE_MAX = 480        # Maximum crop size
JITTER_MAX = 40            # Max random offset from true corner (pixels)
CANVAS_FULL = 640          # Full scene canvas size

# YOLO pose keypoint config for corner regression
KPT_SHAPE = [1, 3]   # 1 keypoint, 3 values (x, y, visibility)
FLIP_IDX = [0]       # Single keypoint, flip maps to itself


def extract_corner_crops_from_scene(canvas, segments, crop_size=CROP_SIZE_DEFAULT,
                                    jitter=JITTER_MAX, canvas_size=CANVAS_FULL):
    """Extract corner crops from a generated scene.

    Uses the already-computed segments with their final-image coordinates
    to find visible corners and extract crops around them.

    Args:
        canvas: The full 640×640 scene image
        segments: List of segment dicts from generate_fiducial_pose_image()
        crop_size: Base crop size (pixels)
        jitter: Max random offset from true corner center
        canvas_size: Full canvas size

    Returns:
        List of dicts with crop image, labels, and metadata
    """
    # Collect all visible corner positions from segment keypoints
    visible_corners = {}  # (x, y) -> list of corner info
    for seg in segments:
        for kpi, kp in enumerate(seg['corner_kps']):
            x, y, vis = kp[0], kp[1], kp[2]
            cidx = seg['corner_indices'][kpi]
            if vis == 2 and cidx is not None:
                # Round to avoid near-duplicate corners from adjacent segments
                key = (round(x, 1), round(y, 1))
                if key not in visible_corners:
                    visible_corners[key] = {
                        'x': x, 'y': y, 'photo_idx_hint': cidx,
                    }

    results = []

    # If we have visible corners, create crops around each one
    if visible_corners:
        corner_list = list(visible_corners.values())
        # Limit to 3 crops per scene
        if len(corner_list) > 3:
            corner_list = random.sample(corner_list, 3)

        for corner_info in corner_list:
            # Add jitter to crop center
            jx = random.uniform(-jitter, jitter)
            jy = random.uniform(-jitter, jitter)
            cs = random.randint(CROP_SIZE_MIN, CROP_SIZE_MAX)

            crop = _make_crop(canvas, corner_info['x'] + jx, corner_info['y'] + jy,
                              cs, segments, canvas_size, has_corner=True)
            if crop is not None:
                results.append(crop)
    else:
        # No visible corners — create an edge-only crop from segment midpoints
        if segments:
            seg = random.choice(segments)
            mid_x = (seg['points'][0][0] + seg['points'][1][0]) / 2
            mid_y = (seg['points'][0][1] + seg['points'][1][1]) / 2
            crop = _make_crop(canvas, mid_x, mid_y,
                              random.randint(CROP_SIZE_MIN, CROP_SIZE_MAX),
                              segments, canvas_size, has_corner=False)
            if crop is not None:
                results.append(crop)

    return results


def _make_crop(canvas, center_x, center_y, crop_size, segments, canvas_size,
               has_corner=True):
    """Create a single corner crop from the full scene.

    Args:
        canvas: Full scene image
        center_x, center_y: Center of the crop in full-image coordinates
        crop_size: Size of the square crop
        segments: All segments in the scene (full-image coordinates)
        canvas_size: Full canvas size
        has_corner: Whether this crop is centered on a visible corner

    Returns:
        Dict with crop image, labels, and metadata; or None if invalid
    """
    h, w = canvas.shape[:2]

    # Compute crop bounds
    x1 = int(center_x - crop_size / 2)
    y1 = int(center_y - crop_size / 2)

    # Extract crop with gray padding for out-of-bounds
    pad_color = (128, 128, 128)
    crop = np.full((crop_size, crop_size, 3), pad_color, dtype=np.uint8)

    # Source region in full image
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x1 + crop_size)
    src_y2 = min(h, y1 + crop_size)

    # Destination region in crop
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = canvas[src_y1:src_y2, src_x1:src_x2]

    offset_x = x1
    offset_y = y1

    # Collect visible corners and segments within the crop
    visible_corners_in_crop = []
    segments_in_crop = []

    # Find segments visible in the crop
    for seg in segments:
        pts = seg['points']
        # Transform segment points to crop coordinates
        p1x, p1y = pts[0][0] - offset_x, pts[0][1] - offset_y
        p2x, p2y = pts[1][0] - offset_x, pts[1][1] - offset_y

        # Check if segment intersects the crop
        seg_min_x = min(p1x, p2x)
        seg_max_x = max(p1x, p2x)
        seg_min_y = min(p1y, p2y)
        seg_max_y = max(p1y, p2y)

        if (seg_max_x > 0 and seg_min_x < crop_size and
            seg_max_y > 0 and seg_min_y < crop_size):
            segments_in_crop.append(seg)

            # Check segment keypoints for visible corners in the crop
            for kpi, kp in enumerate(seg['corner_kps']):
                kx, ky, kvis = kp[0], kp[1], kp[2]
                if kvis == 2:
                    # Transform to crop coordinates
                    ck_x = kx - offset_x
                    ck_y = ky - offset_y
                    # Check if the corner is within the crop (with margin)
                    margin = 3
                    if (margin < ck_x < crop_size - margin and
                        margin < ck_y < crop_size - margin):
                        # Check for duplicates (same corner from adjacent segments)
                        is_dup = False
                        for vc in visible_corners_in_crop:
                            if abs(vc[0] - ck_x) < 2 and abs(vc[1] - ck_y) < 2:
                                is_dup = True
                                break
                        if not is_dup:
                            visible_corners_in_crop.append((ck_x, ck_y))

    # Build YOLO labels — one instance per visible corner
    labels = []

    # First: label visible corners with edges
    for ck_x, ck_y in visible_corners_in_crop:
        # Collect segment endpoints near this corner for the bounding box.
        # Only include endpoints that are close to this specific corner,
        # not all edges in the crop (which may belong to other corners).
        edge_min_x, edge_max_x = crop_size, 0
        edge_min_y, edge_max_y = crop_size, 0
        nearby_endpoint_count = 0

        for seg in segments_in_crop:
            pts = seg['points']
            p1x, p1y = pts[0][0] - offset_x, pts[0][1] - offset_y
            p2x, p2y = pts[1][0] - offset_x, pts[1][1] - offset_y

            # Only include endpoints that are close to THIS corner
            dist_p1 = math.sqrt((p1x - ck_x)**2 + (p1y - ck_y)**2)
            dist_p2 = math.sqrt((p2x - ck_x)**2 + (p2y - ck_y)**2)
            nearby_threshold = crop_size * 0.25  # Tighter radius per-corner

            if dist_p1 < nearby_threshold:
                edge_min_x = min(edge_min_x, p1x)
                edge_max_x = max(edge_max_x, p1x)
                edge_min_y = min(edge_min_y, p1y)
                edge_max_y = max(edge_max_y, p1y)
                nearby_endpoint_count += 1
            if dist_p2 < nearby_threshold:
                edge_min_x = min(edge_min_x, p2x)
                edge_max_x = max(edge_max_x, p2x)
                edge_min_y = min(edge_min_y, p2y)
                edge_max_y = max(edge_max_y, p2y)
                nearby_endpoint_count += 1

            if nearby_endpoint_count == 0:
                # Fallback: small bbox centered on corner
                bbox_min_x = max(0, ck_x - 32)
                bbox_max_x = min(crop_size, ck_x + 32)
                bbox_min_y = max(0, ck_y - 32)
                bbox_max_y = min(crop_size, ck_y + 32)
            else:
                # Expand bbox slightly, and always include the corner point itself
                expand = 4
                bbox_min_x = max(0, min(edge_min_x, ck_x) - expand)
                bbox_max_x = min(crop_size, max(edge_max_x, ck_x) + expand)
                bbox_min_y = max(0, min(edge_min_y, ck_y) - expand)
                bbox_max_y = min(crop_size, max(edge_max_y, ck_y) + expand)

        # Minimum bbox size
        min_bbox = 16
        bw = bbox_max_x - bbox_min_x
        bh = bbox_max_y - bbox_min_y
        if bw < min_bbox:
            cx = (bbox_min_x + bbox_max_x) / 2
            bbox_min_x = cx - min_bbox / 2
            bbox_max_x = cx + min_bbox / 2
        if bh < min_bbox:
            cy = (bbox_min_y + bbox_max_y) / 2
            bbox_min_y = cy - min_bbox / 2
            bbox_max_y = cy + min_bbox / 2

        # Normalize coordinates
        bCx = max(0.0, min(1.0, (bbox_min_x + bbox_max_x) / 2 / crop_size))
        bCy = max(0.0, min(1.0, (bbox_min_y + bbox_max_y) / 2 / crop_size))
        bW = max(0.01, min(1.0, (bbox_max_x - bbox_min_x) / crop_size))
        bH = max(0.01, min(1.0, (bbox_max_y - bbox_min_y) / crop_size))

        nKpX = max(0.0, min(1.0, ck_x / crop_size))
        nKpY = max(0.0, min(1.0, ck_y / crop_size))

        label = (f"0 {bCx:.6f} {bCy:.6f} {bW:.6f} {bH:.6f} "
                 f"{nKpX:.6f} {nKpY:.6f} 2")
        labels.append(label)

    # Edge-only sample (no visible corner, but visible edges)
    if not labels and segments_in_crop:
        all_pts = []
        for seg in segments_in_crop:
            pts = seg['points']
            for p in pts:
                all_pts.append((p[0] - offset_x, p[1] - offset_y))

        if all_pts:
            edge_min_x = min(p[0] for p in all_pts)
            edge_max_x = max(p[0] for p in all_pts)
            edge_min_y = min(p[1] for p in all_pts)
            edge_max_y = max(p[1] for p in all_pts)

            expand = 4
            bMinX = max(0, edge_min_x - expand)
            bMaxX = min(crop_size, edge_max_x + expand)
            bMinY = max(0, edge_min_y - expand)
            bMaxY = min(crop_size, edge_max_y + expand)

            bCx = max(0.0, min(1.0, (bMinX + bMaxX) / 2 / crop_size))
            bCy = max(0.0, min(1.0, (bMinY + bMaxY) / 2 / crop_size))
            bW = max(0.01, min(1.0, (bMaxX - bMinX) / crop_size))
            bH = max(0.01, min(1.0, (bMaxY - bMinY) / crop_size))

            # Edge intersection estimate as invisible keypoint
            est_x = max(0.0, min(1.0, (edge_min_x + edge_max_x) / 2 / crop_size))
            est_y = max(0.0, min(1.0, (edge_min_y + edge_max_y) / 2 / crop_size))

            label = (f"0 {bCx:.6f} {bCy:.6f} {bW:.6f} {bH:.6f} "
                     f"{est_x:.6f} {est_y:.6f} 0")
            labels.append(label)

    # Skip empty crops
    if not labels and not segments_in_crop:
        return None

    return {
        'crop': crop,
        'crop_size': crop_size,
        'offset_x': offset_x,
        'offset_y': offset_y,
        'visible_corners': visible_corners_in_crop,
        'segments_in_crop': segments_in_crop,
        'labels': labels,
        'has_visible_corner': len(visible_corners_in_crop) > 0,
    }


def create_corner_debug_image(crop, labels, crop_size):
    """Create debug image with corner position overlay."""
    debug = crop.copy()

    for label in labels:
        parts = label.strip().split()
        if len(parts) < 8:
            continue
        _, cx, cy, bw, bh, kpx, kpy, kpv = parts
        cx, cy = float(cx) * crop_size, float(cy) * crop_size
        bw, bh = float(bw) * crop_size, float(bh) * crop_size
        kpx, kpy = float(kpx) * crop_size, float(kpy) * crop_size
        kpv = int(kpv)

        # Draw bbox
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # Draw keypoint
        if kpv == 2:
            cv2.circle(debug, (int(kpx), int(kpy)), 8, (0, 255, 0), -1)
            cv2.circle(debug, (int(kpx), int(kpy)), 8, (255, 255, 255), 2)
        else:
            cv2.circle(debug, (int(kpx), int(kpy)), 6, (128, 128, 128), -1)
            cv2.putText(debug, "inv", (int(kpx) - 10, int(kpy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

    return debug


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate corner regression training data')
    parser.add_argument('--source', default='./images',
                        help='Directory containing source photos')
    parser.add_argument('--output', default='./data_corner_regression',
                        help='Output directory')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--mode', choices=['examples', 'batch'], default='examples')
    parser.add_argument('--train-count', type=int, default=5000)
    parser.add_argument('--val-count', type=int, default=1000)
    parser.add_argument('--force-mode', choices=['one_corner', 'grid',
                        'two_corners', 'no_photo'], default=None)
    args = parser.parse_args()

    if args.mode == 'batch':
        _batch_generate(args)
    else:
        _example_generate(args)


def _example_generate(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.count} corner regression example crops...")
    start = time.time()
    total_crops = 0
    total_with_corner = 0

    while total_crops < args.count:
        try:
            img, labels_full, segments, mode = generate_fiducial_pose_image(
                args.source, force_mode=args.force_mode)
        except Exception as e:
            continue

        if not segments:
            continue

        crops = extract_corner_crops_from_scene(img, segments)

        for crop_data in crops:
            if total_crops >= args.count:
                break

            crop = crop_data['crop']
            labels = crop_data['labels']
            has_corner = crop_data['has_visible_corner']

            idx = total_crops + 1
            cv2.imwrite(str(output_dir / f"corner_{idx:03d}.jpg"), crop)
            debug = create_corner_debug_image(crop, labels, crop_data['crop_size'])
            cv2.imwrite(str(output_dir / f"corner_{idx:03d}_debug.jpg"), debug)
            with open(output_dir / f"corner_{idx:03d}.txt", 'w') as f:
                f.write('\n'.join(labels) if labels else '')

            n_corners = len([l for l in labels if l.split()[-1] == '2'])
            print(f"  {idx:3d}: corners={n_corners}, "
                  f"edges={len(crop_data['segments_in_crop'])}, "
                  f"has_corner={has_corner}, size={crop_data['crop_size']}")

            total_crops += 1
            if has_corner:
                total_with_corner += 1

    elapsed = time.time() - start
    print(f"\nDone! {total_crops} crops in {elapsed:.1f}s")
    pct = total_with_corner / max(1, total_crops) * 100
    print(f"  With visible corners: {total_with_corner}/{total_crops} ({pct:.0f}%)")
    print(f"Output: {output_dir.absolute()}")


def _batch_generate(args):
    from datetime import datetime
    total_target = args.train_count + args.val_count
    base_dir = Path(args.output)
    dirs = {
        'img_train': base_dir / "images" / "train",
        'img_val': base_dir / "images" / "val",
        'lbl_train': base_dir / "labels" / "train",
        'lbl_val': base_dir / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"Corner Regression batch: {args.train_count} train + {args.val_count} val "
          f"= {total_target}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    train_idx = 0
    val_idx = 0
    total_crops = 0
    total_with_corner = 0
    failures = 0

    while total_crops < total_target:
        try:
            img, labels_full, segments, mode = generate_fiducial_pose_image(args.source)
        except Exception as e:
            failures += 1
            if failures > 200:
                print(f"Too many failures ({failures}), aborting.")
                break
            continue

        if not segments:
            continue

        crops = extract_corner_crops_from_scene(img, segments)

        for crop_data in crops:
            if total_crops >= total_target:
                break

            crop = crop_data['crop']
            labels = crop_data['labels']
            has_corner = crop_data['has_visible_corner']

            if not labels:
                continue

            is_train = total_crops < args.train_count
            if is_train:
                train_idx += 1
                prefix = f"train_{train_idx:06d}"
                img_dir = dirs['img_train']
                lbl_dir = dirs['lbl_train']
            else:
                val_idx += 1
                prefix = f"val_{val_idx:06d}"
                img_dir = dirs['img_val']
                lbl_dir = dirs['lbl_val']

            cv2.imwrite(str(img_dir / f"{prefix}.jpg"), crop)
            with open(lbl_dir / f"{prefix}.txt", 'w') as f:
                f.write('\n'.join(labels) if labels else '')

            total_crops += 1
            if has_corner:
                total_with_corner += 1

            if total_crops % 200 == 0:
                elapsed = time.time() - start
                rate = total_crops / elapsed if elapsed > 0 else 0
                eta = (total_target - total_crops) / rate / 60 if rate > 0 else 0
                pct_corner = total_with_corner / total_crops * 100
                print(f"  [{total_crops:5d}/{total_target}] {elapsed/60:.1f}m | "
                      f"{rate:.1f}/s | ETA {eta:.0f}m | corners: {pct_corner:.0f}%")

    elapsed = time.time() - start
    pct_corner = total_with_corner / max(1, total_crops) * 100
    print(f"\nComplete! {total_crops} crops ({elapsed/60:.1f}m)")
    print(f"   Train: {train_idx} | Val: {val_idx}")
    print(f"   With visible corners: {total_with_corner} ({pct_corner:.0f}%)")
    print(f"Output: {base_dir.absolute()}")

    # Write dataset YAML
    yaml_content = f"""# YOLO Corner Regression Dataset Configuration
# ==========================================
# Single-class corner detection with 1 keypoint (corner position).
# Input: 320×320 corner crops from detection→pose pipeline.
# Output: bounding box of visible edges + corner keypoint position.
#
# Label format (8 columns per instance):
#   class_id x_center y_center width height kpx kpy kpv
#
# Keypoint: the exact corner position within the crop (normalized).
# Visibility: 2 = visible corner, 0 = edge-only (no corner visible).
#
# Usage:
#   python3 train_corner_regression.py
#   (or: python3 train_corner_regression.py --data dataset_corner_regression.yaml)

path: {base_dir.absolute()}
train: images/train
val: images/val

nc: 1
names:
  0: corner

# Single keypoint: corner position
kpt_shape: [1, 3]
flip_idx: [0]
"""
    yaml_path = base_dir / "dataset_corner_regression.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nDataset config: {yaml_path}")

    # Also write a copy in the training directory
    training_yaml = Path(__file__).parent.parent / "training" / "dataset_corner_regression.yaml"
    training_yaml.parent.mkdir(parents=True, exist_ok=True)
    training_yaml_content = yaml_content.replace(
        f"path: {base_dir.absolute()}",
        f"path: {base_dir}"
    )
    with open(training_yaml, 'w') as f:
        f.write(training_yaml_content)
    print(f"Training config: {training_yaml}")


if __name__ == "__main__":
    main()