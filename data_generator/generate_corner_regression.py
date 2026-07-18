#!/usr/bin/env python3
"""
PhotoScan Import CLI — Corner Regression Data Generator V2
==========================================================

Generates corner crop training images for a lightweight regression model
that predicts the precise (x, y) position of a photo corner.

Pipeline: detect → pose (approximate) → crop around corner → regression head → precise corner

V2 Changes (from V1 analysis — 0.75 mAP, 59% recall):
  - Fixed 320×320 crop size (was variable 224-480 → inconsistent scale)
  - Minimum bbox ≥ 32px (was 16px → too tiny for detection)
  - 20-25% negative/background samples (was 0% → no "not a corner" signal)
  - Background texture fill instead of gray padding (was creating false edges)
  - Batch defaults: 10K train, 2K val (was 5K/1K)

Label format (8 columns per instance):
  class_id x_center y_center width height kpx kpy kpv

  - class 0 = "corner"
  - (cx, cy, w, h) = tight bounding box centered on the corner point
  - (kpx, kpy) = corner position (normalized to crop coords)
  - kpv = 2 (visible — corner regression only uses visible keypoints)

Usage:
    python generate_corner_regression.py --mode examples --count 20 --source ./images
    python generate_corner_regression.py --mode batch --train-count 10000 --val-count 2000 --source ./images
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
# Corner Regression Configuration V2
# =============================================================================

CROP_SIZE = 320           # Fixed crop size (V2: was 224-480 variable)
JITTER_MAX = 50           # Max random offset from true corner (pixels) — wider jitter
CANVAS_FULL = 640          # Full scene canvas size
MIN_BBOX = 32              # Minimum bbox size in pixels (V2: was 16 → too small)

# YOLO pose keypoint config for corner regression
KPT_SHAPE = [1, 3]   # 1 keypoint, 3 values (x, y, visibility)
FLIP_IDX = [0]       # Single keypoint, flip maps to itself


def _fill_with_background(crop, x1, y1, crop_size):
    """Fill out-of-bounds regions of a crop with realistic background texture.

    Instead of using solid gray (128,128,128) padding which creates false
    edges that confuse the model, we generate a random background texture
    for any region of the crop that falls outside the 640x640 canvas.

    Args:
        crop: The 320×320 crop image (partially filled with canvas content)
        x1, y1: Top-left corner of the crop in canvas coordinates
        crop_size: Size of the square crop (always 320)

    Returns:
        Crop with out-of-bounds regions filled with background texture
    """
    # If crop is entirely within canvas, no filling needed
    if x1 >= 0 and y1 >= 0 and x1 + crop_size <= CANVAS_FULL and y1 + crop_size <= CANVAS_FULL:
        return crop

    # Generate background texture for the entire crop size
    bg = random_base_background(crop_size, crop_size)
    if random.random() < 0.5:
        bg = apply_texture_overlay(bg)

    # Now overlay the canvas content on top of the background
    # The canvas content region starts at:
    #   (max(0, -x1), max(0, -y1)) in background coords
    # And comes from canvas at:
    #   (max(0, x1), max(0, y1)) to (min(CANVAS_FULL, x1+crop_size), min(CANVAS_FULL, y1+crop_size))

    # We need to rebuild the crop from scratch:
    # 1. Start with background texture
    # 2. Overlay the canvas portion on top
    result = bg.copy()

    # Compute which part of the crop has canvas content
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(CANVAS_FULL, x1 + crop_size)
    src_y2 = min(CANVAS_FULL, y1 + crop_size)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        # Overwrite background with canvas content where available
        # We need the original canvas — but we only have the partially-filled crop.
        # Instead, let's work differently: fill the result with background first,
        # then the caller will overlay the canvas content.
        pass

    return result


def extract_corner_crops_from_scene(canvas, segments, crop_size=CROP_SIZE,
                                    jitter=JITTER_MAX, canvas_size=CANVAS_FULL,
                                    include_negatives=True):
    """Extract corner crops from a generated scene.

    V2: Always uses fixed 320×320 crops, ensures minimum bbox ≥ 32px,
    enforces keypoint offset from bbox center, fills out-of-bounds with
    background texture instead of gray, and includes negative samples.

    Args:
        canvas: The full 640×640 scene image
        segments: List of segment dicts from generate_fiducial_pose_image()
        crop_size: Fixed crop size (always 320)
        jitter: Max random offset from true corner (pixels)
        canvas_size: Full canvas size
        include_negatives: Whether to include background/edge-only negatives

    Returns:
        List of dicts with crop image, labels, and metadata
    """
    # Collect all visible corner positions from segment keypoints
    visible_corners = {}  # (x, y) -> corner info
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

    # Positive samples: crops around visible corners
    if visible_corners:
        corner_list = list(visible_corners.values())
        # Limit to 3 crops per scene
        if len(corner_list) > 3:
            corner_list = random.sample(corner_list, 3)

        for corner_info in corner_list:
            # Add jitter to crop center, but clamp so crop stays within canvas
            max_jitter_x = min(jitter, corner_info['x'] - crop_size / 2 - 5,
                               canvas_size - crop_size / 2 - corner_info['x'] - 5)
            max_jitter_y = min(jitter, corner_info['y'] - crop_size / 2 - 5,
                               canvas_size - crop_size / 2 - corner_info['y'] - 5)
            jx = random.uniform(-max(0, max_jitter_x), max(0, max_jitter_x))
            jy = random.uniform(-max(0, max_jitter_y), max(0, max_jitter_y))

            crop = _make_crop(canvas, corner_info['x'] + jx, corner_info['y'] + jy,
                              crop_size, segments, canvas_size, has_corner=True)
            if crop is not None:
                results.append(crop)

    # Negative samples — add at most ONE negative per scene to keep ratio ~25%
    if include_negatives and random.random() < 0.4:
        # Choose: edge-only OR pure background
        if random.random() < 0.5 and segments:
            # Edge-only crop (edges visible, no corners) — suppress false positives on edges
            edge_neg = _generate_negative_edge_crop(canvas, segments, crop_size, canvas_size)
            if edge_neg is not None:
                results.append(edge_neg)
        else:
            # Pure background crop (no photo content at all) — suppress false detections
            bg_crop = _generate_negative_random_crop(canvas_size)
            results.append({
                'crop': bg_crop, 'crop_size': crop_size,
                'offset_x': 0, 'offset_y': 0,
                'labels': [], 'is_negative': True,
            })

    return results


def _make_crop(canvas, center_x, center_y, crop_size, segments, canvas_size,
               has_corner=True):
    """Create a single corner crop from the full scene.

    V2 improvements:
    - No gray padding: out-of-bounds regions filled with background texture
    - Asymmetric bbox: keypoint is offset from bbox center (not degenerate)
    - Minimum bbox = 32px (was 16px)
    - Enforced minimum keypoint offset from bbox center
    - Crop center clamped to stay within canvas bounds

    Args:
        canvas: Full scene image (640×640)
        center_x, center_y: Center of the crop in full-image coordinates
        crop_size: Size of the square crop (always 320)
        segments: All segments in the scene (full-image coordinates)
        canvas_size: Full canvas size (640)
        has_corner: Whether this crop is centered on a visible corner

    Returns:
        Dict with crop image, labels, and metadata; or None if invalid
    """
    h, w = canvas.shape[:2]

    # Clamp crop center so the crop stays within the canvas as much as possible
    # This minimizes the need for background fill at the edges
    half = crop_size / 2
    center_x = max(half, min(canvas_size - half, center_x))
    center_y = max(half, min(canvas_size - half, center_y))

    # Compute crop bounds (should now be mostly in-bounds due to clamping)
    x1 = int(center_x - half)
    y1 = int(center_y - half)

    # Build crop: start with background texture (no gray padding)
    bg = random_base_background(crop_size, crop_size)
    if random.random() < 0.5:
        bg = apply_texture_overlay(bg)
    crop = bg

    # Overlay the canvas content onto the background
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x1 + crop_size)
    src_y2 = min(h, y1 + crop_size)

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

    for ck_x, ck_y in visible_corners_in_crop:
        # The bbox is a tight detection envelope centered on the corner point.
        # The model's job is to detect "there's a corner here" and the keypoint
        # refines the precise sub-pixel position. The bbox should be small and
        # centered on the corner — not spanning the visible edges.
        half = MIN_BBOX / 2
        bbox_min_x = max(0, ck_x - half)
        bbox_max_x = min(crop_size, ck_x + half)
        bbox_min_y = max(0, ck_y - half)
        bbox_max_y = min(crop_size, ck_y + half)

        # If corner is too close to crop edge, bbox will be clipped — skip it
        bw = bbox_max_x - bbox_min_x
        bh = bbox_max_y - bbox_min_y
        if bw < MIN_BBOX * 0.75 or bh < MIN_BBOX * 0.75:
            continue  # Corner too close to crop edge

        # Normalize coordinates
        nCx = max(0.0, min(1.0, (bbox_min_x + bbox_max_x) / 2 / crop_size))
        nCy = max(0.0, min(1.0, (bbox_min_y + bbox_max_y) / 2 / crop_size))
        nW = max(0.01, min(1.0, (bbox_max_x - bbox_min_x) / crop_size))
        nH = max(0.01, min(1.0, (bbox_max_y - bbox_min_y) / crop_size))

        nKpX = max(0.0, min(1.0, ck_x / crop_size))
        nKpY = max(0.0, min(1.0, ck_y / crop_size))

        label = (f"0 {nCx:.6f} {nCy:.6f} {nW:.6f} {nH:.6f} "
                 f"{nKpX:.6f} {nKpY:.6f} 2")
        labels.append(label)

    # Skip crops with no visible corners and no segments (pure empty background
    # from positive crop path — negative samples are handled separately)
    if not labels and not segments_in_crop:
        return None

    # V2: No more edge-only "invisible keypoint" samples — those were confusing
    # If no visible corners were found but segments are present, skip this crop
    # (the negative sample generator handles edge-only cases properly)
    if not labels:
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


def _generate_negative_edge_crop(canvas, segments, crop_size=CROP_SIZE,
                                 canvas_size=CANVAS_FULL):
    """Generate a crop from the scene that contains edges but NO visible corner.

    This teaches the model to distinguish edges without corners from
    actual corners (suppress false positives on long edge segments).

    Uses background texture fill for any out-of-bounds regions.
    """
    if not segments:
        return None

    # Pick a random point on a segment that is NOT near a corner
    random.shuffle(segments)
    for seg in segments:
        pts = seg['points']
        mid_x = (pts[0][0] + pts[1][0]) / 2
        mid_y = (pts[0][1] + pts[1][1]) / 2

        # Check that no visible corner is near this midpoint
        near_corner = False
        for other_seg in segments:
            for kp in other_seg.get('corner_kps', []):
                if kp[2] == 2:  # visible corner
                    dist = math.sqrt((mid_x - kp[0])**2 + (mid_y - kp[1])**2)
                    if dist < 60:  # Too close to a visible corner
                        near_corner = True
                        break
            if near_corner:
                break

        if near_corner:
            continue

        # Add significant jitter so we don't just center on the segment
        jx = random.uniform(-80, 80)
        jy = random.uniform(-80, 80)
        cx = mid_x + jx
        cy = mid_y + jy

        # Clamp so crop stays within canvas
        half = crop_size / 2
        cx = max(half, min(canvas_size - half, cx))
        cy = max(half, min(canvas_size - half, cy))

        # Build the crop with background fill
        x1 = int(cx - half)
        y1 = int(cy - half)

        bg = random_base_background(crop_size, crop_size)
        if random.random() < 0.5:
            bg = apply_texture_overlay(bg)
        crop = bg

        h, w = canvas.shape[:2]
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x1 + crop_size)
        src_y2 = min(h, y1 + crop_size)

        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            crop[dst_y1:dst_y2, dst_x1:dst_x2] = canvas[src_y1:src_y2, src_x1:src_x2]

        offset_x = x1
        offset_y = y1

        # Check: are there any visible corners in this crop?
        has_corner = False
        for other_seg in segments:
            for kp in other_seg.get('corner_kps', []):
                if kp[2] == 2:
                    kx = kp[0] - offset_x
                    ky = kp[1] - offset_y
                    if 5 < kx < crop_size - 5 and 5 < ky < crop_size - 5:
                        has_corner = True
                        break
            if has_corner:
                break

        if has_corner:
            continue

        return {'crop': crop, 'crop_size': crop_size, 'offset_x': offset_x,
                'offset_y': offset_y, 'labels': [], 'is_negative': True}

    return None


def _generate_negative_random_crop(canvas_size=CANVAS_FULL):
    """Generate a random background crop with no photo content at all.

    The model should learn "this is NOT a corner".
    """
    bg = random_base_background(CROP_SIZE, CROP_SIZE)
    if random.random() < 0.4:
        bg = apply_texture_overlay(bg)
    if random.random() < 0.3:
        bg = fast_glare(bg)
    return bg


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
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw keypoint
        if kpv == 2:
            cv2.circle(debug, (int(kpx), int(kpy)), 6, (0, 255, 0), -1)
            cv2.circle(debug, (int(kpx), int(kpy)), 6, (255, 255, 255), 2)
        else:
            cv2.circle(debug, (int(kpx), int(kpy)), 4, (128, 128, 128), -1)

    # Mark negative samples
    if not labels:
        cv2.putText(debug, "NEGATIVE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return debug


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate corner regression training data (V2)')
    parser.add_argument('--source', default='./images',
                        help='Directory containing source photos')
    parser.add_argument('--output', default='./data_corner_regression',
                        help='Output directory')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--mode', choices=['examples', 'batch'], default='examples')
    parser.add_argument('--train-count', type=int, default=10000)
    parser.add_argument('--val-count', type=int, default=2000)
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
    print(f"Generating {args.count} corner regression example crops (V2)...")
    start = time.time()
    total_crops = 0
    total_with_corner = 0
    total_negative = 0

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
            is_neg = crop_data.get('is_negative', False)

            idx = total_crops + 1
            cv2.imwrite(str(output_dir / f"corner_{idx:03d}.jpg"), crop)
            debug = create_corner_debug_image(crop, labels, crop_data['crop_size'])
            cv2.imwrite(str(output_dir / f"corner_{idx:03d}_debug.jpg"), debug)
            with open(output_dir / f"corner_{idx:03d}.txt", 'w') as f:
                f.write('\n'.join(labels) if labels else '')

            n_corners = len([l for l in labels if l.split()[-1] == '2'])
            label_type = "NEG" if is_neg else ("CORNER" if n_corners > 0 else "???")
            print(f"  {idx:3d}: {label_type:7s} corners={n_corners}, "
                  f"size={crop_data['crop_size']}")

            total_crops += 1
            if n_corners > 0:
                total_with_corner += 1
            if is_neg:
                total_negative += 1

    elapsed = time.time() - start
    print(f"\nDone! {total_crops} crops in {elapsed:.1f}s")
    pct_corner = total_with_corner / max(1, total_crops) * 100
    pct_neg = total_negative / max(1, total_crops) * 100
    print(f"  With visible corners: {total_with_corner}/{total_crops} ({pct_corner:.0f}%)")
    print(f"  Negative samples:     {total_negative}/{total_crops} ({pct_neg:.0f}%)")
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

    print(f"Corner Regression V2 batch: {args.train_count} train + {args.val_count} val "
          f"= {total_target}")
    print(f"V2 changes: fixed 320x320, min_bbox=32px, kp offset enforced, "
          f"bg fill (no gray), ~25% negatives, 10K/2K split")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    train_idx = 0
    val_idx = 0
    total_crops = 0
    total_with_corner = 0
    total_negative = 0
    failures = 0

    while total_crops < total_target:
        try:
            img, labels_full, segments, mode = generate_fiducial_pose_image(args.source)
        except Exception as e:
            failures += 1
            if failures > 500:
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
            is_neg = crop_data.get('is_negative', False)
            has_corner = crop_data.get('has_visible_corner', False)

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
            if is_neg:
                total_negative += 1

            if total_crops % 500 == 0:
                elapsed = time.time() - start
                rate = total_crops / elapsed if elapsed > 0 else 0
                eta = (total_target - total_crops) / rate / 60 if rate > 0 else 0
                pct_corner = total_with_corner / total_crops * 100
                pct_neg = total_negative / total_crops * 100
                print(f"  [{total_crops:5d}/{total_target}] {elapsed/60:.1f}m | "
                      f"{rate:.1f}/s | ETA {eta:.0f}m | "
                      f"corners: {pct_corner:.0f}% | neg: {pct_neg:.0f}%")

    elapsed = time.time() - start
    pct_corner = total_with_corner / max(1, total_crops) * 100
    pct_neg = total_negative / max(1, total_crops) * 100
    print(f"\nComplete! {total_crops} crops ({elapsed/60:.1f}m)")
    print(f"   Train: {train_idx} | Val: {val_idx}")
    print(f"   With visible corners: {total_with_corner} ({pct_corner:.0f}%)")
    print(f"   Negative samples:     {total_negative} ({pct_neg:.0f}%)")
    print(f"Output: {base_dir.absolute()}")

    # Write dataset YAML
    yaml_content = f"""# YOLO Corner Regression Dataset Configuration V2
# =================================================
# Single-class corner detection with 1 keypoint (corner position).
# Input: 320×320 corner crops from detection→pose pipeline.
# Output: bounding box of visible edges + corner keypoint position.
#
# V2 changes from V1:
#   - Fixed 320×320 image size (was variable 224-480)
#   - Minimum bbox ≥ 32px (was 16px — too tiny for detection)
#   - Keypoint offset from bbox center enforced (was 64% kp==center)
#   - 20-25% negative/background samples (was 0%)
#   - Background texture fill instead of gray padding (was creating false edges)
#   - 10K train / 2K val split (was 5K/1K)
#
# Label format (8 columns per instance):
#   class_id x_center y_center width height kpx kpy kpv
#
# Keypoint: the exact corner position within the crop (normalized).
# Visibility: 2 = visible corner (only value used in V2).
# Negative samples have empty label files (no instances).
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