#!/usr/bin/env python3
"""
Evaluation script for corner regression model.

Measures pixel-level accuracy of keypoint (corner) predictions against
ground truth labels on the validation set.

Label format (YOLO pose, 8 columns per instance):
    class_id x_center y_center width height kpx kpy kpv

    kpx, kpy : normalized keypoint coordinates (0-1)
    kpv      : visibility flag (2=visible, 0=edge-only/no corner)

Predicted keypoints from YOLO are in absolute pixels (keypoints.xy)
or normalized (keypoints.xyn).
"""

import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image


# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = Path(
    "/Users/krys.petrie/dev/photo-pose-detector/training/"
    "runs/pose/runs/pose/corner-regression-v1/weights/best.pt"
)
VAL_IMG_DIR = Path(
    "/Users/krys.petrie/dev/photo-pose-detector/data_corner_regression/images/val/"
)
VAL_LBL_DIR = Path(
    "/Users/krys.petrie/dev/photo-pose-detector/data_corner_regression/labels/val/"
)

# Thresholds for "within X pixels" metrics (in pixels)
PIXEL_THRESHOLDS = [1, 2, 5, 10, 20, 50]


# ── Helper functions ───────────────────────────────────────────────────────────
def parse_label_file(label_path: str) -> list[dict]:
    """Parse a YOLO pose label file.

    Returns a list of dicts, one per instance:
        {
            'class': int,
            'bbox': [x_center, y_center, w, h],   # normalized
            'kpt': [kpx, kpy],                       # normalized
            'vis': int,                               # visibility flag
        }
    """
    instances = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Expected: class xc yc w h kpx kpy kpv
            assert len(parts) == 8, f"Unexpected format in {label_path}: {parts}"
            instances.append({
                'class': int(parts[0]),
                'bbox': [float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])],
                'kpt': [float(parts[5]), float(parts[6])],
                'vis': int(parts[7]),
            })
    return instances


def get_image_size(img_path: str) -> tuple[int, int]:
    """Return (width, height) of an image without loading full pixel data."""
    with Image.open(img_path) as img:
        return img.size  # (width, height)


# ── Main evaluation ────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Corner Regression Model — Pixel-Level Evaluation")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # Gather validation images
    img_files = sorted([f for f in os.listdir(VAL_IMG_DIR)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    print(f"Found {len(img_files)} validation images")

    # Check coordinate form
    print("\n── Coordinate Format Check ─────────────────────────────────")
    sample_img = str(VAL_IMG_DIR / img_files[0])
    r = model.predict(sample_img, verbose=False)[0]
    print(f"  keypoints.xy  shape: {r.keypoints.xy.shape}  → absolute pixels")
    print(f"  keypoints.xyn shape: {r.keypoints.xyn.shape}  → normalized (0-1)")
    print(f"  Sample xy  : {r.keypoints.xy[0].tolist()}")
    print(f"  Sample xyn : {r.keypoints.xyn[0].tolist()}")
    print(f"  Image orig_shape: {r.orig_shape}")
    print(f"  → Predicted keypoints.xy are in ABSOLUTE pixels.")
    print(f"  → Ground truth keypoints are NORMALIZED (0-1).")
    print(f"  → Must convert GT to pixels or pred to normalized for comparison.")

    # ── Per-image evaluation ───────────────────────────────────────────────
    pixel_errors = []       # L2 distance in pixels
    abs_x_errors = []       # absolute error in x (pixels)
    abs_y_errors = []       # absolute error in y (pixels)
    per_error_img_diag = []  # image diagonal for each error entry
    matched = 0
    unmatched_pred = 0
    unmatched_gt = 0
    multi_instance = 0

    print(f"\n── Running evaluation on {len(img_files)} images ─────────────")

    for i, img_file in enumerate(img_files):
        img_path = str(VAL_IMG_DIR / img_file)
        lbl_path = str(VAL_LBL_DIR / (os.path.splitext(img_file)[0] + '.txt'))

        if not os.path.exists(lbl_path):
            print(f"  ⚠ No label file for {img_file}, skipping")
            continue

        # Get image dimensions
        img_w, img_h = get_image_size(img_path)
        img_dims.append((img_w, img_h))

        # Parse ground truth
        gt_instances = parse_label_file(lbl_path)
        # Only consider visible keypoints for evaluation
        gt_visible = [inst for inst in gt_instances if inst['vis'] == 2]
        if not gt_visible:
            # If no visible keypoints, skip this image
            continue
        img_diag = np.sqrt(img_w**2 + img_h**2)

        if len(gt_visible) > 1:
            multi_instance += 1

        # Run prediction
        results = model.predict(img_path, verbose=False)
        r = results[0]

        # Extract predictions
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            # No predictions — all GT keypoints are unmatched
            unmatched_gt += len(gt_visible)
            for inst in gt_visible:
                # Record max possible error (diagonal of image)
                pixel_errors.append(np.sqrt(img_w**2 + img_h**2))
                abs_x_errors.append(img_w)
                abs_y_errors.append(img_h)
                per_error_img_diag.append(np.sqrt(img_w**2 + img_h**2))
            continue

        # Match predictions to GT using closest keypoint (greedy)
        # For single-keypoint pose, this is straightforward
        pred_kpts_px = r.keypoints.xy.cpu().numpy()   # (N_pred, 1, 2)
        pred_confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones(len(pred_kpts_px))

        # Convert GT keypoints to absolute pixels
        gt_kpts_px = []
        for inst in gt_visible:
            gt_kpts_px.append([inst['kpt'][0] * img_w, inst['kpt'][1] * img_h])
        gt_kpts_px = np.array(gt_kpts_px)  # (M_gt, 2)

        # Flatten predicted keypoints (single keypoint per detection)
        pred_kpts_flat = pred_kpts_px.squeeze(1)  # (N_pred, 2)

        # Greedy matching: for each GT keypoint, find closest prediction
        used_pred = set()
        for gt_kpt in gt_kpts_px:
            if len(pred_kpts_flat) == 0:
                unmatched_gt += 1
                pixel_errors.append(np.sqrt(img_w**2 + img_h**2))
                abs_x_errors.append(img_w)
                abs_y_errors.append(img_h)
                per_error_img_diag.append(np.sqrt(img_w**2 + img_h**2))
                continue

            distances = np.sqrt(np.sum((pred_kpts_flat - gt_kpt) ** 2, axis=1))
            best_idx = np.argmin(distances)
            best_dist = distances[best_idx]

            # Simple greedy: take closest, even if already used
            # (handles cases where multiple GT map to same pred)
            pixel_errors.append(best_dist)
            abs_x_errors.append(abs(pred_kpts_flat[best_idx][0] - gt_kpt[0]))
            abs_y_errors.append(abs(pred_kpts_flat[best_idx][1] - gt_kpt[1]))
            per_error_img_diag.append(img_diag)
            matched += 1

            if best_idx not in used_pred:
                used_pred.add(best_idx)
            # Count extra predictions as unmatched
        unmatched_pred += max(0, len(pred_kpts_flat) - len(used_pred))

    # ── Compute metrics ────────────────────────────────────────────────────
    pixel_errors = np.array(pixel_errors)
    abs_x_errors = np.array(abs_x_errors)
    abs_y_errors = np.array(abs_y_errors)
    per_error_img_diag = np.array(per_error_img_diag)

    print(f"\n{'=' * 70}")
    print(f"PIXEL-LEVEL ACCURACY RESULTS")
    print(f"{'=' * 70}")

    print(f"\n── Dataset Info ──────────────────────────────────────────")
    print(f"  Total validation images: {len(img_files)}")
    print(f"  Images with multi-instance GT: {multi_instance}")
    print(f"  Image size range: {img_dims[:, 0].min()}×{img_dims[:, 1].min()} "
          f"→ {img_dims[:, 0].max()}×{img_dims[:, 1].max()}")
    print(f"  Median image size: {np.median(img_dims[:, 0]):.0f}×{np.median(img_dims[:, 1]):.0f}")
    print(f"  Mean image size: {img_dims[:, 0].mean():.0f}×{img_dims[:, 1].mean():.0f}")

    print(f"\n── Matching Summary ──────────────────────────────────────")
    print(f"  Matched GT keypoints:  {matched}")
    print(f"  Unmatched GT (missed):  {unmatched_gt}")
    print(f"  Unmatched pred (FP):   {unmatched_pred}")
    total_gt = matched + unmatched_gt
    print(f"  Total GT keypoints:     {total_gt}")
    if total_gt > 0:
        print(f"  Detection rate:        {matched / total_gt * 100:.1f}%")

    print(f"\n── Pixel-Level Error Metrics (L2 distance) ──────────────")
    print(f"  Mean error:      {pixel_errors.mean():.4f} px")
    print(f"  Median error:    {np.median(pixel_errors):.4f} px")
    print(f"  Std deviation:   {pixel_errors.std():.4f} px")
    print(f"  Min error:       {pixel_errors.min():.4f} px")
    print(f"  Max error:       {pixel_errors.max():.4f} px")
    print(f"  90th percentile: {np.percentile(pixel_errors, 90):.4f} px")
    print(f"  95th percentile: {np.percentile(pixel_errors, 95):.4f} px")
    print(f"  99th percentile: {np.percentile(pixel_errors, 99):.4f} px")

    print(f"\n── Per-Axis Error (absolute) ─────────────────────────────")
    print(f"  X-axis mean:    {abs_x_errors.mean():.4f} px")
    print(f"  X-axis median:   {np.median(abs_x_errors):.4f} px")
    print(f"  Y-axis mean:    {abs_y_errors.mean():.4f} px")
    print(f"  Y-axis median:   {np.median(abs_y_errors):.4f} px")

    print(f"\n── Error Distribution (fraction within threshold) ───────")
    for threshold in PIXEL_THRESHOLDS:
        within = np.sum(pixel_errors <= threshold)
        pct = within / len(pixel_errors) * 100
        print(f"  Within {threshold:3d} px:  {within:5d} / {len(pixel_errors)}  ({pct:6.2f}%)")

    # Error histogram
    print(f"\n── Error Histogram ───────────────────────────────────────")
    bins = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, float('inf')]
    hist, _ = np.histogram(pixel_errors, bins=bins)
    for i in range(len(hist)):
        lower = bins[i]
        upper = bins[i + 1]
        if upper == float('inf'):
            label = f"  > {lower:.0f} px"
        else:
            label = f"  {lower:5.0f} - {upper:5.0f} px"
        bar = '█' * max(1, int(hist[i] / max(hist) * 50))
        print(f"{label}: {hist[i]:5d}  {bar}")

    # Normalized error (as fraction of image diagonal)
    img_diags = np.sqrt(img_dims[:, 0]**2 + img_dims[:, 1]**2)
    norm_errors = pixel_errors / img_diags
    print(f"\n── Normalized Error (fraction of image diagonal) ────────")
    print(f"  Mean normalized error:   {norm_errors.mean():.6f}")
    print(f"  Median normalized error: {np.median(norm_errors):.6f}")

    print(f"\n{'=' * 70}")
    print("Evaluation complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()