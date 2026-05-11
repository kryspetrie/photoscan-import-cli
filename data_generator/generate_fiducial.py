#!/usr/bin/env python3
"""
Photo Pose Detector — Fiducial Corner Data Generator
=====================================================

Generates synthetic training data for a SINGLE fiducial corner detection
model with 4 classes (UL, UR, LL, LR). The model learns to both DETECT
the corner and CLASSIFY its orientation from the distinctive L-shaped
boundary pattern.

APPROACH: Render a single large photo on a background (using the SAME pipeline
as the detection/pose generators), then extract 640×640 crops around each
corner. A binary photo mask is rendered through the same warp pipeline to
find pixel-precise corner positions via Harris detection.

SINGLE MODEL WITH 4 CLASSES:
  - Class 0: UL (┏) — photo extends right + down
  - Class 1: UR (┓) — photo extends left + down
  - Class 2: LL (┗) — photo extends right + up
  - Class 3: LR (┛) — photo extends left + up

The model can determine orientation because each corner has a unique L-shape.
At inference, we run ONE model on each corner crop, and the class output tells
us which corner type it found. This is simpler and more robust than running
4 separate models.

KEY DESIGN PRINCIPLES:
  1. Background uses the SAME generation pipeline as detection/pose data
  2. Photo compositing uses proper PRE-MULTIPLIED alpha blending
  3. Corner positions within crops are RANDOMIZED (not always centered)
  4. Corner positions refined via Harris on binary mask (pixel-precise)
  5. NO border replication — large canvas, skip out-of-bounds corners
  6. Output: single dataset with 4 YOLO classes

Usage:
    python generate_fiducial.py --mode examples --count 10 --source ./images
    python generate_fiducial.py --mode batch --train-count 4000 --val-count 1000 --source ./images
"""

import cv2
import numpy as np
from pathlib import Path
import random
import time
import math
import argparse

from generate_common import (
    CANVAS_SIZE, ROTATION_RANGE,
    get_rotated_polygon, rotate_photo,
    random_base_background, apply_texture_overlay,
    fast_glare,
    apply_photo_shadow,
    load_and_prepare_photo,
)

# =============================================================================
# Fiducial-specific configuration
# =============================================================================

FIDUCIAL_RENDER_SIZE = 2000   # Large canvas ensures all corners have margin
FIDUCIAL_CROP_SIZE = 640      # Final crop size
HALF_CROP = FIDUCIAL_CROP_SIZE // 2  # 320px — minimum margin needed

# Photo size range at render scale — large enough so L-shape fills the crop
FIDUCIAL_PHOTO_SIZE_MIN = 800
FIDUCIAL_PHOTO_SIZE_MAX = 1400

# Photo center offset from canvas center (for variety in corner positions)
FIDUCIAL_CENTER_OFFSET = 150

# Bounding box size as fraction of the crop
FIDUCIAL_BBOX_FRAC_MIN = 0.20
FIDUCIAL_BBOX_FRAC_MAX = 0.45

# How much the corner can be offset from the crop center (in pixels).
# This ensures the model learns to find corners at various positions.
FIDUCIAL_CORNER_JITTER = 180

# Rotation range for photo placement (degrees)
FIDUCIAL_ROTATION_RANGE = 30

# Corner type mapping
# get_rotated_polygon returns corners in order: LL, UL, UR, LR
# We map each to a YOLO class ID based on its orientation
CORNER_NAMES = ['ll', 'ul', 'ur', 'lr']
CORNER_CLASS_IDS = {
    'ul': 0,
    'ur': 1,
    'll': 2,
    'lr': 3,
}

# Debug colors for each corner type (BGR)
CORNER_COLORS = {
    'ul': (0, 255, 0),    # Green
    'ur': (255, 0, 0),    # Blue
    'll': (0, 255, 255),  # Yellow
    'lr': (255, 0, 255),  # Magenta
}


# =============================================================================
# PHOTO COMPOSITING — Proper pre-multiplied alpha blending
# =============================================================================

def composite_photo_onto(canvas_bgr, photo_bgra, cx, cy):
    """Composite a BGRA photo onto a BGR canvas using pre-multiplied alpha.

    OpenCV's warpAffine produces PRE-MULTIPLIED alpha — the RGB channels at
    anti-aliased edges already have alpha factored in. Using the naive formula
    (photo * alpha + background * (1-alpha)) would apply alpha twice, creating
    a dark halo. The correct formula is:

        result = photo_premultiplied + background * (1 - alpha)

    Args:
        canvas_bgr: Background image (BGR, uint8) — modified in place
        photo_bgra: Photo with alpha (BGRA, uint8, pre-multiplied alpha expected)
        cx, cy: Center position for the photo

    Returns:
        (canvas_bgr, tl_x, tl_y, pw, ph): The modified canvas and placement info
    """
    h, w = canvas_bgr.shape[:2]
    ph, pw = photo_bgra.shape[:2]

    tl_x = int(round(cx - pw / 2))
    tl_y = int(round(cy - ph / 2))

    # Compute overlap region
    src_x1 = max(0, -tl_x)
    src_y1 = max(0, -tl_y)
    src_x2 = min(pw, w - tl_x)
    src_y2 = min(ph, h - tl_y)
    dst_x1 = max(0, tl_x)
    dst_y1 = max(0, tl_y)
    dst_x2 = min(w, tl_x + pw)
    dst_y2 = min(h, tl_y + ph)

    copy_w = dst_x2 - dst_x1
    copy_h = dst_y2 - dst_y1

    if copy_w <= 0 or copy_h <= 0:
        return canvas_bgr, tl_x, tl_y, 0, 0

    # Extract regions
    canvas_region = canvas_bgr[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    photo_region = photo_bgra[src_y1:src_y2, src_x1:src_x2].astype(np.float32)

    alpha = photo_region[:, :, 3:4] / 255.0

    # Pre-multiplied alpha compositing
    result = photo_region[:, :, :3] + canvas_region * (1.0 - alpha)

    canvas_bgr[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(result, 0, 255).astype(np.uint8)

    return canvas_bgr, tl_x, tl_y, pw, ph


# =============================================================================
# SHADOW ON BGR BACKGROUND
# =============================================================================

def apply_shadow_bgr(canvas_bgr, photo_bgra, cx, cy,
                     offset_x, offset_y, blur_sigma, opacity,
                     orig_w, orig_h, rotation):
    """Apply drop shadow to a BGR canvas (same logic as BGRA version)."""
    ch, cw = canvas_bgr.shape[:2]
    canvas_f = canvas_bgr.astype(np.float32) / 255.0

    blur_pad = int(3 * blur_sigma) + 1
    mask_w = orig_w + blur_pad * 2
    mask_h = orig_h + blur_pad * 2
    shadow_mask = np.zeros((mask_h, mask_w), dtype=np.float32)
    shadow_mask[blur_pad:blur_pad + orig_h, blur_pad:blur_pad + orig_w] = 1.0
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=blur_sigma)

    if abs(rotation) > 0.5:
        center_rot = (mask_w / 2, mask_h / 2)
        rot_matrix = cv2.getRotationMatrix2D(center_rot, rotation, 1.0)
        cos_a = abs(rot_matrix[0, 0])
        sin_a = abs(rot_matrix[0, 1])
        new_w = int(mask_h * sin_a + mask_w * cos_a)
        new_h = int(mask_w * sin_a + mask_h * cos_a)
        rot_matrix[0, 2] += (new_w - mask_w) / 2
        rot_matrix[1, 2] += (new_h - mask_h) / 2
        shadow_mask = cv2.warpAffine(shadow_mask, rot_matrix, (new_w, new_h),
                                     borderValue=0, flags=cv2.INTER_LINEAR)
        mask_w = new_w
        mask_h = new_h

    if shadow_mask.max() > 0:
        shadow_mask = shadow_mask / shadow_mask.max()

    # Rotate the offset direction to match scene lighting
    rot_rad = math.radians(rotation)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    rotated_offset_x = offset_x * cos_r - offset_y * sin_r
    rotated_offset_y = offset_x * sin_r + offset_y * cos_r

    shadow_cx = cx + rotated_offset_x * 0.5
    shadow_cy = cy + rotated_offset_y * 0.5

    shadow_top_left_x = int(shadow_cx - mask_w / 2)
    shadow_top_left_y = int(shadow_cy - mask_h / 2)

    y1, y2 = shadow_top_left_y, shadow_top_left_y + mask_h
    x1, x2 = shadow_top_left_x, shadow_top_left_x + mask_w

    clip_y1 = max(0, y1)
    clip_y2 = min(ch, y2)
    clip_x1 = max(0, x1)
    clip_x2 = min(cw, x2)

    if clip_y2 > clip_y1 and clip_x2 > clip_x1:
        src_y1 = clip_y1 - y1
        src_x1 = clip_x1 - x1
        src_y2 = src_y1 + (clip_y2 - clip_y1)
        src_x2 = src_x1 + (clip_x2 - clip_x1)

        shadow_region = shadow_mask[src_y1:src_y2, src_x1:src_x2]
        shadow_vals = shadow_region * opacity

        for c in range(3):
            canvas_f[clip_y1:clip_y2, clip_x1:clip_x2, c] *= (1 - shadow_vals)

    canvas_bgr[:] = np.clip(canvas_f * 255, 0, 255).astype(np.uint8)
    return canvas_bgr


# =============================================================================
# MASK-BASED CORNER REFINEMENT
# =============================================================================

def find_exact_corners(mask, geometric_corners, search_radius=20):
    """Find pixel-precise corner positions using Harris on a binary mask."""
    mask_float = np.float32(mask) / 255.0
    harris = cv2.cornerHarris(mask_float, blockSize=3, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)

    h, w = mask.shape[:2]
    exact = np.zeros_like(geometric_corners, dtype=np.float64)

    for i in range(4):
        gx, gy = geometric_corners[i]
        ix, iy = int(round(gx)), int(round(gy))

        x1 = max(0, ix - search_radius)
        y1 = max(0, iy - search_radius)
        x2 = min(w, ix + search_radius + 1)
        y2 = min(h, iy + search_radius + 1)

        region = harris[y1:y2, x1:x2]

        if region.size == 0 or region.max() < 1e-6:
            exact[i] = geometric_corners[i]
            continue

        _, max_val, _, max_loc = cv2.minMaxLoc(region)
        exact[i, 0] = x1 + max_loc[0]
        exact[i, 1] = y1 + max_loc[1]

    return exact


def transform_corners(corners, M):
    """Apply a 3×3 perspective matrix to an array of (x, y) corners."""
    warped = np.zeros_like(corners, dtype=np.float64)
    for i in range(corners.shape[0]):
        pt = np.array([corners[i, 0], corners[i, 1], 1])
        result = M @ pt
        warped[i, 0] = result[0] / result[2]
        warped[i, 1] = result[1] / result[2]
    return warped


# =============================================================================
# SCENE GENERATION — Same background pipeline as detection/pose generators
# =============================================================================

def generate_scene(source_dir, render_size=FIDUCIAL_RENDER_SIZE):
    """Generate a single-photo scene with pixel-precise corner tracking.

    Uses the SAME background generation as detection/pose generators:
      random_base_background → apply_texture_overlay → shadow → photo composite

    Returns:
        scene (BGR, render_size × render_size): The composite image
        exact_corners (4, 2): Pixel-precise corner positions [LL, UL, UR, LR]
    """
    # Step 1: Create textured background (same as detection/pose generators)
    canvas = random_base_background(render_size, render_size)
    canvas = apply_texture_overlay(canvas)

    # Step 2: Load and prepare photo (glare already applied)
    photo_size = random.randint(FIDUCIAL_PHOTO_SIZE_MIN, FIDUCIAL_PHOTO_SIZE_MAX)
    aspect = random.uniform(0.7, 1.3)
    target_w = photo_size
    target_h = int(photo_size * aspect)

    photo, orig_w, orig_h, new_w, new_h = load_and_prepare_photo(
        source_dir, target_w, target_h)

    # Step 3: Rotate photo
    rotation = random.uniform(-FIDUCIAL_ROTATION_RANGE, FIDUCIAL_ROTATION_RANGE)
    photo_rot = rotate_photo(photo, rotation)
    ph_rot, pw_rot = photo_rot.shape[:2]

    # Step 4: Place photo near center (with offset for variety)
    cx = render_size / 2 + random.uniform(-FIDUCIAL_CENTER_OFFSET,
                                           FIDUCIAL_CENTER_OFFSET)
    cy = render_size / 2 + random.uniform(-FIDUCIAL_CENTER_OFFSET,
                                           FIDUCIAL_CENTER_OFFSET)

    # Step 5: Shadow (same as other generators)
    shadow_offset = random.randint(2, 5)
    angle = random.uniform(0, 2 * math.pi)
    offset_x = int(shadow_offset * math.cos(angle))
    offset_y = int(shadow_offset * math.sin(angle))

    canvas = apply_shadow_bgr(
        canvas, photo_rot, cx, cy,
        offset_x, offset_y,
        random.uniform(1.5, 3.0), random.uniform(0.15, 0.35),
        new_w, new_h, rotation,
    )

    # Step 6: Composite photo (pre-multiplied alpha — no dark halo)
    canvas, tl_x, tl_y, pw, ph = composite_photo_onto(canvas, photo_rot, cx, cy)

    # Step 7: Build binary photo mask at EXACT same placement
    mask = np.zeros((render_size, render_size), dtype=np.uint8)

    if pw > 0 and ph > 0:
        src_x1_m = max(0, -tl_x)
        src_y1_m = max(0, -tl_y)
        src_x2_m = photo_rot.shape[1] - max(0, tl_x + photo_rot.shape[1] - render_size)
        src_y2_m = photo_rot.shape[0] - max(0, tl_y + photo_rot.shape[0] - render_size)
        dst_x1_m = max(0, tl_x)
        dst_y1_m = max(0, tl_y)
        dst_x2_m = min(render_size, tl_x + photo_rot.shape[1])
        dst_y2_m = min(render_size, tl_y + photo_rot.shape[0])

        if dst_x2_m > dst_x1_m and dst_y2_m > dst_y1_m:
            alpha_crop = photo_rot[src_y1_m:src_y2_m, src_x1_m:src_x2_m, 3]
            mask[dst_y1_m:dst_y2_m, dst_x1_m:dst_x2_m] = np.where(
                alpha_crop > 128, 255, 0
            ).astype(np.uint8)

    # Step 8: Compute geometric corners (pre-warp)
    geometric_corners = get_rotated_polygon(new_w, new_h, cx, cy, rotation)

    # Step 9: Apply perspective warp to both scene and mask
    perspective_strength = random.uniform(0.0, 0.03)
    max_disp = int(min(render_size, render_size) * perspective_strength)

    if max_disp > 1 and random.random() < 0.7:
        w = h = render_size
        safety_margin = HALF_CROP + 50

        persp_M = None
        for _ in range(50):
            tl = (random.randint(-max_disp, 0), random.randint(-max_disp, 0))
            tr = (random.randint(0, max_disp), random.randint(-max_disp, 0))
            bl = (random.randint(-max_disp, 0), random.randint(0, max_disp))
            br = (random.randint(0, max_disp), random.randint(0, max_disp))

            src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            dst_pts = np.array([
                [tl[0], tl[1]],
                [w + tr[0], tr[1]],
                [w + br[0], h + br[1]],
                [bl[0], h + bl[1]],
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            all_safe = True
            for i in range(4):
                pt = np.array([geometric_corners[i, 0], geometric_corners[i, 1], 1])
                result = M @ pt
                wx = result[0] / result[2]
                wy = result[1] / result[2]
                if (wx < safety_margin or wx > w - safety_margin or
                        wy < safety_margin or wy > h - safety_margin):
                    all_safe = False
                    break

            if all_safe:
                persp_M = M
                break

        if persp_M is not None:
            canvas = cv2.warpPerspective(
                canvas, persp_M, (render_size, render_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(128, 128, 128),
            )
            mask = cv2.warpPerspective(
                mask, persp_M, (render_size, render_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            geometric_corners = transform_corners(geometric_corners, persp_M)

    # Step 10: Refine corners using Harris on the mask
    exact_corners = find_exact_corners(mask, geometric_corners, search_radius=20)

    return canvas, exact_corners


# =============================================================================
# CROP EXTRACTION — Random corner position, no border replication
# =============================================================================

def extract_corner_crops(scene, corners, crop_size=FIDUCIAL_CROP_SIZE):
    """Extract 640×640 corner crops from the scene.

    Each crop is labeled with its corner CLASS (UL=0, UR=1, LL=2, LR=3)
    so the model learns to both detect and classify the corner orientation.

    The corner vertex is NOT always centered in the crop — it's randomly
    offset so the model learns to find corners at various positions.

    Only extracts crops where the ENTIRE crop fits within the scene.

    Returns:
        list of dicts with image, corner_type, class_id, label, corner_x, corner_y, bbox
    """
    h, w = scene.shape[:2]
    half = crop_size // 2  # 320
    crops = []

    for idx, corner_name in enumerate(CORNER_NAMES):
        cx, cy = corners[idx]

        # Randomize where the corner appears within the crop
        jitter_x = random.uniform(-FIDUCIAL_CORNER_JITTER, FIDUCIAL_CORNER_JITTER)
        jitter_y = random.uniform(-FIDUCIAL_CORNER_JITTER, FIDUCIAL_CORNER_JITTER)

        crop_center_x = cx + jitter_x
        crop_center_y = cy + jitter_y

        # Check that the ENTIRE crop fits within the scene — no padding!
        x1 = int(round(crop_center_x - half))
        y1 = int(round(crop_center_y - half))
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue

        crop = scene[y1:y2, x1:x2].copy()

        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            continue

        # Corner vertex position in crop coordinates
        corner_in_crop_x = cx - x1
        corner_in_crop_y = cy - y1

        # Ensure corner is far enough from crop edge that the bbox
        # (centered on the corner) fits entirely within the crop.
        # Max bbox half-dimension = 0.45 * 640 / 2 = 144px.
        # We skip crops where clamping would shift the bbox center
        # away from the corner vertex, since the bbox center MUST
        # precisely mark the corner location for the fiducial model.
        max_bbox_half = int(FIDUCIAL_BBOX_FRAC_MAX * crop_size / 2) + 2  # +2 for rounding
        if (corner_in_crop_x < max_bbox_half or corner_in_crop_x > crop_size - max_bbox_half or
                corner_in_crop_y < max_bbox_half or corner_in_crop_y > crop_size - max_bbox_half):
            continue

        # Bounding box centered on the corner vertex — NO clamping.
        # The bbox center = corner position = precise corner location.
        # This is critical: the model predicts bbox center to locate corners.
        bbox_frac_w = random.uniform(FIDUCIAL_BBOX_FRAC_MIN, FIDUCIAL_BBOX_FRAC_MAX)
        bbox_frac_h = random.uniform(FIDUCIAL_BBOX_FRAC_MIN, FIDUCIAL_BBOX_FRAC_MAX)
        bbox_half_w = (bbox_frac_w * crop_size) / 2
        bbox_half_h = (bbox_frac_h * crop_size) / 2

        bbox_x1 = corner_in_crop_x - bbox_half_w
        bbox_y1 = corner_in_crop_y - bbox_half_h
        bbox_x2 = corner_in_crop_x + bbox_half_w
        bbox_y2 = corner_in_crop_y + bbox_half_h

        bbox_w = bbox_x2 - bbox_x1
        bbox_h = bbox_y2 - bbox_y1

        # YOLO detection label: class_id x_center y_center width height (normalized)
        # IMPORTANT: x_center and y_center MUST equal the corner position.
        # The whole point of the fiducial model is that detecting the bbox
        # IS finding the precise corner — the bbox center = corner vertex.
        class_id = CORNER_CLASS_IDS[corner_name]
        label_x = corner_in_crop_x / crop_size
        label_y = corner_in_crop_y / crop_size
        label_w = bbox_frac_w
        label_h = bbox_frac_h

        label = f"{class_id} {label_x:.6f} {label_y:.6f} {label_w:.6f} {label_h:.6f}"

        crops.append({
            'image': crop,
            'corner_type': corner_name,
            'class_id': class_id,
            'label': label,
            'corner_x': corner_in_crop_x,
            'corner_y': corner_in_crop_y,
            'bbox': (bbox_x1, bbox_y1, bbox_x2, bbox_y2),
        })

    return crops


def create_fiducial_debug_image(img, corner_x, corner_y, bbox_x1, bbox_y1,
                                bbox_x2, bbox_y2, corner_name, class_id):
    """Create debug image showing the corner bbox and vertex crosshair."""
    debug = img.copy()
    color = CORNER_COLORS.get(corner_name, (0, 255, 0))

    cv2.rectangle(debug,
                   (int(bbox_x1), int(bbox_y1)),
                   (int(bbox_x2), int(bbox_y2)),
                   color, 2)
    # Crosshair at exact corner position
    cv2.circle(debug, (int(corner_x), int(corner_y)), 5, (0, 0, 255), -1)
    cv2.line(debug, (int(corner_x) - 20, int(corner_y)),
             (int(corner_x) + 20, int(corner_y)), (0, 0, 255), 1)
    cv2.line(debug, (int(corner_x), int(corner_y) - 20),
             (int(corner_x), int(corner_y) + 20), (0, 0, 255), 1)
    cv2.putText(debug, f"{corner_name.upper()} (class {class_id})",
                (int(bbox_x1), int(bbox_y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return debug


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate fiducial corner detection training data (4-class single model)')
    parser.add_argument('--source', default='./images',
                        help='Directory containing source photos')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: <project_root>/data_fiducial)')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of example scenes')
    parser.add_argument('--mode', choices=['examples', 'batch'], default='examples',
                        help='Examples: debug images. Batch: train/val split.')
    parser.add_argument('--train-count', type=int, default=4000,
                        help='Training scenes (batch mode)')
    parser.add_argument('--val-count', type=int, default=1000,
                        help='Validation scenes (batch mode)')
    args = parser.parse_args()

    # Fixup source path
    if not Path(args.source).is_absolute():
        script_dir = Path(__file__).resolve().parent
        candidate = script_dir.parent / args.source
        if not Path(args.source).exists() and candidate.exists():
            args.source = str(candidate)

    if args.mode == 'batch':
        _batch_generate(args)
    else:
        _example_generate(args)


def _example_generate(args):
    """Generate example crops with debug overlays for visual verification."""
    output_base = Path(args.output) if args.output else Path('/tmp/fiducial_examples')
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} example fiducial corner crops (4-class single model)...")
    start = time.time()
    total_crops = 0
    total_scenes = 0
    skipped_scenes = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for i in range(args.count):
        try:
            scene, corners = generate_scene(args.source)
        except Exception as e:
            print(f"  Scene {i+1} failed: {e}")
            skipped_scenes += 1
            continue

        if scene is None:
            skipped_scenes += 1
            continue

        total_scenes += 1
        crops = extract_corner_crops(scene, corners)

        if not crops:
            skipped_scenes += 1
            continue

        for crop_data in crops:
            idx = total_crops
            ct = crop_data['corner_type']
            cid = crop_data['class_id']
            corner_x, corner_y = crop_data['corner_x'], crop_data['corner_y']
            bbox = crop_data['bbox']

            cv2.imwrite(str(output_base / f"crop_{idx + 1:04d}.jpg"),
                        crop_data['image'])

            debug = create_fiducial_debug_image(
                crop_data['image'], corner_x, corner_y,
                bbox[0], bbox[1], bbox[2], bbox[3],
                ct, cid)
            cv2.imwrite(str(output_base / f"crop_{idx + 1:04d}_debug.jpg"),
                        debug)

            with open(output_base / f"crop_{idx + 1:04d}.txt", 'w') as f:
                f.write(crop_data['label'] + '\n')

            total_crops += 1
            class_counts[cid] += 1

        if (i + 1) % 5 == 0:
            print(f"  {i + 1:2d}/{args.count} scenes, {total_crops} crops "
                  f"(UL={class_counts[0]}, UR={class_counts[1]}, "
                  f"LL={class_counts[2]}, LR={class_counts[3]})")

    elapsed = time.time() - start
    print(f"\n✅ Done in {elapsed:.1f}s")
    print(f"   Total: {total_crops} crops from {total_scenes} valid scenes "
          f"({skipped_scenes} skipped)")
    print(f"   Class distribution: UL={class_counts[0]}, UR={class_counts[1]}, "
          f"LL={class_counts[2]}, LR={class_counts[3]}")


def _batch_generate(args):
    """Generate train/val split dataset for the single 4-class fiducial model."""
    from datetime import datetime

    data_dir = Path(args.output) if args.output else Path(__file__).resolve().parent.parent / "data_fiducial"

    total = args.train_count + args.val_count

    # Create single dataset directory
    dirs = {
        'img_train': data_dir / "images" / "train",
        'img_val': data_dir / "images" / "val",
        'lbl_train': data_dir / "labels" / "train",
        'lbl_val': data_dir / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"Fiducial corner batch: {args.train_count} train + {args.val_count} val scenes")
    print(f"Output: {data_dir}")
    print(f"Classes: 0=UL, 1=UR, 2=LL, 3=LR")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()
    total_crops = {'train': 0, 'val': 0}
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    skipped = 0

    for i in range(total):
        is_train = i < args.train_count

        try:
            scene, corners = generate_scene(args.source)
        except Exception:
            skipped += 1
            continue

        if scene is None:
            skipped += 1
            continue

        crops = extract_corner_crops(scene, corners)
        if not crops:
            skipped += 1
            continue

        split = 'train' if is_train else 'val'

        for crop_data in crops:
            idx = total_crops[split]
            prefix = f"{split}_{idx + 1:06d}"

            img_dir = dirs[f'img_{split}']
            lbl_dir = dirs[f'lbl_{split}']

            cv2.imwrite(str(img_dir / f"{prefix}.jpg"), crop_data['image'])
            with open(lbl_dir / f"{prefix}.txt", 'w') as f:
                f.write(crop_data['label'] + '\n')

            total_crops[split] += 1
            class_counts[crop_data['class_id']] += 1

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i + 1:5d}/{total}] {elapsed / 60:.1f}m | {rate:.1f}/s | ETA {eta:.0f}m "
                  f"| train: {total_crops['train']} | val: {total_crops['val']} "
                  f"| UL={class_counts[0]} UR={class_counts[1]} "
                  f"LL={class_counts[2]} LR={class_counts[3]} "
                  f"| skipped: {skipped}")

    elapsed = time.time() - start
    print(f"\n✅ Complete! {total} scenes ({elapsed / 60:.1f}m, {skipped} skipped)")
    print(f"   train: {total_crops['train']} images")
    print(f"   val: {total_crops['val']} images")
    print(f"   Classes: UL={class_counts[0]}, UR={class_counts[1]}, "
          f"LL={class_counts[2]}, LR={class_counts[3]}")
    print(f"   Output: {data_dir}")


if __name__ == "__main__":
    main()