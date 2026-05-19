#!/usr/bin/env python3
"""
Photo Pose Detector — Fiducial Pose Data Generator
===================================================

Generates synthetic training images for a **multi-pose corner fiducial model**.
This model operates on 640×640 crops from the detection→pose pipeline and
detects visible photo boundary segments with their endpoints as keypoints.

Scene approach: photos are physically larger than the crop, arranged in a **grid
layout with gaps** on a virtual surface. The 640×640 canvas takes a crop near a
corner of one of the photos, so each photo shows at most 2 edges (max 1 corner),
and 30-70% of the canvas contains photos.

Scene distribution:
  - ~55%: 1 photo with 1 corner visible (2 edges meeting at L-shape)
  - ~25%: 2-4 photos, each with 1-2 edges visible, arranged in grid with gaps
  - ~15%: 1 photo with 2 corners visible (3 segments)
  - ~ 5%: No photo visible (background only, negative samples)

Single class: `photo_segment` (geometric grouping determines quadrilateral assignment)
Keypoints: 2 per instance (segment endpoints)
Flip index: [1, 0] (horizontal flip swaps kp0↔kp1)

Label format (11 columns per instance):
  class_id x_center y_center width height kp0x kp0y kp0v kp1x kp1y kp1v

Usage:
    python generate_fiducial_pose.py --mode examples --count 20 --source ./images
    python generate_fiducial_pose.py --mode batch --train-count 8000 --val-count 2000 --source ./images --output ../data_fiducial_pose
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

# =============================================================================
# Fiducial pose configuration
# =============================================================================

CANVAS_SIZE = 640
MIN_SEGMENT_LENGTH = 12
CORNER_VISIBILITY_MARGIN = 5

MODE_WEIGHTS = {
    'one_corner':     0.45,
    'grid':           0.35,
    'two_corners':    0.15,
    'no_photo':       0.05,
}

# Grid layout parameters
GRID_GAP_MIN = 8
GRID_GAP_MAX = 25
GRID_PHOTO_W_MIN = 0.8   # Photos are physically larger than the crop
GRID_PHOTO_W_MAX = 2.0
GRID_PHOTO_H_MIN = 0.8
GRID_PHOTO_H_MAX = 2.0
GRID_JITTER = 3           # Reduced to prevent overlaps
GRID_ROTATION_MAX = 3
MIN_BBOX_COVERAGE = 0.20  # Minimum coverage for modes with corners


# =============================================================================
# SEGMENT EXTRACTION
# =============================================================================

def clip_line_to_rect(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    """Clip a line segment to a rectangle using Liang-Barsky algorithm."""
    dx = x2 - x1
    dy = y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
    t_enter = 0.0
    t_exit = 1.0
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return None
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                t_enter = max(t_enter, t)
            else:
                t_exit = min(t_exit, t)
    if t_enter > t_exit:
        return None
    cx1 = x1 + t_enter * dx
    cy1 = y1 + t_enter * dy
    cx2 = x1 + t_exit * dx
    cy2 = y1 + t_exit * dy
    return (cx1, cy1, cx2, cy2, t_enter, t_exit)


def extract_segments_from_photo(warped_corners, canvas_size,
                                 min_length=MIN_SEGMENT_LENGTH,
                                 corner_margin=CORNER_VISIBILITY_MARGIN):
    """Extract visible photo boundary segments from a photo's corner positions."""
    xmin, ymin = 0, 0
    xmax, ymax = canvas_size, canvas_size
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edge_names = ['left', 'top', 'right', 'bottom']
    segments = []

    for edge_idx, (ci, cj) in enumerate(edges):
        x1, y1 = warped_corners[ci]
        x2, y2 = warped_corners[cj]
        result = clip_line_to_rect(x1, y1, x2, y2, xmin, ymin, xmax, ymax)
        if result is None:
            continue
        cx1, cy1, cx2, cy2, t_enter, t_exit = result
        seg_len = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
        if seg_len < min_length:
            continue

        segment_points = np.array([[cx1, cy1], [cx2, cy2]], dtype=np.float32)
        corner_kps = []
        corner_indices = []

        for endpoint_idx in range(2):
            if endpoint_idx == 0:
                orig_corner_idx = ci
                orig_corner = warped_corners[ci]
                t_boundary = t_enter
            else:
                orig_corner_idx = cj
                orig_corner = warped_corners[cj]
                t_boundary = t_exit

            near_orig = (endpoint_idx == 0 and t_boundary < 0.01) or \
                       (endpoint_idx == 1 and t_boundary > 0.99)

            if near_orig:
                cx_c, cy_c = orig_corner
                visible = (corner_margin <= cx_c <= canvas_size - corner_margin and
                          corner_margin <= cy_c <= canvas_size - corner_margin)
                if visible:
                    corner_kps.append((cx_c, cy_c, 2))
                else:
                    corner_kps.append((cx_c, cy_c, 0))
                corner_indices.append(orig_corner_idx)
            else:
                bx, by = (cx1, cy1) if endpoint_idx == 0 else (cx2, cy2)
                corner_kps.append((bx, by, 2))
                corner_indices.append(None)

        segments.append({
            'points': segment_points,
            'corner_kps': corner_kps,
            'corner_indices': corner_indices,
            'edge_name': edge_names[edge_idx],
            'length': seg_len,
        })
    return segments


def count_unique_visible_corners(segments):
    """Count unique visible corner indices (v=2) from segment data."""
    visible = set()
    for s in segments:
        for kp_idx, c_idx in enumerate(s['corner_indices']):
            if c_idx is not None and s['corner_kps'][kp_idx][2] == 2:
                visible.add(c_idx)
    return len(visible)


def compute_bbox_coverage(segments, canvas_size):
    """Compute bounding box coverage of segments as fraction of canvas area."""
    if not segments:
        return 0.0
    all_pts = np.concatenate([s['points'] for s in segments])
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)
    bbox_area = (max_x - min_x) * (max_y - min_y)
    return bbox_area / (canvas_size * canvas_size)


def segment_to_yolo_label(segment, canvas_size):
    """Convert a segment dict to a YOLO-pose label line (11 columns)."""
    points = segment['points']
    kps = segment['corner_kps']
    x1, y1 = points[0]
    x2, y2 = points[1]
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    min_box = 8.0
    if max_x - min_x < min_box:
        cx = (min_x + max_x) / 2
        min_x, max_x = cx - min_box / 2, cx + min_box / 2
    if max_y - min_y < min_box:
        cy = (min_y + max_y) / 2
        min_y, max_y = cy - min_box / 2, cy + min_box / 2
    x_center = max(0.0, min(1.0, ((min_x + max_x) / 2) / canvas_size))
    y_center = max(0.0, min(1.0, ((min_y + max_y) / 2) / canvas_size))
    width = max(0.01, min(1.0, (max_x - min_x) / canvas_size))
    height = max(0.01, min(1.0, (max_y - min_y) / canvas_size))
    kp0x = max(0.0, min(1.0, kps[0][0] / canvas_size))
    kp0y = max(0.0, min(1.0, kps[0][1] / canvas_size))
    kp1x = max(0.0, min(1.0, kps[1][0] / canvas_size))
    kp1y = max(0.0, min(1.0, kps[1][1] / canvas_size))
    return (f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} "
            f"{kp0x:.6f} {kp0y:.6f} {kps[0][2]} {kp1x:.6f} {kp1y:.6f} {kps[1][2]}")


# =============================================================================
# MILD PERSPECTIVE WARP
# =============================================================================

def apply_perspective_mild(canvas):
    """Apply a mild perspective warp to the canvas image."""
    h, w = canvas.shape[:2]
    max_disp = int(min(w, h) * 0.02)
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array([
        [random.randint(-max_disp, max_disp), random.randint(-max_disp, max_disp)],
        [w + random.randint(-max_disp, max_disp), random.randint(-max_disp, max_disp)],
        [w + random.randint(-max_disp, max_disp), h + random.randint(-max_disp, max_disp)],
        [random.randint(-max_disp, max_disp), h + random.randint(-max_disp, max_disp)],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR) if canvas.shape[2] == 4 else canvas.copy()
    warped = cv2.warpPerspective(canvas_bgr, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(128, 128, 128))
    return warped, M


def _transform_corners(corners, M):
    """Apply a 3×3 perspective matrix to corner coordinates."""
    warped = np.zeros_like(corners, dtype=np.float64)
    for i in range(corners.shape[0]):
        pt = np.array([corners[i, 0], corners[i, 1], 1.0])
        result = M @ pt
        warped[i, 0] = result[0] / result[2]
        warped[i, 1] = result[1] / result[2]
    return warped.astype(np.float32)


# =============================================================================
# PHOTO COMPOSITING
# =============================================================================

def _place_photo_on_canvas(canvas, source_dir, center_x, center_y, width, height, rotation):
    """Place a single photo on the canvas with effects."""
    photo, orig_w, orig_h, new_w, new_h = load_and_prepare_photo(
        source_dir, width, height)
    photo = rotate_photo(photo, rotation)
    shadow_offset = random.randint(2, 5)
    angle = random.uniform(0, 2 * math.pi)
    offset_x = int(shadow_offset * math.cos(angle))
    offset_y = int(shadow_offset * math.sin(angle))
    canvas = apply_photo_shadow(canvas, photo, center_x, center_y,
                                offset_x, offset_y,
                                random.uniform(1.5, 3.0), random.uniform(0.15, 0.35),
                                new_w, new_h, rotation)
    canvas = composite_photo_at_center(canvas, photo, center_x, center_y)
    corners = get_rotated_polygon(new_w, new_h, center_x, center_y, rotation)
    return canvas, corners, new_w, new_h


# =============================================================================
# GRID-BASED SCENE GENERATION
# =============================================================================

def _photo_rects_overlap(photos, margin=0):
    """Check if any pair of photos overlap (axis-aligned bounding boxes)."""
    for i in range(len(photos)):
        pi = photos[i]
        pi_left = pi['center_x'] - pi['width'] / 2 - margin
        pi_right = pi['center_x'] + pi['width'] / 2 + margin
        pi_top = pi['center_y'] - pi['height'] / 2 - margin
        pi_bottom = pi['center_y'] + pi['height'] / 2 + margin
        for j in range(i + 1, len(photos)):
            pj = photos[j]
            pj_left = pj['center_x'] - pj['width'] / 2 - margin
            pj_right = pj['center_x'] + pj['width'] / 2 + margin
            pj_top = pj['center_y'] - pj['height'] / 2 - margin
            pj_bottom = pj['center_y'] + pj['height'] / 2 + margin
            x_overlap = max(0, min(pi_right, pj_right) - max(pi_left, pj_left))
            y_overlap = max(0, min(pi_bottom, pj_bottom) - max(pi_top, pj_top))
            if x_overlap > 0 and y_overlap > 0:
                return True
    return False


def build_photo_grid(canvas_size, n_photos):
    """Build a grid of non-overlapping photos with gaps.

    Photos are physically larger than the canvas (1-2x), arranged with gaps.
    Returns list of photo placements and the gap size (or None if overlap-free
    layout couldn't be found).
    """
    for _attempt in range(20):
        gap = random.randint(GRID_GAP_MIN, GRID_GAP_MAX)

        # Photos are larger than canvas (physically realistic)
        base_w = random.randint(int(canvas_size * GRID_PHOTO_W_MIN),
                                int(canvas_size * GRID_PHOTO_W_MAX))
        base_h = random.randint(int(canvas_size * GRID_PHOTO_H_MIN),
                                int(canvas_size * GRID_PHOTO_H_MAX))

        # Arrange in rows (wrapping)
        cols = max(1, math.ceil(math.sqrt(n_photos * base_w / base_h)))
        rows = max(1, math.ceil(n_photos / cols))

        step_x = base_w + gap
        step_y = base_h + gap

        photos = []
        for idx in range(n_photos):
            row = idx // cols
            col = idx % cols
            x = col * step_x + random.randint(-GRID_JITTER, GRID_JITTER)
            y = row * step_y + random.randint(-GRID_JITTER, GRID_JITTER)
            w = max(200, base_w + random.randint(-30, 30))
            h = max(200, base_h + random.randint(-30, 30))
            rotation = random.uniform(-GRID_ROTATION_MAX, GRID_ROTATION_MAX)
            cx = x + w / 2
            cy = y + h / 2
            photos.append({
                'center_x': cx, 'center_y': cy,
                'width': w, 'height': h,
                'rotation': rotation,
            })

        # Ensure no overlaps between photos
        if not _photo_rects_overlap(photos, margin=5):
            return photos, gap

    # Fallback: generate with no jitter and larger gaps
    gap = GRID_GAP_MAX
    base_w = int(canvas_size * 1.2)
    base_h = int(canvas_size * 1.2)
    cols = max(1, math.ceil(math.sqrt(n_photos * base_w / base_h)))
    step_x = base_w + gap
    step_y = base_h + gap

    photos = []
    for idx in range(n_photos):
        row = idx // cols
        col = idx % cols
        x = col * step_x
        y = row * step_y
        photos.append({
            'center_x': x + base_w / 2, 'center_y': y + base_h / 2,
            'width': base_w, 'height': base_h,
            'rotation': 0,
        })
    return photos, gap


def choose_crop_offset(photos, canvas_size, target_mode):
    """Choose a crop offset so the 640×640 canvas captures the desired scene.

    The crop is a window into the photo grid surface. The key insight:
    to show exactly 1-2 edges per photo, the crop must be positioned near
    a CORNER of one of the photos, not near the center.
    """
    for _ in range(100):
        # Pick primary photo
        primary = random.choice(photos)
        pw, ph = primary['width'], primary['height']
        pcx, pcy = primary['center_x'], primary['center_y']
        prot = primary['rotation']

        corners = get_rotated_polygon(pw, ph, pcx, pcy, prot)

        if target_mode == 'one_corner':
            # Position crop so exactly 1 corner of primary photo is in-frame
            corner_idx = random.choice([0, 1, 2, 3])
            target_corner = corners[corner_idx]
            # Place corner at a position in the canvas that gives good coverage
            target_x = random.triangular(canvas_size * 0.15, canvas_size * 0.85,
                                         canvas_size * 0.5)
            target_y = random.triangular(canvas_size * 0.15, canvas_size * 0.85,
                                         canvas_size * 0.5)
            offset_x = target_corner[0] - target_x
            offset_y = target_corner[1] - target_y

        elif target_mode == 'two_corners':
            # Position crop so 2 adjacent corners of primary photo are in-frame
            edge = random.choice([(0, 1), (1, 2), (2, 3), (3, 0)])
            ci, cj = edge
            mid_x = (corners[ci][0] + corners[cj][0]) / 2
            mid_y = (corners[ci][1] + corners[cj][1]) / 2
            target_x = canvas_size / 2 + random.uniform(-canvas_size * 0.15,
                                                          canvas_size * 0.15)
            target_y = canvas_size / 2 + random.uniform(-canvas_size * 0.15,
                                                          canvas_size * 0.15)
            offset_x = mid_x - target_x
            offset_y = mid_y - target_y

        elif target_mode == 'grid':
            # Position near a junction where corners from multiple photos are close
            # Strategy: find corner positions from all photos, cluster them,
            # and prefer offsets that place multiple corners near each other
            all_corners = []
            for pi, ph in enumerate(photos):
                ph_corners = get_rotated_polygon(ph['width'], ph['height'],
                                                 ph['center_x'], ph['center_y'],
                                                 ph['rotation'])
                for ci, c in enumerate(ph_corners):
                    all_corners.append((pi, ci, c[0], c[1]))

            if len(all_corners) >= 2 and random.random() < 0.7:
                # Pick a corner, then find the nearest corner from a different photo
                idx1 = random.randint(0, len(all_corners) - 1)
                pi1, ci1, cx1, cy1 = all_corners[idx1]
                best_dist = float('inf')
                best_cx, best_cy = cx1, cy1
                for pi2, ci2, cx2, cy2 in all_corners:
                    if pi2 == pi1:
                        continue
                    d = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
                    if d < best_dist:
                        best_dist = d
                        best_cx, best_cy = cx2, cy2
                # If nearest corner is close (within 2x canvas), use midpoint
                if best_dist < canvas_size * 2:
                    mid_x = (cx1 + best_cx) / 2
                    mid_y = (cy1 + best_cy) / 2
                    target_x = random.triangular(canvas_size * 0.2, canvas_size * 0.8,
                                                 canvas_size * 0.5)
                    target_y = random.triangular(canvas_size * 0.2, canvas_size * 0.8,
                                                 canvas_size * 0.5)
                    offset_x = mid_x - target_x
                    offset_y = mid_y - target_y
                else:
                    # Fall back to single corner positioning
                    corner_idx = random.choice([0, 1, 2, 3])
                    target_corner = corners[corner_idx]
                    target_x = random.triangular(canvas_size * 0.2, canvas_size * 0.8,
                                                 canvas_size * 0.5)
                    target_y = random.triangular(canvas_size * 0.2, canvas_size * 0.8,
                                                 canvas_size * 0.5)
                    offset_x = target_corner[0] - target_x
                    offset_y = target_corner[1] - target_y
            else:
                # Single corner positioning
                corner_idx = random.choice([0, 1, 2, 3])
                target_corner = corners[corner_idx]
                target_x = random.triangular(canvas_size * 0.2, canvas_size * 0.8,
                                             canvas_size * 0.5)
                target_y = random.triangular(canvas_size * 0.2, canvas_size * 0.8,
                                             canvas_size * 0.5)
                offset_x = target_corner[0] - target_x
                offset_y = target_corner[1] - target_y

        else:
            offset_x, offset_y = 0, 0

        # Validate: check that no photo in the crop has more than 2 visible edges
        valid = True
        for photo in photos:
            cx = photo['center_x'] - offset_x
            cy = photo['center_y'] - offset_y
            shifted = get_rotated_polygon(photo['width'], photo['height'],
                                          cx, cy, photo['rotation'])
            segs = extract_segments_from_photo(shifted, canvas_size)
            n_segs = len(segs)
            if n_segs > 2:  # Max 2 visible edges per photo
                valid = False
                break

        if valid:
            return offset_x, offset_y

    # Fallback: offset 0 (shows center of grid)
    return 0, 0


# =============================================================================
# IMAGE GENERATION
# =============================================================================

def generate_fiducial_pose_image(source_dir, force_mode=None, retry_count=0):
    """Generate a single fiducial pose training image."""
    if retry_count > 10:
        cs = CANVAS_SIZE
        canvas = random_base_background(cs, cs)
        canvas = apply_texture_overlay(canvas)
        if canvas.shape[2] == 4:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        return canvas, [], [], 'no_photo'

    canvas_size = CANVAS_SIZE

    if force_mode:
        mode = force_mode
    else:
        modes = list(MODE_WEIGHTS.keys())
        weights = list(MODE_WEIGHTS.values())
        mode = random.choices(modes, weights=weights, k=1)[0]

    # Determine grid size based on mode
    if mode == 'one_corner':
        n_photos = random.choices([1, 2], weights=[6, 4], k=1)[0]
    elif mode == 'two_corners':
        n_photos = random.choices([1, 2], weights=[7, 3], k=1)[0]
    elif mode == 'grid':
        n_photos = random.choices([2, 3, 4], weights=[3, 5, 2], k=1)[0]
    else:
        n_photos = 0

    if n_photos > 0:
        photos, gap = build_photo_grid(canvas_size, n_photos)
        offset_x, offset_y = choose_crop_offset(photos, canvas_size, mode)

        canvas = random_base_background(canvas_size, canvas_size)
        canvas = apply_texture_overlay(canvas)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
        canvas[:, :, 3] = 255

        all_corners_before_warp = []
        for photo in photos:
            cx = photo['center_x'] - offset_x
            cy = photo['center_y'] - offset_y
            w, h = photo['width'], photo['height']
            # Skip photos entirely off-screen
            if (cx + w / 2 < -50 or cx - w / 2 > canvas_size + 50 or
                cy + h / 2 < -50 or cy - h / 2 > canvas_size + 50):
                continue
            canvas, corners, new_w, new_h = _place_photo_on_canvas(
                canvas, source_dir, cx, cy, w, h, photo['rotation'])
            all_corners_before_warp.append(corners)

        if all_corners_before_warp:
            canvas, persp_M = apply_perspective_mild(canvas)
        else:
            if canvas.shape[2] == 4:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
            persp_M = np.eye(3)

        if canvas.shape[:2] != (canvas_size, canvas_size):
            canvas = canvas[:canvas_size, :canvas_size]
        if canvas.shape[2] == 4:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)

        all_segments = []
        all_labels = []
        per_photo_seg_counts = []  # Track segment count per photo
        for corners in all_corners_before_warp:
            warped_corners = _transform_corners(corners, persp_M)
            segments = extract_segments_from_photo(warped_corners, canvas_size)
            per_photo_seg_counts.append(len(segments))
            for seg in segments:
                label = segment_to_yolo_label(seg, canvas_size)
                all_labels.append(label)
            all_segments.extend(segments)

    else:
        canvas = random_base_background(canvas_size, canvas_size)
        canvas = apply_texture_overlay(canvas)
        if canvas.shape[2] == 4:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        all_segments = []
        all_labels = []
        per_photo_seg_counts = []

    # Validate result
    if mode != 'no_photo':
        n_segs = len(all_segments)
        n_corners = count_unique_visible_corners(all_segments)
        coverage = compute_bbox_coverage(all_segments, canvas_size)

        valid = True
        if mode == 'one_corner':
            if n_segs != 2 or n_corners != 1:
                valid = False
            elif coverage < MIN_BBOX_COVERAGE:
                valid = False
        elif mode == 'two_corners':
            if n_corners != 2 or n_segs < 2:
                valid = False
            elif coverage < MIN_BBOX_COVERAGE:
                valid = False
        elif mode == 'grid':
            if n_segs < 2:
                valid = False
            elif coverage < MIN_BBOX_COVERAGE:
                valid = False
            # Check no photo has >2 visible segments
            if per_photo_seg_counts and max(per_photo_seg_counts) > 2:
                valid = False

        if not valid:
            return generate_fiducial_pose_image(source_dir, force_mode=mode,
                                                  retry_count=retry_count + 1)

    return canvas, all_labels, all_segments, mode


# =============================================================================
# DEBUG VISUALIZATION
# =============================================================================

def create_fiducial_debug_image(img, segments, canvas_size=CANVAS_SIZE):
    """Create debug image with segment overlays."""
    debug = img.copy()
    edge_colors = [(0, 200, 0), (200, 0, 200), (0, 150, 255), (255, 100, 0)]

    for i, seg in enumerate(segments):
        pts = seg['points'].astype(np.int32)
        color = edge_colors[i % len(edge_colors)]
        cv2.line(debug, tuple(pts[0]), tuple(pts[1]), color, 3)

        for j, kp in enumerate(seg['corner_kps']):
            x, y, vis = int(kp[0]), int(kp[1]), kp[2]
            if seg['corner_indices'][j] is not None:
                if vis == 2:
                    cv2.circle(debug, (x, y), 8, (0, 255, 0), -1)
                    cv2.circle(debug, (x, y), 8, (255, 255, 255), 2)
                else:
                    cv2.circle(debug, (x, y), 6, (128, 128, 128), -1)
            else:
                cv2.circle(debug, (x, y), 6, (0, 0, 255), -1)
                cv2.circle(debug, (x, y), 6, (255, 255, 255), 2)

        mid_x = (pts[0][0] + pts[1][0]) // 2
        mid_y = (pts[0][1] + pts[1][1]) // 2
        cv2.putText(debug, seg['edge_name'][:3], (mid_x - 10, mid_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return debug


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate fiducial pose training data (multi-segment detection)')
    parser.add_argument('--source', default='./images',
                        help='Directory containing source photos')
    parser.add_argument('--output', default='./data/examples_fiducial',
                        help='Output directory')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--mode', choices=['examples', 'batch'], default='examples')
    parser.add_argument('--train-count', type=int, default=8000)
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
    print(f"Generating {args.count} fiducial pose example images...")
    start = time.time()
    mode_counts = {}

    for i in range(args.count):
        img, labels, segments, mode = generate_fiducial_pose_image(
            args.source, force_mode=args.force_mode)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        cv2.imwrite(str(output_dir / f"fid_pose_{i + 1:02d}.jpg"), img)
        cv2.imwrite(str(output_dir / f"fid_pose_{i + 1:02d}_debug.jpg"),
                    create_fiducial_debug_image(img, segments))
        with open(output_dir / f"fid_pose_{i + 1:02d}.txt", 'w') as f:
            f.write('\n'.join(labels) if labels else '')

        n_segs = len(segments)
        n_corners = count_unique_visible_corners(segments)
        coverage = compute_bbox_coverage(segments, CANVAS_SIZE) * 100
        print(f"  {i + 1:2d}/{args.count}: mode={mode}, segs={n_segs}, corners={n_corners}, coverage={coverage:.0f}%")

    elapsed = time.time() - start
    print(f"\nDone! {args.count} images in {elapsed:.1f}s")
    print(f"Output: {output_dir.absolute()}")
    print(f"Mode distribution: {mode_counts}")


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

    print(f"Fiducial Pose batch: {args.train_count} train + {args.val_count} val = {total}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()
    total_segments = 0
    mode_counts = {}

    for i in range(total):
        is_train = i < args.train_count
        img, labels, segments, mode = generate_fiducial_pose_image(args.source)
        total_segments += len(segments)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        if is_train:
            prefix = f"train_{i + 1:06d}"
            img_dir, lbl_dir = dirs['img_train'], dirs['lbl_train']
        else:
            idx = i - args.train_count + 1
            prefix = f"val_{idx:06d}"
            img_dir, lbl_dir = dirs['img_val'], dirs['lbl_val']

        cv2.imwrite(str(img_dir / f"{prefix}.jpg"), img)
        with open(lbl_dir / f"{prefix}.txt", 'w') as f:
            f.write('\n'.join(labels) if labels else '')

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i + 1:5d}/{total}] {elapsed / 60:.1f}m | "
                  f"{rate:.1f}/s | ETA {eta:.0f}m | segs: {total_segments}")

    elapsed = time.time() - start
    print(f"\nComplete! {total} images ({elapsed / 60:.1f}m)")
    print(f"   Train: {args.train_count} | Val: {args.val_count}")
    print(f"   Total segments: {total_segments} (avg {total_segments / total:.1f}/img)")
    print(f"   Mode distribution: {mode_counts}")
    print(f"Output: {base_dir.absolute()}")


if __name__ == "__main__":
    main()