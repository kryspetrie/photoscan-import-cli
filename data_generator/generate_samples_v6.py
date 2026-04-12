#!/usr/bin/env python3
"""
Photo Pose Detector - Fast Synthetic Data Generator (v4)

IMPROVEMENTS OVER v3:
1. Larger photos (~40% canvas width) with less size variation (±5%)
2. True spiral packing from center with polygon-based collision (not rectangular BB)
3. Proper margin from edges - no edge-extension hack
4. Crop to content bounds after global perspective with clean margin

CORRECT ARCHITECTURE:
1. Pack photos FLAT (rectangles) with ±30° rotation on a flat background
2. Apply ONE global perspective warp to the ENTIRE composite at the end
3. Crop to content bounds (photos never touch final image edge)
"""

import signal
import sys
import os
from pathlib import Path
import traceback
import time
import math
import random

import numpy as np
import cv2
from PIL import Image


# =============================================================================
# POLYGON UTILITIES
# =============================================================================

def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """
    Get the actual rotated polygon corners for a photo.
    
    Args:
        width, height: Original photo dimensions
        center_x, center_y: Center position
        rotation: Rotation angle in degrees
    
    Returns:
        4x2 numpy array of corner coordinates [TL, TR, BR, BL]
    """
    # Photo corners relative to center
    hw, hh = width / 2, height / 2
    corners = np.array([
        [-hw, -hh],  # TL
        [ hw, -hh],  # TR
        [ hw,  hh],  # BR
        [-hw,  hh]   # BL
    ], dtype=np.float32)
    
    # Apply rotation
    angle_rad = math.radians(rotation)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    rotated = np.array([
        [
            corners[i, 0] * cos_a - corners[i, 1] * sin_a + center_x,
            corners[i, 0] * sin_a + corners[i, 1] * cos_a + center_y
        ]
        for i in range(4)
    ], dtype=np.float32)
    
    return rotated


def polygons_overlap(poly1, poly2, buffer_pixels=0):
    """
    Check if two convex polygons overlap.
    Uses Separating Axis Theorem (SAT) for accurate collision detection.
    
    Args:
        buffer_pixels: Extra margin to require gap between polygons (default 0)
    """
    def get_axes(poly):
        """Get potential separating axes from polygon edges."""
        axes = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 0:
                axes.append(normal / norm)
        return axes
    
    def project_poly(poly, axis):
        dots = [np.dot(p, axis) for p in poly]
        return min(dots), max(dots)
    
    poly1 = np.array(poly1, dtype=np.float32)
    poly2 = np.array(poly2, dtype=np.float32)
    
    axes = get_axes(poly1) + get_axes(poly2)
    
    for axis in axes:
        min1, max1 = project_poly(poly1, axis)
        min2, max2 = project_poly(poly2, axis)
        
        # If intervals don't overlap (with buffer), polygons are separated
        if max1 + buffer_pixels < min2 or max2 + buffer_pixels < min1:
            return False
    
    return True


def get_polygon_extent(poly):
    """Get the maximum extent (radius) of polygon from its center."""
    cx = sum(p[0] for p in poly) / 4
    cy = sum(p[1] for p in poly) / 4
    max_dist = max(math.sqrt((p[0] - cx)**2 + (p[1] - cy)**2) for p in poly)
    return max_dist


def get_polygon_bounds(poly):
    """Get bounding box of polygon."""
    poly = np.array(poly)
    return (
        min(p[0] for p in poly),
        min(p[1] for p in poly),
        max(p[0] for p in poly),
        max(p[1] for p in poly)
    )


def pack_photos_single_attempt(canvas_w, canvas_h, num_photos, base_w, base_h, edge_margin, size_variation, seed):
    """Radial packing with pixel-based collision detection."""
    import random as rnd
    rnd.seed(seed)
    
    # 1-bit occupancy mask (0=background, 255=photo placed)
    occupancy = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    cx, cy = canvas_w / 2, canvas_h / 2
    
    # Generate candidates in spiral from center
    candidates = []
    golden_angle = 137.508
    max_radius = min(canvas_w, canvas_h) * 0.8
    
    for i in range(num_photos * 10 + 50):
        r = max_radius * (i / (num_photos * 10 + 50)) ** 0.7
        angle = math.radians(i * golden_angle)
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        candidates.append((float(x), float(y)))
    
    candidates.sort(key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2)
    
    # Photo configs
    photo_configs = []
    for _ in range(num_photos):
        scale = rnd.uniform(1 - size_variation, 1 + size_variation)
        width = int(base_w * scale)
        height = int(base_h * scale)
        rotation = rnd.uniform(-10, 10)
        photo_configs.append((width, height, rotation))
    
    photo_configs.sort(key=lambda c: c[0] * c[1], reverse=True)
    
    placements = []
    
    def create_photo_mask(width, height, center_x, center_y, rotation):
        """
        Create 1-bit mask of photo using polygon fill.
        Returns mask and bounding box for efficient collision checking.
        """
        poly = get_rotated_polygon(width, height, center_x, center_y, rotation)
        
        # Get bounding box of polygon
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        min_x, max_x = int(min(xs)), int(max(xs)) + 1
        min_y, max_y = int(min(ys)), int(max(ys)) + 1
        
        # Clamp to canvas
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(canvas_w, max_x)
        max_y = min(canvas_h, max_y)
        
        # Create small mask just for the bounding box
        bb_w = max_x - min_x
        bb_h = max_y - min_y
        
        if bb_w <= 0 or bb_h <= 0:
            return None, (min_x, min_y, max_x, max_y)
        
        small_mask = np.zeros((bb_h, bb_w), dtype=np.uint8)
        
        # Translate polygon points to bounding box coordinates
        pts = np.array([[int(p[0]) - min_x, int(p[1]) - min_y] for p in poly], dtype=np.int32)
        cv2.fillPoly(small_mask, [pts], 255)
        
        return small_mask, (min_x, min_y, max_x, max_y)
    
    def check_pixel_collision(bb_x, bb_y, small_mask, overlap_threshold=5):
        """
        Check collision using pixel-based overlap detection on bounding box region.
        
        overlap_threshold: max number of overlapping pixels allowed (default 50)
        """
        if small_mask is None:
            return True  # Reject if mask invalid
        
        bb_h, bb_w = small_mask.shape
        x2 = min(bb_x + bb_w, canvas_w)
        y2 = min(bb_y + bb_h, canvas_h)
        
        # Get the region of occupancy we need to check
        occ_region = occupancy[bb_y:y2, bb_x:x2]
        new_region = small_mask[:y2-bb_y, :x2-bb_x]
        
        # Bitwise AND to find overlap
        overlap = cv2.bitwise_and(occ_region, new_region)
        overlap_pixels = np.count_nonzero(overlap)
        
        return overlap_pixels > overlap_threshold
    
    def place_photo_mask(bb_x, bb_y, small_mask):
        """Add photo to occupancy mask within bounding box region."""
        if small_mask is None:
            return
        bb_h, bb_w = small_mask.shape
        x2 = min(bb_x + bb_w, canvas_w)
        y2 = min(bb_y + bb_h, canvas_h)
        new_region = small_mask[:y2-bb_y, :x2-bb_x]
        
        # Use bitwise OR to add to occupancy
        occupancy[bb_y:y2, bb_x:x2] = cv2.bitwise_or(occupancy[bb_y:y2, bb_x:x2], new_region)
    
    for photo_idx, (width, height, rotation) in enumerate(photo_configs):
        placed = False
        
        # Try rotations
        if rnd.random() < 0.4:
            angles_to_try = [rotation, 0, 90, -90]
        else:
            angles_to_try = [0, rotation, 90, -90]
        
        for angle in angles_to_try:
            if placed:
                break
            for shrink in [1.0, 0.85, 0.70, 0.55, 0.45]:
                if placed:
                    break
                w_s = int(width * shrink)
                h_s = int(height * shrink)
                
                for px, py in candidates:
                    cx_pos, cy_pos = px, py
                    
                    # Bounds check using actual polygon corners
                    poly_test = get_rotated_polygon(w_s, h_s, cx_pos, cy_pos, angle)
                    bx1 = min(p[0] for p in poly_test)
                    by1 = min(p[1] for p in poly_test)
                    bx2 = max(p[0] for p in poly_test)
                    by2 = max(p[1] for p in poly_test)
                    
                    # Strict bounds check - photo must be fully inside
                    if bx1 < edge_margin or by1 < edge_margin:
                        continue
                    if bx2 > canvas_w - edge_margin or by2 > canvas_h - edge_margin:
                        continue
                    
                    # Create 1-bit mask for this photo
                    new_mask, (bb_x, bb_y, _, _) = create_photo_mask(w_s, h_s, cx_pos, cy_pos, angle)
                    
                    # Pixel-based collision check using bitwise AND
                    if check_pixel_collision(bb_x, bb_y, new_mask):
                        continue
                    
                    # Place photo (add to occupancy)
                    place_photo_mask(bb_x, bb_y, new_mask)
                    
                    shadow_params = {
                        'offset_x': rnd.choice([-1, 1]) * rnd.randint(0, 20),
                        'offset_y': rnd.choice([-1, 1]) * rnd.randint(0, 20),
                        'blur_sigma': rnd.uniform(8, 20),
                        'opacity': rnd.choice([rnd.uniform(0.15, 0.25), rnd.uniform(0.45, 0.60)])
                    }
                    
                    placements.append({
                        'x': cx_pos - w_s / 2,
                        'y': cy_pos - h_s / 2,
                        'width': w_s,
                        'height': h_s,
                        'rotation': angle,
                        'center_x': cx_pos,
                        'center_y': cy_pos,
                        'polygon': poly_test,
                        'shadow_params': shadow_params
                    })
                    placed = True
                    break
    
    # Add circle_radius for compatibility
    for p in placements:
        p['circle_radius'] = math.sqrt(p['width']**2 + p['height']**2) / 2 + 5
    
    total_area = sum(p['width'] * p['height'] for p in placements)
    return placements, total_area


def spiral_pack_photos(canvas_w, canvas_h, num_photos=None):
    """
    Pack photos trying multiple configurations and keeping the best one.
    Uses radial placement with 20 packing attempts.
    """
    if num_photos is None:
        num_photos = random.randint(5, 8)
    
    # Base photo: 30% of canvas width (large photos)
    base_w = int(canvas_w * 0.30)
    base_h = int(base_w * 0.75)
    size_variation = 0.10
    edge_margin = 160  # Safe margin to prevent photos going outside after perspective
    
    # Try 20 different configurations and keep the best (most photo area)
    best_placements = []
    best_area = 0
    
    for attempt in range(20):
        placements, total_area = pack_photos_single_attempt(
            canvas_w, canvas_h, num_photos, base_w, base_h, 
            edge_margin, size_variation, seed=attempt * 1000 + random.randint(0, 999)
        )
        
        if total_area > best_area:
            best_area = total_area
            best_placements = placements
    
    return best_placements

# =============================================================================
# ROTATION FUNCTION
# =============================================================================

def rotate_photo(photo, angle):
    """
    Rotate a photo by ±30 degrees.
    
    Args:
        photo: BGRA photo image array
        angle: Rotation angle in degrees
    
    Returns:
        Rotated photo with transparent background for empty corners
    """
    h, w = photo.shape[:2]
    
    if abs(angle) < 1:
        return photo
    
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box to avoid clipping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Adjust translation to center
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # Rotate with transparent background
    rotated = cv2.warpAffine(
        photo, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128, 0)
    )
    
    return rotated


# =============================================================================
# COMPOSITING
# =============================================================================

def composite_photo_at_center(canvas, photo, cx, cy):
    """
    Composite photo onto canvas with center at (cx, cy).
    """
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    
    top_left_x = int(cx - pw / 2)
    top_left_y = int(cy - ph / 2)
    
    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = pw, ph
    dst_x1, dst_y1 = top_left_x, top_left_y
    dst_x2, dst_y2 = top_left_x + pw, top_left_y + ph
    
    # Clip to canvas bounds
    if dst_x1 < 0:
        src_x1 = -dst_x1
        dst_x1 = 0
    if dst_y1 < 0:
        src_y1 = -dst_y1
        dst_y1 = 0
    if dst_x2 > cw:
        src_x2 = cw - dst_x1
        dst_x2 = cw
    if dst_y2 > ch:
        src_y2 = ch - dst_y1
        dst_y2 = ch
    
    copy_w = int(dst_x2 - dst_x1)
    copy_h = int(dst_y2 - dst_y1)
    
    if copy_w <= 0 or copy_h <= 0:
        return canvas
    
    src_x1, src_y1 = int(src_x1), int(src_y1)
    
    # Alpha blend
    canvas_f = canvas.astype(np.float32) / 255.0
    photo_f = photo[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w].astype(np.float32) / 255.0
    
    alpha = photo_f[:, :, 3:4]
    
    canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, :3] = (
        photo_f[:, :, :3] * alpha + 
        canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, :3] * (1 - alpha)
    ).astype(np.float32)
    canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, 3] = np.maximum(
        canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, 3],
        photo_f[:, :, 3]
    )
    
    return (canvas_f * 255).astype(np.uint8)


# =============================================================================
# GLOBAL PERSPECTIVE WARP - Applied once to entire composite
# =============================================================================

def trim_pure_black_edges(img, photo_corners):
    """
    Trim only the most egregious dark edges (perspective artifacts).
    
    Uses a very strict threshold: only trim if >95% of edge is near-black.
    """
    h, w = img.shape[:2]
    if h < 200 or w < 200:
        return img
    
    luma = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]
    
    # Find min/max bounds of all photo corners
    if photo_corners and len(photo_corners) > 0:
        all_corners = np.vstack(photo_corners)
        photo_min_x = np.min(all_corners[:, 0])
        photo_max_x = np.max(all_corners[:, 0])
        photo_min_y = np.min(all_corners[:, 1])
        photo_max_y = np.max(all_corners[:, 1])
    else:
        return img
    
    # Threshold for "dark" - pixels darker than this are considered edge
    dark_threshold = 20  # Below this is "dark"
    dark_ratio_threshold = 0.15  # 80% of edge must be dark
    
    # Convert photo bounds to pixels
    photo_min_x_px = photo_min_x * w
    photo_max_x_px = photo_max_x * w
    photo_min_y_px = photo_min_y * h
    photo_max_y_px = photo_max_y * h
    
    # Trim TOP
    trim_top = 0
    while trim_top < h - 500:
        if photo_min_y_px > trim_top + 50:  # No corner within 50px
            if np.sum(luma[trim_top, :] < dark_threshold) / w > dark_ratio_threshold:
                trim_top += 1
            else:
                break
        else:
            break
    
    # Trim BOTTOM
    trim_bottom = 0
    while trim_bottom < h - 500:
        row = h - 1 - trim_bottom
        if photo_max_y_px < row - 50:
            if np.sum(luma[row, :] < dark_threshold) / w > dark_ratio_threshold:
                trim_bottom += 1
            else:
                break
        else:
            break
    
    # Trim LEFT
    trim_left = 0
    while trim_left < w - 500:
        if photo_min_x_px > trim_left + 50:
            if np.sum(luma[:, trim_left] < dark_threshold) / h > dark_ratio_threshold:
                trim_left += 1
            else:
                break
        else:
            break
    
    # Trim RIGHT
    trim_right = 0
    while trim_right < w - 500:
        col = w - 1 - trim_right
        if photo_max_x_px < col - 50:
            if np.sum(luma[:, col] < dark_threshold) / h > dark_ratio_threshold:
                trim_right += 1
            else:
                break
        else:
            break
    
    # Apply trim
    new_h = h - trim_top - trim_bottom
    new_w = w - trim_left - trim_right
    
    if new_h < 400 or new_w < 400:
        return img
    
    img = img[trim_top:h-trim_bottom, trim_left:w-trim_right]
    
    # Adjust photo corners
    for corners in photo_corners:
        corners[:, 0] -= trim_left
        corners[:, 1] -= trim_top
    
    return img


def apply_global_perspective(canvas, canvas_w, canvas_h, photo_corners=None, crop_margin=250):
    """
    Apply a SINGLE perspective warp to the entire canvas composite.
    
    This simulates the camera viewing the scene at an angle.
    After warp, crops to photo corner bounds with margin.
    
    Args:
        canvas: The flat composite image (BGR)
        canvas_w, canvas_h: Original canvas dimensions
        photo_corners: List of photo corner arrays (for cropping)
        crop_margin: Margin around photo corners after crop (default 50px)
    
    Returns:
        (warped_canvas, global_corners, transform_matrix, content_bounds)
    """
    # Source corners (original rectangle)
    src_corners = np.array([
        [0, 0],
        [canvas_w - 1, 0],
        [canvas_w - 1, canvas_h - 1],
        [0, canvas_h - 1]
    ], dtype=np.float32)
    
    # Calculate corner displacement for visible perspective
    perspective_strength = random.uniform(0.02, 0.05)
    direction = random.randint(0, 3)
    max_offset_x = canvas_w * perspective_strength
    max_offset_y = canvas_h * perspective_strength
    
    if direction == 0:  # Left side closer
        tl_offset_x = random.uniform(max_offset_x * 0.4, max_offset_x)
        tr_offset_x = random.uniform(-max_offset_x * 0.2, max_offset_x * 0.2)
        bl_offset_x = random.uniform(max_offset_x * 0.4, max_offset_x)
        br_offset_x = random.uniform(-max_offset_x * 0.2, max_offset_x * 0.2)
    elif direction == 1:  # Right side closer
        tl_offset_x = random.uniform(-max_offset_x * 0.2, max_offset_x * 0.2)
        tr_offset_x = random.uniform(max_offset_x * 0.4, max_offset_x)
        bl_offset_x = random.uniform(-max_offset_x * 0.2, max_offset_x * 0.2)
        br_offset_x = random.uniform(max_offset_x * 0.4, max_offset_x)
    elif direction == 2:  # Top closer
        tl_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
        tr_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
        bl_offset_x = random.uniform(-max_offset_x * 0.7, max_offset_x * 0.7)
        br_offset_x = random.uniform(-max_offset_x * 0.7, max_offset_x * 0.7)
    else:  # Bottom closer
        tl_offset_x = random.uniform(-max_offset_x * 0.7, max_offset_x * 0.7)
        tr_offset_x = random.uniform(-max_offset_x * 0.7, max_offset_x * 0.7)
        bl_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
        br_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
    
    v_tilt = random.uniform(-max_offset_y * 0.1, max_offset_y * 0.1)
    
    # Build destination corners
    dst_corners = np.array([
        [tl_offset_x, v_tilt],
        [canvas_w - 1 + tr_offset_x, v_tilt],
        [canvas_w - 1 + br_offset_x, canvas_h - 1 - v_tilt],
        [bl_offset_x, canvas_h - 1 - v_tilt]
    ], dtype=np.float32)
    
    # Calculate output bounds
    min_x = min(c[0] for c in dst_corners)
    max_x = max(c[0] for c in dst_corners)
    min_y = min(c[1] for c in dst_corners)
    max_y = max(c[1] for c in dst_corners)
    
    out_w = int(max_x - min_x) + 1
    out_h = int(max_y - min_y) + 1
    
    # Offset destination corners
    offset_x = -min_x
    offset_y = -min_y
    dst_offset = dst_corners.copy()
    dst_offset[:, 0] += offset_x
    dst_offset[:, 1] += offset_y
    
    # Get perspective transform
    M = cv2.getPerspectiveTransform(src_corners, dst_offset)
    
    # Apply warp
    warped = cv2.warpPerspective(
        canvas, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Transform all photo corners through the perspective warp
    warped_photo_corners = []
    if photo_corners is not None:
        for corners in photo_corners:
            ones = np.ones((len(corners), 1))
            corners_h = np.hstack([corners, ones])
            warped_corners = corners_h @ M.T
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:3]
            warped_photo_corners.append(warped_corners)
    
    # Crop to bounds: use BOTH photo corners AND canvas corners
    # This ensures we capture the full background extent
    # Get bounds from photo corners
    if warped_photo_corners:
        all_photo_corners = np.vstack(warped_photo_corners)
        photo_min_x = np.min(all_photo_corners[:, 0])
        photo_max_x = np.max(all_photo_corners[:, 0])
        photo_min_y = np.min(all_photo_corners[:, 1])
        photo_max_y = np.max(all_photo_corners[:, 1])
    else:
        photo_min_x = photo_max_x = photo_min_y = photo_max_y = None
    
    # Canvas corners after perspective
    canvas_min_x = np.min(dst_offset[:, 0])
    canvas_max_x = np.max(dst_offset[:, 0])
    canvas_min_y = np.min(dst_offset[:, 1])
    canvas_max_y = np.max(dst_offset[:, 1])
    
    # Combine: use outermost bounds from photos AND canvas
    if warped_photo_corners:
        # Photos define inner content, canvas may extend beyond
        min_x = min(photo_min_x, canvas_min_x)
        max_x = max(photo_max_x, canvas_max_x)
        min_y = min(photo_min_y, canvas_min_y)
        max_y = max(photo_max_y, canvas_max_y)
    else:
        min_x, max_x, min_y, max_y = canvas_min_x, canvas_max_x, canvas_min_y, canvas_max_y
    
    # Margin proportional to combined bounds
    combined_w = max_x - min_x
    combined_h = max_y - min_y
    crop_margin = int(min(combined_w, combined_h) * 0.15)
    crop_margin = max(crop_margin, 50)  # At least 50px
    crop_margin = min(crop_margin, 150)  # At most 150px
    
    # Crop with margin (photo corners + background extent)
    crop_x1 = max(0, int(min_x) + crop_margin)
    crop_y1 = max(0, int(min_y) + crop_margin)
    crop_x2 = min(out_w, int(max_x) - crop_margin)
    crop_y2 = min(out_h, int(max_y) - crop_margin)
    
    # Ensure minimum size
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    min_size = 600
    if crop_w < min_size or crop_h < min_size:
        # Expand crop to minimum size (centered)
        if crop_w < min_size:
            excess = min_size - crop_w
            crop_x1 -= excess // 2
            crop_x2 += excess - excess // 2
            crop_x1 = max(0, crop_x1)
            crop_x2 = min(out_w, crop_x2)
        if crop_h < min_size:
            excess = min_size - crop_h
            crop_y1 -= excess // 2
            crop_y2 += excess - excess // 2
            crop_y1 = max(0, crop_y1)
            crop_y2 = min(out_h, crop_y2)
    
    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        warped = warped[crop_y1:crop_y2, crop_x1:crop_x2]
        dst_offset[:, 0] -= crop_x1
        dst_offset[:, 1] -= crop_y1
        # Also adjust photo corners
        for i in range(len(warped_photo_corners)):
            warped_photo_corners[i][:, 0] -= crop_x1
            warped_photo_corners[i][:, 1] -= crop_y1
    
    global_corners = dst_offset
    content_bounds = (warped.shape[1], warped.shape[0])
    
    return warped, global_corners, M, content_bounds, warped_photo_corners


def trim_dark_edges(img, min_luma=8, max_dark_ratio=0.25):
    """
    Remove dark edges from image by trimming rows/columns.
    
    Keeps trimming until all edge pixels are above min_luma.
    Uses 10% threshold for more aggressive corner handling.
    """
    if len(img.shape) == 3:
        luma = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]
    else:
        luma = img.astype(float)
    
    h, w = luma.shape
    max_iterations = 500  # Allow enough iterations to remove entire dark borders
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        changed = False
        
        # Check if edges are clean (using luma)
        top_row = luma[0, :]
        bottom_row = luma[-1, :]
        left_col = luma[:, 0]
        right_col = luma[:, -1]
        
        top_dark = np.sum(top_row < min_luma)
        bottom_dark = np.sum(bottom_row < min_luma)
        left_dark = np.sum(left_col < min_luma)
        right_dark = np.sum(right_col < min_luma)
        
        # If edges are clean enough, we're done
        top_thresh = w * max_dark_ratio
        bottom_thresh = w * max_dark_ratio
        left_thresh = h * max_dark_ratio
        right_thresh = h * max_dark_ratio
        
        if (top_dark <= top_thresh and 
            bottom_dark <= bottom_thresh and
            left_dark <= left_thresh and
            right_dark <= right_thresh):
            break
        
        # Update dimensions
        h, w = luma.shape
        
        # Trim from the edge with the most dark pixels
        # (to handle corners that span multiple edges)
        dark_counts = [
            ('TOP', top_dark, top_thresh, 'h'),
            ('BOTTOM', bottom_dark, bottom_thresh, 'h'),
            ('LEFT', left_dark, left_thresh, 'w'),
            ('RIGHT', right_dark, right_thresh, 'w')
        ]
        
        # Sort by amount over threshold (most severe first)
        dark_counts.sort(key=lambda x: x[1] - x[2], reverse=True)
        
        for name, dark, thresh, dim in dark_counts:
            if dark > thresh and w > 50 and h > 50:
                if name in ('TOP', 'BOTTOM'):
                    if name == 'TOP':
                        img = img[1:, :]
                        luma = luma[1:, :]
                    else:
                        img = img[:-1, :]
                        luma = luma[:-1, :]
                else:  # LEFT/RIGHT
                    if name == 'LEFT':
                        img = img[:, 1:]
                        luma = luma[:, 1:]
                    else:
                        img = img[:, :-1]
                        luma = luma[:, :-1]
                changed = True
                break
        
        if not changed:
            break
    
    return img

# =============================================================================
# CALCULATE FINAL KEYPOINTS (photo corners after global warp)
# =============================================================================

def calculate_warped_photo_corners(flat_corners, transform_matrix):
    """
    Transform photo corners through the global perspective warp.
    """
    ones = np.ones((4, 1))
    corners_h = np.hstack([flat_corners, ones])
    warped_h = corners_h @ transform_matrix.T
    warped = warped_h[:, :2] / warped_h[:, 2:3]
    return warped


# =============================================================================
# BACKGROUND GENERATION
# =============================================================================

def fast_background(w, h):
    """Generate background in ~0.1s."""
    from numpy.random import randint, uniform, normal
    
    bg_type = random.choices(['solid', 'wood', 'fabric', 'laminate'], weights=[0.4, 0.25, 0.2, 0.15])[0]
    
    if bg_type == 'solid':
        LIGHT = [(240,242,245),(235,240,242),(250,248,245),(245,245,248),(255,255,252)]
        DARK  = [(15,15,18),(20,20,25),(12,15,20),(25,22,28),(18,18,22)]
        MID   = [(165,160,155),(140,145,138),(180,175,170),(155,150,148),(170,168,162)]
        tier = random.choices(['light','dark','mid'], weights=[0.30,0.30,0.40])[0]
        color = random.choice(LIGHT) if tier=='light' else random.choice(DARK) if tier=='dark' else random.choice(MID)
        img = np.ones((h, w, 3), dtype=np.uint8) * np.array(color)
        noise_sigma = random.uniform(3, 15)
        img = np.clip(img + normal(0, noise_sigma, (h, w, 3)), 0, 255).astype(np.uint8)
        return img
    
    elif bg_type == 'wood':
        base_colors = [(139,90,43),(160,110,65),(120,75,35),(180,130,80)]
        base = random.choice(base_colors)
        img = np.ones((h, w, 3), dtype=np.uint8) * np.array(base)
        num_grains = randint(15, 25)
        for _ in range(num_grains):
            y = randint(0, h-1)
            thickness = randint(2, 8)
            intensity = randint(-30, 30)
            offset = random.uniform(-0.02, 0.02)
            y1, y2 = max(0, y-thickness), min(h, y+thickness)
            for i in range(y1, y2):
                wavy_x = int(math.sin(i * offset) * 20)
                blend = 1 - abs(i - y) / thickness
                x1, x2 = max(0, wavy_x), min(w, wavy_x + w)
                img[i, x1:x2] = np.clip(img[i, x1:x2] + (intensity + randint(-15,15)) * blend, 0, 255)
        noise = normal(0, 8, (h, w, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(img, (3, 3), 0)
    
    elif bg_type == 'fabric':
        base_colors = [(120,110,100),(100,95,90),(140,130,120),(110,105,100)]
        color = random.choice(base_colors)
        img = np.ones((h, w, 3), dtype=np.uint8) * np.array(color)
        weave_scale = randint(3, 6)
        for i in range(0, h, weave_scale):
            for j in range(0, w, weave_scale):
                shade = randint(-5, 5)
                i2 = min(i+weave_scale, h)
                j2 = min(j+weave_scale, w)
                img[i:i2, j:j2] = np.clip(np.array(color) + shade, 0, 255)
        noise = normal(0, 6, (h, w, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(img, (3, 3), 0)
    
    else:  # laminate
        base_colors = [(180,160,140),(160,145,125),(190,175,155),(170,155,135)]
        color = random.choice(base_colors)
        img = np.ones((h, w, 3), dtype=np.uint8) * np.array(color)
        plank_h = randint(60, 120)
        for i in range(0, h, plank_h):
            shade = randint(-15, 15)
            i2 = min(i+plank_h, h)
            img[i:i2, :] = np.clip(np.array(color) + shade, 0, 255)
        noise = normal(0, 10, (h, w, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(img, (5, 5), 0)


def fast_luma_gradient(img):
    """Add luma gradient overlay."""
    h, w = img.shape[:2]
    direction = random.choice(['top', 'bottom', 'left', 'right', 'radial'])
    
    if direction == 'top':
        grad = np.linspace(0, 0.15, h)[:, np.newaxis, np.newaxis]
    elif direction == 'bottom':
        grad = np.linspace(0.15, 0, h)[:, np.newaxis, np.newaxis]
    elif direction == 'left':
        grad = np.linspace(0, 0.15, w)[np.newaxis, :, np.newaxis]
    elif direction == 'right':
        grad = np.linspace(0.15, 0, w)[np.newaxis, :, np.newaxis]
    else:
        cx, cy = w/2, h/2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        grad = (1 - dist/max_dist * 0.3)[:, :, np.newaxis]
        grad = np.clip(grad, 0.3, 1.0)
    
    factor = random.uniform(0.5, 1.5)
    img = np.clip(img * (1 + grad * factor - 0.5 * factor), 0, 255).astype(np.uint8)
    return img


def random_background_gradient(img):
    """Add 1-2 intense gradient overlays simulating uneven lighting."""
    h, w = img.shape[:2]
    num_gradients = random.randint(1, 2)
    
    for _ in range(num_gradients):
        grad_type = random.choice(['radial', 'horizontal', 'vertical', 'diagonal'])
        # More intense gradients (20-45%)
        alpha = random.uniform(0.20, 0.45)
        direction = random.choice([-1, 1])
        
        if grad_type == 'radial':
            # Offset center for more dramatic effect
            cx = random.uniform(w * 0.2, w * 0.8)
            cy = random.uniform(h * 0.2, h * 0.8)
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
            max_dist = np.sqrt(max(cx, w-cx)**2 + max(cy, h-cy)**2)
            gradient = (1 - dist/max_dist) * alpha * direction
        elif grad_type == 'horizontal':
            gradient = (np.linspace(0, 1, w) * alpha * direction)[np.newaxis, :]
        elif grad_type == 'vertical':
            gradient = (np.linspace(0, 1, h) * alpha * direction)[:, np.newaxis]
        else:
            Y, X = np.ogrid[:h, :w]
            gradient = ((X/w + Y/h) / 2) * alpha * direction
        
        gradient = gradient[:, :, np.newaxis]
        img = np.clip(img * (1 + gradient), 0, 255).astype(np.uint8)
    
    return img


def crop_to_content_bounds(img, margin=50):
    """
    Crop image to content bounds, removing pure black edge artifacts.
    
    Only trims edges that are PURE BLACK (luma < 5), not dark background.
    This distinguishes perspective artifacts from legitimate dark background.
    """
    h, w = img.shape[:2]
    if h <= 2 * margin + 200 or w <= 2 * margin + 200:
        return img
    
    # Calculate luma for edge detection
    if len(img.shape) == 3:
        luma = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]
    else:
        luma = img.astype(float)
    
    # Iteratively trim PURE BLACK edges only
    crop_top = 0
    crop_bottom = h
    crop_left = 0
    crop_right = w
    
    max_iterations = 500
    for _ in range(max_iterations):
        changed = False
        
        current_h = crop_bottom - crop_top
        current_w = crop_right - crop_left
        
        # Safety: don't crop too small
        if current_h < 800 or current_w < 800:
            break
        
        # Only trim if edge is almost entirely PURE BLACK (luma < 5)
        # This is 90% threshold for near-solid black
        if current_h > 2 * margin:
            top_row = luma[crop_top, crop_left:crop_right]
            if np.sum(top_row < 5) / current_w > 0.90:
                crop_top += 1
                changed = True
        
        if current_h > 2 * margin and crop_bottom > crop_top + margin:
            bottom_row = luma[crop_bottom - 1, crop_left:crop_right]
            if np.sum(bottom_row < 5) / current_w > 0.90:
                crop_bottom -= 1
                changed = True
        
        if current_w > 2 * margin:
            left_col = luma[crop_top:crop_bottom, crop_left]
            if np.sum(left_col < 5) / current_h > 0.90:
                crop_left += 1
                changed = True
        
        if current_w > 2 * margin and crop_right > crop_left + margin:
            right_col = luma[crop_top:crop_bottom, crop_right - 1]
            if np.sum(right_col < 5) / current_h > 0.90:
                crop_right -= 1
                changed = True
        
        if not changed:
            break
    
    # Apply final crop with minimum margin
    final_top = max(crop_top, 0)
    final_bottom = min(crop_bottom, h)
    final_left = max(crop_left, 0)
    final_right = min(crop_right, w)
    
    # Ensure minimum size
    final_h = final_bottom - final_top
    final_w = final_right - final_left
    
    if final_h < 800 or final_w < 800:
        return img
    
    return img[final_top:final_bottom, final_left:final_right]


# =============================================================================
# PHOTO EFFECTS
# =============================================================================

def fast_photo_manipulation(img):
    """Apply brightness/contrast/saturation/gamma."""
    brightness = random.uniform(0.85, 1.20)
    contrast   = random.uniform(0.85, 1.20)
    saturation = random.uniform(0.80, 1.25)
    gamma      = random.uniform(0.80, 1.30)
    
    img_f = img.astype(np.float32) / 255.0
    img_f = ((img_f - 0.5) * contrast + 0.5) * brightness
    img_f = np.clip(img_f, 0, 1)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= saturation
    hsv[:,:,2] *= brightness * contrast
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    
    blend = random.uniform(0.0, 1.0)
    img_f = img_f * (1 - blend) + img_sat * blend
    img_f = np.power(np.clip(img_f, 0.001, 1), 1.0/gamma)
    
    return np.clip(img_f * 255, 0, 255).astype(np.uint8)


def fast_glare(img):
    """Add large, spread-out glare highlights using screen blend mode.
    
    Screen mode: result = 1 - (1-a) * (1-b)
    This ONLY brightens pixels, never darkens.
    """
    if random.random() < 0.4:
        h, w = img.shape[:2]
        
        num_flares = random.randint(1, 2)
        for _ in range(num_flares):
            img_f = img.astype(np.float32) / 255.0
            
            # Flare positioned anywhere in upper portion
            cx = random.uniform(w * 0.2, w * 0.8)
            cy = random.uniform(h * 0.1, h * 0.6)
            
            # LARGE flare (15-25% of image dimensions) with lots of spread
            rx = random.uniform(w * 0.15, w * 0.25)
            ry = random.uniform(h * 0.15, h * 0.25)
            
            y, x = np.ogrid[:h, :w]
            
            # Soft elliptical flare with gradual falloff
            flare = np.maximum(0, 1 - (x - cx)**2 / (rx**2) - (y - cy)**2 / (ry**2))
            # WIDE blur for spread-out glow effect
            flare = cv2.GaussianBlur(flare.astype(np.float32), (31, 31), 0)
            
            # Screen blend: 1 - (1 - base) * (1 - flare * opacity)
            # This ONLY brightens, never darkens
            opacity = random.uniform(0.5, 0.8)  # Bright and visible
            flare_f = flare[:, :, np.newaxis]
            img_f = 1 - (1 - img_f) * (1 - flare_f * opacity)
            
            img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    
    return img


def apply_photo_shadow(canvas, photo, cx, cy, offset_x, offset_y, blur_sigma, opacity, rotation=0):
    """
    Render a drop shadow onto the canvas with spread.
    Shadow is rotated to match the photo rotation.
    
    Args:
        rotation: Photo rotation in degrees (shadow will be rotated same amount)
    """
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    num_channels = canvas.shape[2]
    
    # Rotate the offset vector to match photo rotation
    rot_rad = math.radians(rotation)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    
    # Rotate offset (offset_x, offset_y) by rotation angle
    rotated_offset_x = offset_x * cos_r - offset_y * sin_r
    rotated_offset_y = offset_x * sin_r + offset_y * cos_r
    
    # Shadow center is offset from photo center
    shadow_cx = cx + rotated_offset_x * 0.5
    shadow_cy = cy + rotated_offset_y * 0.5
    
    # Create a large enough shadow mask (unrotated)
    # Extra padding to prevent blur clipping at edges
    blur_pad = int(blur_sigma * 6) + 10  # Large buffer for blur
    shadow_w = pw + blur_pad * 2
    shadow_h = ph + blur_pad * 2
    shadow_mask = np.zeros((shadow_h, shadow_w), dtype=np.float32)
    
    # Fill center rectangle (photo shape, centered in mask)
    shadow_mask[blur_pad:blur_pad+ph, blur_pad:blur_pad+pw] = 1.0
    
    # Apply blur - moderate intensity, not too spread
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=blur_sigma)
    
    # Rotate the shadow mask to match photo rotation
    center_rot = (shadow_w / 2, shadow_h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center_rot, rotation, 1.0)
    
    # Rotate with black background for any corners
    shadow_mask = cv2.warpAffine(shadow_mask, rot_matrix, (shadow_w, shadow_h), 
                                   borderValue=0, flags=cv2.INTER_LINEAR)
    
    # Normalize mask to max of 1.0 (in case rotation spread it)
    if shadow_mask.max() > 0:
        shadow_mask = shadow_mask / shadow_mask.max()
    
    # Calculate where shadow top-left maps to on canvas
    shadow_top_left_x = int(shadow_cx - shadow_w / 2)
    shadow_top_left_y = int(shadow_cy - shadow_h / 2)
    
    # Apply shadow using multiply blend
    canvas_f = canvas.astype(np.float32) / 255.0
    
    mask_h, mask_w = shadow_mask.shape
    for dy in range(mask_h):
        for dx in range(mask_w):
            if shadow_mask[dy, dx] < 0.01:  # Skip transparent pixels
                continue
            canvas_y = shadow_top_left_y + dy
            canvas_x = shadow_top_left_x + dx
            if 0 <= canvas_y < ch and 0 <= canvas_x < cw:
                shadow_val = shadow_mask[dy, dx] * opacity
                canvas_f[canvas_y, canvas_x, :num_channels] *= (1 - shadow_val)
    
    canvas[:, :, :num_channels] = np.clip(canvas_f * 255, 0, 255).astype(np.uint8)
    return canvas


def add_rgba_alpha(img):
    """Convert BGR image to BGRA with alpha channel."""
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = 255
    return bgra


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_rotation_applied(rotation):
    """Verify rotation was applied."""
    return abs(rotation) > 5, f"Rotation: {rotation:.1f}°"


def verify_perspective_subtle(global_corners, out_w, out_h):
    """Verify global perspective is subtle."""
    src = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32)
    displacements = [np.linalg.norm(global_corners[i] - src[i]) for i in range(4)]
    
    max_disp = max(displacements)
    
    if max_disp > 300:
        return False, f"Perspective too strong: {max_disp:.0f}px"
    
    return True, f"Perspective OK: {max_disp:.0f}px"


def verify_trapezoid_shape(corners):
    """Verify shape is a proper trapezoid (not bowtie/concave)."""
    def cross_product_2d(A, B, C):
        return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])
    
    signs = []
    for i in range(4):
        A = corners[i]
        B = corners[(i + 1) % 4]
        C = corners[(i + 2) % 4]
        cp = cross_product_2d(A, B, C)
        signs.append(cp > 0)
    
    if not (all(signs) or not any(signs)):
        return False, "Non-convex shape"
    
    top_width = np.linalg.norm(corners[1] - corners[0])
    bot_width = np.linalg.norm(corners[2] - corners[3])
    
    if top_width > 0 and bot_width > 0:
        h_ratio = min(top_width, bot_width) / max(top_width, bot_width)
        if h_ratio < 0.5:
            return False, f"Edge ratio too extreme: {h_ratio:.2f}"
    
    return True, "Valid trapezoid"


def verify_corners_in_bounds(corners, out_w, out_h, margin=0):
    """
    Verify all corners are within output bounds.
    
    After global perspective, corners may legitimately extend to edges
    or even beyond due to the trapezoid transformation. We check that
    corners are within the output dimensions (with optional small margin
    for safety).
    """
    in_bounds = all(
        -margin <= corners[i, 0] < out_w + margin and 
        -margin <= corners[i, 1] < out_h + margin
        for i in range(4)
    )
    return in_bounds, "Corners in bounds" if in_bounds else "Corners out of bounds"


# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================

def timeout_handler(signum, frame):
    print("\n⏱️ TIMEOUT: Generation took too long, stopping...")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(180)

print("🖼️  Generating 10 example images (v4 - spiral packing, larger photos)...")
print("⏱️  Timeout: 180 seconds")
print()

start_time = time.time()

try:
    source_dir = Path("./images")
    sources = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.jpeg')) + list(source_dir.glob('*.png')) + list(source_dir.glob('*.webp'))
    
    print(f"📁 Found {len(sources)} source images")
    
    if len(sources) < 5:
        print("ERROR: Need at least 5 source images")
        sys.exit(1)
    
    output_dir = Path("../data/examples_v18")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    CANVAS_W, CANVAS_H = 3000, 1800  # Larger canvas to avoid dark corners
    
    random.seed(42)
    np.random.seed(42)
    
    print("\n📸 Generating images...\n")
    
    verification_stats = {
        'total_photos': 0,
        'photos_with_rotation': 0,
        'photos_in_bounds': 0,
        'valid_trapezoids': 0,
        'subtle_perspective': 0,
        'global_perspective_applied': 0
    }
    
    for i in range(10):
        img_start = time.time()
        
        # Generate background
        bg = fast_background(CANVAS_W, CANVAS_H)
        bg = fast_luma_gradient(bg)
        bg = random_background_gradient(bg)
        
        # Convert to BGRA for compositing
        canvas = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
        
        # Spiral packing with polygon collision
        placements = spiral_pack_photos(CANVAS_W, CANVAS_H)
        
        placed_photos = []
        
        for placement in placements:
            # Load photo
            photo = cv2.imread(str(random.choice(sources)))
            if photo is None:
                continue
            
            # Scale to target size
            target_w, target_h = placement['width'], placement['height']
            h_orig, w_orig = photo.shape[:2]
            scale = min(target_w / w_orig, target_h / h_orig)
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            photo = cv2.resize(photo, (new_w, new_h))
            
            # Apply effects BEFORE rotation
            photo = fast_photo_manipulation(photo)
            photo = fast_glare(photo)
            
            # Add alpha channel
            photo = add_rgba_alpha(photo)
            
            # CRITICAL: Apply rotation BEFORE compositing!
            photo = rotate_photo(photo, placement['rotation'])
            
            # Composite photo at center position
            center_x = placement['center_x']
            center_y = placement['center_y']
            
            # Apply shadow FIRST (renders shadow onto canvas before photo)
            shadow_params = placement.get('shadow_params', {})
            if shadow_params:
                canvas = apply_photo_shadow(
                    canvas, photo, center_x, center_y,
                    shadow_params['offset_x'], shadow_params['offset_y'],
                    shadow_params['blur_sigma'], shadow_params['opacity'],
                    rotation=placement['rotation']  # Shadow matches photo rotation
                )
            
            # Then composite the rotated photo on top
            canvas = composite_photo_at_center(canvas, photo, center_x, center_y)
            
            # Store polygon for global warp transformation
            placed_photos.append({
                'polygon': placement['polygon'].copy(),
                'rotation': placement['rotation'],
            })
            
            verification_stats['total_photos'] += 1
            if abs(placement['rotation']) > 5:
                verification_stats['photos_with_rotation'] += 1
        
        # Apply ONE global perspective warp to entire composite
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        
        photo_corners_list = [photo_data['polygon'] for photo_data in placed_photos]
        
        warped_canvas, global_corners, transform_matrix, content_bounds, warped_photo_corners = apply_global_perspective(
            canvas_bgr, CANVAS_W, CANVAS_H,
            photo_corners=photo_corners_list,
            crop_margin=250  # Larger to avoid dark corners
        )
        
        # Use the pre-computed warped corners
        final_photos = []
        out_w, out_h = warped_canvas.shape[1], warped_canvas.shape[0]
        
        for idx, photo_data in enumerate(placed_photos):
            warped_corners = warped_photo_corners[idx]
            
            final_photos.append({
                'corners': warped_corners,
                'rotation': photo_data['rotation']
            })
            
            ok, _ = verify_corners_in_bounds(warped_corners, out_w, out_h, margin=0)
            if ok:
                verification_stats['photos_in_bounds'] += 1
            
            ok, _ = verify_trapezoid_shape(warped_corners)
            if ok:
                verification_stats['valid_trapezoids'] += 1
        
        # Global perspective check
        ok, _ = verify_perspective_subtle(global_corners, out_w, out_h)
        if ok:
            verification_stats['subtle_perspective'] += 1
        verification_stats['global_perspective_applied'] += 1
        
        # Save image
        pil_img = Image.fromarray(cv2.cvtColor(warped_canvas, cv2.COLOR_BGR2RGB))
        img_path = output_dir / f"example_{i+1:02d}.jpg"
        pil_img.save(img_path, quality=90)
        
        # Save label file
        lbl_path = output_dir / f"example_{i+1:02d}.txt"
        with open(lbl_path, 'w') as f:
            for p in final_photos:
                kps = p['corners']
                x_center = sum(k[0] for k in kps) / 4 / out_w
                y_center = sum(k[1] for k in kps) / 4 / out_h
                width = (max(k[0] for k in kps) - min(k[0] for k in kps)) / out_w
                height = (max(k[1] for k in kps) - min(k[1] for k in kps)) / out_h
                
                line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                for kx, ky in kps:
                    # Clamp keypoints to [0, 1] for YOLO format
                    kx_clamped = max(0, min(1, kx / out_w))
                    ky_clamped = max(0, min(1, ky / out_h))
                    line += f" {kx_clamped:.6f} {ky_clamped:.6f} 2"
                line += "\n"
                f.write(line)
        
        img_time = time.time() - img_start
        print(f"  {i+1:2d}/10: {img_time:5.1f}s ({len(final_photos)} photos, global warp applied)")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("📊 VERIFICATION RESULTS")
    print(f"{'='*60}")
    
    total = verification_stats['total_photos']
    print(f"\n  Total photos: {total}")
    print(f"  Photos with rotation (±15°): {verification_stats['photos_with_rotation']}/{total}")
    print(f"  Photos in bounds: {verification_stats['photos_in_bounds']}/{total}")
    print(f"  Valid trapezoids: {verification_stats['valid_trapezoids']}/{total}")
    print(f"  Subtle global perspective: {verification_stats['subtle_perspective']}/{verification_stats['global_perspective_applied']}")
    
    print(f"\n✅ Done! 10 images in {total_time:.1f}s")
    print(f"📂 {output_dir.absolute()}")
    
    for f in sorted(output_dir.glob("example_*.jpg")):
        print(f"   - {f.name} ({f.stat().st_size//1024} KB)")
    
    signal.alarm(0)
    
except Exception as e:
    signal.alarm(0)
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
    sys.exit(1)
