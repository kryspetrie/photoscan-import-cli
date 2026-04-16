#!/usr/bin/env python3
"""
Photo Pose Detector - Synthetic Training Data Generator (v31)
===============================================================

This module generates synthetic training images for YOLO26-pose corner detection.
It creates realistic images of photographs arranged on a table surface, simulating
real-world scanned documents.

ARCHITECTURE
------------
The generator uses a "single global perspective warp" architecture:

    1. PACK: Place flat photos (rectangles) with rotation on a flat background
    2. WARP: Apply ONE perspective warp to the ENTIRE composite
    3. CROP: Trim to content bounds

This ensures:
- Photos start as perfect rectangles (ground truth = 4 corners)
- Global warp distorts all photos uniformly
- Ground truth corners track correctly through the warp

OUTPUT FORMAT
-------------
YOLO26-pose format with 4 keypoints per photo:

    <class> <x> <y> <w> <h> <kx0> <ky0> <kc0> ... <kx3> <ky3> <kc3>

Where keypoints 0-3 are: Top-Left, Top-Right, Bottom-Right, Bottom-Left

KEY FEATURES
------------
- Background Distribution: 30% dark, 30% light, 40% colored (reduced saturation)
- 3 linear gradients with screen blend overlay per background
- Real texture overlays with multiply/screen blend
- Soft photo edges (0-5px blur on ALL edges) - simulates imperfect focus
- Drop shadows with blur - simulates depth/elevation
- Large, intense glare with screen blend - simulates lighting
- Pixel-based collision detection with dilated masks (~20px gap)
- Spiral packing from center - efficient space utilization
- Global perspective warp (2-5% strength) - simulates camera angle

USAGE
-----
    # Generate example images
    python generate_dataset.py

    # Images saved to ../data/examples_v31/

CONFIGURATION
-------------
Key parameters (search for CONFIG or edit constants):
- CANVAS_W, CANVAS_H: Canvas dimensions (3000x1800)
- NUM_PHOTOS: Photos per frame (5-10)
- ROTATION_RANGE: ±30 degrees
- EDGE_BLUR: 0-5 pixels
- SHADOW_OFFSET: 0-15 pixels
- SHADOW_BLUR: 6-15 sigma
- GLARE_SIZE: 20-40% of photo
- GLARE_OPACITY: 60-100%

Author: Photo Pose Detector Project
Version: 31
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
# CONFIGURATION CONSTANTS
# =============================================================================
# Modify these values to adjust the dataset generation behavior.
# These are the key parameters that affect output quality and variety.

CONFIG = {
    # Canvas settings
    'CANVAS_WIDTH': 3000,           # Canvas width in pixels
    'CANVAS_HEIGHT': 1800,          # Canvas height in pixels
    
    # Photo packing
    'NUM_PHOTOS_MIN': 5,            # Minimum photos per frame
    'NUM_PHOTOS_MAX': 10,           # Maximum photos per frame
    'PHOTO_SIZE_RATIO': 0.30,       # Base photo size as ratio of canvas width
    'SIZE_VARIATION': 0.10,         # ±10% size variation
    'EDGE_MARGIN': 160,             # Margin from canvas edge (pixels)
    
    # Photo rotation
    'ROTATION_RANGE': 30,           # Max rotation angle (degrees)
    'ROTATION_PROBABILITY': 0.10,   # 10% of photos are rotated
    
    # Photo edges (focus simulation)
    'EDGE_BLUR_MIN': 0,             # Minimum edge blur (pixels)
    'EDGE_BLUR_MAX': 5,             # Maximum edge blur (pixels)
    
    # Drop shadows
    'SHADOW_OFFSET_MAX': 15,        # Maximum shadow offset (pixels)
    'SHADOW_BLUR_MIN': 6,           # Minimum shadow blur sigma
    'SHADOW_BLUR_MAX': 15,          # Maximum shadow blur sigma
    
    # Glare effects
    'GLARE_PROBABILITY': 0.50,       # 50% of photos have glare
    'GLARE_COUNT_MIN': 2,           # Minimum glare flares per photo
    'GLARE_COUNT_MAX': 4,           # Maximum glare flares per photo
    'GLARE_SIZE_MIN': 0.20,         # Minimum glare size (% of photo)
    'GLARE_SIZE_MAX': 0.40,         # Maximum glare size (% of photo)
    'GLARE_OPACITY_MIN': 0.60,      # Minimum glare opacity
    'GLARE_OPACITY_MAX': 1.00,      # Maximum glare opacity
    
    # Background
    'BG_DARK_RATIO': 0.30,          # 30% dark backgrounds
    'BG_LIGHT_RATIO': 0.30,        # 30% light backgrounds
    'BG_COLORED_RATIO': 0.40,       # 40% colored backgrounds
}


# =============================================================================
# POLYGON UTILITIES
# =============================================================================
# These functions handle geometric calculations for rotated photo rectangles.
# Used for collision detection and keypoint tracking.

def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """
    Calculate the 4 corner coordinates of a rotated rectangle.
    
    This is used to:
    1. Track photo corners through global perspective warp
    2. Generate ground truth keypoints for YOLO training
    3. Perform collision detection between photos
    
    Args:
        width, height: Original photo dimensions in pixels
        center_x, center_y: Center position on canvas
        rotation: Rotation angle in degrees (positive = clockwise)
    
    Returns:
        numpy.ndarray: 4x2 array of corner coordinates [[x,y], ...]
                       Order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    
    Example:
        >>> corners = get_rotated_polygon(800, 600, 1500, 900, 15)
        >>> # Returns array of 4 [x,y] coordinates
    """
    # Photo corners relative to center (before rotation)
    hw, hh = width / 2, height / 2
    corners = np.array([
        [-hw, -hh],  # TL
        [ hw, -hh],  # TR
        [ hw,  hh],  # BR
        [-hw,  hh]   # BL
    ], dtype=np.float32)
    
    # Apply rotation matrix
    angle_rad = math.radians(rotation)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    # Rotate each corner and translate to canvas position
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
    """Radial packing with pixel-based collision detection and 20px margin."""
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
    
    # Collision margin and kernel
    margin = 20
    kernel = np.ones((margin*2+1, margin*2+1), np.uint8)
    
    def create_dilated_mask(width, height, center_x, center_y, rotation):
        """Create mask dilated by 20px for collision checking."""
        poly = get_rotated_polygon(width, height, center_x, center_y, rotation)
        
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        min_x, max_x = int(min(xs)), int(max(xs)) + 1
        min_y, max_y = int(min(ys)), int(max(ys)) + 1
        
        bb_x = max(0, min_x - margin)
        bb_y = max(0, min_y - margin)
        bb_w = min(canvas_w, max_x + margin) - bb_x
        bb_h = min(canvas_h, max_y + margin) - bb_y
        
        if bb_w <= 0 or bb_h <= 0:
            return None, (bb_x, bb_y, bb_w, bb_h)
        
        mask = np.zeros((bb_h, bb_w), dtype=np.uint8)
        pts = np.array([[int(p[0]) - bb_x, int(p[1]) - bb_y] for p in poly], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        dilated = cv2.dilate(mask, kernel)
        return dilated, (bb_x, bb_y, bb_w, bb_h)
    
    def check_collision(dilated_mask, bb_x, bb_y, occ_dilated, threshold=5):
        """Check if dilated mask overlaps with dilated occupancy."""
        if dilated_mask is None:
            return True
        
        bb_h, bb_w = dilated_mask.shape
        x2 = min(bb_x + bb_w, canvas_w)
        y2 = min(bb_y + bb_h, canvas_h)
        
        occ_region = occ_dilated[bb_y:y2, bb_x:x2]
        mask_region = dilated_mask[:y2-bb_y, :x2-bb_x]
        
        overlap = cv2.bitwise_and(occ_region, mask_region)
        return np.count_nonzero(overlap) > threshold
    
    def place_photo(dilated_mask, bb_x, bb_y, occ_dilated):
        """Add dilated mask to occupancy and pre-dilated occupancy."""
        if dilated_mask is None:
            return
        bb_h, bb_w = dilated_mask.shape
        x2 = min(bb_x + bb_w, canvas_w)
        y2 = min(bb_y + bb_h, canvas_h)
        region = dilated_mask[:y2-bb_y, :x2-bb_x]
        occupancy[bb_y:y2, bb_x:x2] = cv2.bitwise_or(occupancy[bb_y:y2, bb_x:x2], region)
        occ_dilated[bb_y:y2, bb_x:x2] = cv2.bitwise_or(occ_dilated[bb_y:y2, bb_x:x2], region)
    
    for photo_idx, (width, height, rotation) in enumerate(photo_configs):
        placed = False
        
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
                
                # Pre-dilate occupancy once per shrink iteration
                occ_dilated = cv2.dilate(occupancy, kernel) if np.any(occupancy) else occupancy.copy()
                
                for px, py in candidates:
                    cx_pos, cy_pos = px, py
                    
                    poly_test = get_rotated_polygon(w_s, h_s, cx_pos, cy_pos, angle)
                    bx1 = min(p[0] for p in poly_test)
                    by1 = min(p[1] for p in poly_test)
                    bx2 = max(p[0] for p in poly_test)
                    by2 = max(p[1] for p in poly_test)
                    
                    if bx1 < edge_margin or by1 < edge_margin:
                        continue
                    if bx2 > canvas_w - edge_margin or by2 > canvas_h - edge_margin:
                        continue
                    
                    dilated_mask, (bb_x, bb_y, _, _) = create_dilated_mask(w_s, h_s, cx_pos, cy_pos, angle)
                    
                    if check_collision(dilated_mask, bb_x, bb_y, occ_dilated):
                        continue
                    
                    place_photo(dilated_mask, bb_x, bb_y, occ_dilated)
                    
                    shadow_params = {
                        'offset_x': rnd.choice([-1, 1]) * rnd.randint(0, 15),
                        'offset_y': rnd.choice([-1, 1]) * rnd.randint(0, 15),
                        'blur_sigma': rnd.uniform(6, 15),
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
    
    for attempt in range(10):
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

def random_base_background(w, h):
    """
    Generate a random background with controlled brightness and saturation distribution.
    
    BACKGROUND DISTRIBUTION
    -----------------------
    - 30% Dark grey to black (lightness 5-28%, saturation 0-4%)
    - 30% Light grey to white (lightness 69-96%, saturation 0-4%)
    - 40% Colored (lightness 19-86%, saturation 4-40% - reduced by 20%)
    
    PROCESSING
    ----------
    1. Select background type based on random distribution
    2. Convert HLS to RGB using colorsys
    3. Create solid color base
    4. Add subtle Gaussian noise (σ = 1-4)
    5. Apply 3 linear gradients with screen blend overlay
    
    This distribution mimics real-world scanning scenarios:
    - Dark backgrounds: scanner bed, dark desk surfaces
    - Light backgrounds: white paper, light desks
    - Colored backgrounds: patterned surfaces, varied environments
    
    Args:
        w: Background width in pixels
        h: Background height in pixels
    
    Returns:
        numpy.ndarray: BGR image (uint8)
    
    See Also:
        apply_3_linear_gradients(): Adds gradient overlays
        apply_texture_overlay(): Adds texture layer
    """
    import colorsys
    
    # Categorize background type
    rand_val = random.random()
    
    if rand_val < 0.30:
        # 30% dark-grey to black (very low saturation)
        lightness = random.uniform(0.04, 0.28)  # Dark: 10-72 out of 255
        saturation = random.uniform(0, 0.04)     # Very low saturation: 0-4%
    elif rand_val < 0.60:
        # 30% light-grey to white (very low saturation)
        lightness = random.uniform(0.69, 0.96)  # Light: 176-245 out of 255
        saturation = random.uniform(0, 0.04)   # Very low saturation: 0-4%
    else:
        # 40% full color variety with 50% reduced saturation, 20% overall reduction
        lightness = random.uniform(0.19, 0.86)  # Full range: 48-219 out of 255
        saturation = random.uniform(0.04, 0.40)  # 20% reduced max: 4-40%
    
    # Random hue (0-1)
    hue = random.uniform(0, 1)
    
    # Convert HLS to RGB (colorsys uses HLS order: Hue, Lightness, Saturation)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    color = (int(r * 255), int(g * 255), int(b * 255))
    
    # Create solid color background
    img = np.ones((h, w, 3), dtype=np.float32) * np.array(color, dtype=np.float32)
    
    # Add subtle noise
    noise_sigma = random.uniform(1, 4)
    noise = np.random.normal(0, noise_sigma, (h, w, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Apply 3 random linear gradients with screen overlay
    img = apply_3_linear_gradients(img)
    
    return img


def apply_3_linear_gradients(img):
    """Apply 3 random linear gradients with screen blend (0-20% opacity).
    
    Gradients go from transparent to white, creating subtle light overlays.
    """
    h, w = img.shape[:2]
    img_f = img.astype(np.float32) / 255.0
    
    for _ in range(3):
        # Random gradient direction
        direction = random.choice(['horizontal', 'vertical', 'diagonal_tl', 'diagonal_tr'])
        
        # Create gradient mask using meshgrid for efficiency
        if direction == 'horizontal':
            x = np.linspace(0, 1, w)
            gradient = np.tile(x, (h, 1))
        elif direction == 'vertical':
            y = np.linspace(0, 1, h)[:, np.newaxis]
            gradient = np.tile(y, (1, w))
        elif direction == 'diagonal_tl':
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)[:, np.newaxis]
            gradient = x + y  # Diagonal from top-left
            gradient = gradient / gradient.max()
        else:  # diagonal_tr
            x = np.linspace(1, 0, w)
            y = np.linspace(0, 1, h)[:, np.newaxis]
            gradient = x + y  # Diagonal from top-right
            gradient = gradient / gradient.max()
        
        # Gradient goes from 0 (no overlay) to 1 (full white overlay)
        # Random opacity (0-20%)
        opacity = random.uniform(0, 0.20)
        
        # Screen blend: result = 1 - (1 - base) * (1 - overlay)
        # The overlay is white (1,1,1) at varying intensities
        overlay = gradient[:, :, np.newaxis] * opacity
        
        # Screen blend
        result = 1.0 - (1.0 - img_f) * (1.0 - overlay)
        
        img_f = result
    
    return np.clip(img_f * 255, 0, 255).astype(np.uint8)


def apply_texture_overlay(canvas):
    """Apply a random texture overlay to the background.
    
    - Load random texture from textures folder
    - Stretch to canvas size
    - Apply with random opacity (0-40%)
    - Multiply blend for light backgrounds, Screen blend for dark backgrounds
    - Random flip (horizontal, vertical, both)
    """
    textures_dir = Path("/Users/krys.petrie/dev/photo-pose-detector/textures")
    
    if not textures_dir.exists():
        return canvas
    
    textures = list(textures_dir.glob("*.jpg")) + list(textures_dir.glob("*.png"))
    if not textures:
        return canvas
    
    # Pick random texture
    texture_path = random.choice(textures)
    texture = cv2.imread(str(texture_path))
    
    if texture is None:
        return canvas
    
    # Resize to canvas
    texture = cv2.resize(texture, (canvas.shape[1], canvas.shape[0]))
    
    # Random flip
    flip_code = random.choice([-1, 0, 1, None])  # None=no flip, -1=both, 0=vertical, 1=horizontal
    if flip_code is not None:
        texture = cv2.flip(texture, flip_code)
    
    # Random opacity 0-40%
    opacity = random.uniform(0, 0.40)
    
    # Choose blend mode randomly (50/50 screen vs multiply)
    use_screen = random.choice([True, False])
    
    # Convert canvas to float for blending
    canvas_f = canvas.astype(np.float32) / 255.0
    texture_f = texture.astype(np.float32) / 255.0
    
    if use_screen:
        # Screen blend: result = 1 - (1 - canvas) * (1 - texture)
        # This lightens the image
        blended = 1.0 - (1.0 - canvas_f) * (1.0 - texture_f)
    else:
        # Multiply blend: result = canvas * texture
        # This darkens/adds texture
        blended = canvas_f * texture_f
    
    # Mix original and blended based on opacity
    result = canvas_f * (1 - opacity) + blended * opacity
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)


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
    if random.random() < 0.5:
        h, w = img.shape[:2]
        
        num_flares = random.randint(2, 4)
        for _ in range(num_flares):
            img_f = img.astype(np.float32) / 255.0
            
            # Flare positioned anywhere in upper portion
            cx = random.uniform(w * 0.15, w * 0.85)
            cy = random.uniform(h * 0.1, h * 0.7)
            
            # LARGER flare (20-40% of image dimensions) with lots of spread
            rx = random.uniform(w * 0.20, w * 0.40)
            ry = random.uniform(h * 0.20, h * 0.40)
            
            y, x = np.ogrid[:h, :w]
            
            # Soft elliptical flare with gradual falloff
            flare = np.maximum(0, 1 - (x - cx)**2 / (rx**2) - (y - cy)**2 / (ry**2))
            # WIDE blur for spread-out glow effect
            flare = cv2.GaussianBlur(flare.astype(np.float32), (51, 51), 0)
            
            # Screen blend: 1 - (1 - base) * (1 - flare * opacity)
            # This ONLY brightens, never darkens
            opacity = random.uniform(0.6, 1.0)  # Brighter and more intense
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
    
    mask_h, mask_w = shadow_mask.shape
    
    # Apply shadow using vectorized multiply blend (540x faster than pixel loop)
    canvas_f = canvas.astype(np.float32) / 255.0
    
    y1, y2 = shadow_top_left_y, shadow_top_left_y + mask_h
    x1, x2 = shadow_top_left_x, shadow_top_left_x + mask_w
    
    # Clip to canvas bounds
    clip_y1 = max(0, y1)
    clip_y2 = min(ch, y2)
    clip_x1 = max(0, x1)
    clip_x2 = min(cw, x2)
    
    if clip_y2 > clip_y1 and clip_x2 > clip_x1:
        # Calculate source offset
        src_y1 = clip_y1 - y1
        src_x1 = clip_x1 - x1
        src_y2 = src_y1 + (clip_y2 - clip_y1)
        src_x2 = src_x1 + (clip_x2 - clip_x1)
        
        # Apply shadow to clipped region
        shadow_region = shadow_mask[src_y1:src_y2, src_x1:src_x2]
        shadow_vals = shadow_region * opacity
        
        # Multiply blend: multiply canvas by (1 - shadow)
        for c in range(num_channels):
            canvas_f[clip_y1:clip_y2, clip_x1:clip_x2, c] *= (1 - shadow_vals)
    
    canvas[:, :, :num_channels] = np.clip(canvas_f * 255, 0, 255).astype(np.uint8)
    return canvas


def add_rgba_alpha(img):
    """Convert BGR image to BGRA with alpha channel."""
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = 255
    return bgra


def blur_alpha_edges(photo, edge_effect=None):
    """
    Create soft/blurred edges on ALL photo edges to simulate imperfect camera focus.
    
    FOCUS SIMULATION
    Real camera images often have soft/blurry edges due to depth of field,
    lens imperfections, or slightly off focus. This function simulates that by:
    
    1. Create distance-to-edge map for ALL edges (not just corners)
    2. Create soft alpha mask (center=opaque, edges=fade)
    3. Blend original with Gaussian-blurred version at edges
    4. Fade alpha at edges for clean compositing
    
    The result: center is sharp, edges are soft with gradual transition.
    
    Args:
        photo: BGRA photo image (numpy.ndarray, shape [h, w, 4])
        edge_effect: Blur amount in pixels. If None, randomly selected [0, 5].
                    Larger values = more blur, smaller values = sharper.
    
    Returns:
        numpy.ndarray: BGRA photo with soft/blurred edges on all sides.
                       Alpha channel is feathered at edges for clean compositing.
    """
    if photo.shape[2] != 4:
        return photo
    
    h, w = photo.shape[:2]
    
    # Random edge effect if not specified (0-5 pixels)
    if edge_effect is None:
        edge_effect = random.uniform(0, 5)
    
    if edge_effect < 1:
        return photo
    
    alpha = photo[:, :, 3].astype(np.float32) / 255.0
    rgb = photo[:, :, :3].astype(np.float32)
    
    # Step 1: Create distance-to-edge map for ALL edges (not just corners)
    # Distance to top/bottom edges
    y_dist = np.minimum(np.arange(h)[:, np.newaxis], np.arange(h)[::-1, np.newaxis])
    # Distance to left/right edges
    x_dist = np.minimum(np.arange(w), np.arange(w)[::-1])
    
    # Use MIN distance to any edge (creates uniform band around ALL edges)
    edge_dist = np.minimum(y_dist, x_dist)
    
    # Step 2: Create soft alpha mask based on distance
    # Center = 1.0 (opaque), edges = fade based on edge_effect
    fade_dist = edge_effect * 2.5  # How far the fade extends
    soft_alpha = np.clip(edge_dist / fade_dist, 0, 1)
    
    # Step 3: Create blurred version of photo (simulates focus blur at edges)
    blur_size = int(edge_effect * 2) * 2 + 1
    blur_size = max(blur_size, 5)
    rgb_blurred = cv2.GaussianBlur(rgb, (blur_size, blur_size), 0)
    
    # Step 4: Blend original with blurred based on soft alpha
    # Near edges (low soft_alpha) = more blurred, center (high soft_alpha) = original
    blend_factor = 1.0 - soft_alpha
    rgb_soft = rgb * (1 - blend_factor)[:, :, np.newaxis] + rgb_blurred * blend_factor[:, :, np.newaxis]
    
    # Step 5: Apply soft alpha (multiply by original alpha for clean center)
    soft_alpha_final = alpha * soft_alpha
    
    # Apply
    photo[:, :, :3] = np.clip(rgb_soft, 0, 255).astype(np.uint8)
    photo[:, :, 3] = np.clip(soft_alpha_final * 255, 0, 255).astype(np.uint8)
    
    return photo


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

print("🖼️  Generating 10 example images (v31 - full feature set)...")
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
    
    output_dir = Path("../data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    CANVAS_W, CANVAS_H = 1920, 1080  # HD resolution for quality + speed
    
    random.seed(None)  # Use random seeds for variety
    np.random.seed(None)
    
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
        
        # Generate background with random colors, brightness, gradients, noise, texture
        bg = random_base_background(CANVAS_W, CANVAS_H)
        bg = apply_texture_overlay(bg)
        
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
            
            # Blur alpha edges to simulate imperfect focus (0-10% of image size, max 20px)
            photo = blur_alpha_edges(photo)
            
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
