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


def polygons_overlap(poly1, poly2):
    """
    Check if two convex polygons overlap.
    Uses Separating Axis Theorem (SAT) for accurate collision detection.
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
        
        if max1 < min2 or max2 < min1:
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


# =============================================================================
# SPIRAL PACKING ALGORITHM (Polygon-based collision)
# =============================================================================

def spiral_pack_photos(canvas_w, canvas_h, num_photos=None):
    """
    Pack photos using spiral placement from center with polygon collision.
    
    Uses actual rotated polygon geometry for collision detection,
    not rectangular bounding boxes. This allows tighter packing.
    
    Args:
        canvas_w, canvas_h: Canvas dimensions
        num_photos: Number of photos (default: random 5-7)
    
    Returns:
        List of placements with polygon geometry
    """
    if num_photos is None:
        num_photos = random.randint(8, 12)  # More photos per image
    
    # Photo size - large enough to fill frame but allow multiple photos
    base_w = int(canvas_w * 0.30)  # ~30% of canvas width (smaller to fit more)
    base_h = int(base_w * 0.75)    # ~75% (4:3 landscape aspect)
    
    # Moderate variation (±12%)
    size_variation = 0.12
    
    placements = []
    
    # Center of canvas
    center_x = canvas_w / 2
    center_y = canvas_h / 2
    
    # Safety margin: photos must stay this far from canvas edge
    # This prevents photos from extending to image edges after perspective warp
    # Add extra buffer for rotation (max ~30% diagonal increase at 15°)
    edge_margin = 250  # Reduced from 400 to fit more photos
    
    # Safe bounds for center positions
    min_x = edge_margin
    max_x = canvas_w - edge_margin
    min_y = edge_margin
    max_y = canvas_h - edge_margin
    
    # Generate spiral positions from center outward
    def generate_spiral_positions(center_x, center_y, max_radius, num_points):
        """Generate positions in a spiral pattern."""
        positions = []
        angle_step = 137.5  # Golden angle for good spiral distribution
        
        for i in range(num_points):
            # Spiral: radius grows, angle increases
            t = i / max(num_points - 1, 1)
            radius = t * max_radius
            angle = math.radians(i * angle_step)
            
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Add some random jitter for variety
            jitter_x = random.uniform(-30, 30)
            jitter_y = random.uniform(-30, 30)
            
            positions.append((x + jitter_x, y + jitter_y))
        
        return positions
    
    # Maximum radius to cover usable area
    max_radius = math.sqrt((max_x - center_x)**2 + (max_y - center_y)**2)
    
    # Generate more candidate positions than needed
    candidate_positions = generate_spiral_positions(
        center_x, center_y, max_radius, 
        num_photos * 4 + 20  # Extra candidates for flexibility
    )
    
    # Shuffle to add randomness while maintaining spiral-like distribution
    random.shuffle(candidate_positions)
    
    # Try to place each photo
    for photo_idx in range(num_photos):
        # Random size (small variation) and rotation
        scale = random.uniform(1 - size_variation, 1 + size_variation)
        width = int(base_w * scale)
        height = int(base_h * scale)
        rotation = random.uniform(-10, 10)  # Reduced rotation to limit corner displacement
        
        placed = False
        
        # Try each candidate position
        for cx, cy in candidate_positions:
            # Skip if outside safe bounds
            if cx < min_x or cx > max_x or cy < min_y or cy > max_y:
                continue
            
            # Get polygon for this placement
            poly = get_rotated_polygon(width, height, cx, cy, rotation)
            
            # Bounds check using actual polygon
            # Account for rotation by checking polygon corners
            bx1, by1, bx2, by2 = get_polygon_bounds(poly)
            if bx1 < edge_margin or by1 < edge_margin or bx2 > canvas_w - edge_margin or by2 > canvas_h - edge_margin:
                continue
            
            # Collision check against all placed polygons (using SAT)
            collision = False
            for existing in placements:
                if polygons_overlap(poly, existing['polygon']):
                    collision = True
                    break
            
            if not collision:
                placements.append({
                    'x': cx - width / 2,
                    'y': cy - height / 2,
                    'width': width,
                    'height': height,
                    'rotation': rotation,
                    'center_x': cx,
                    'center_y': cy,
                    'polygon': poly
                })
                placed = True
                break
        
        # If can't place, try with progressively smaller sizes
        if not placed:
            for shrink in [0.90, 0.80, 0.70, 0.60]:
                width_s = int(width * shrink)
                height_s = int(height * shrink)
                
                for cx, cy in candidate_positions:
                    if cx < min_x or cx > max_x or cy < min_y or cy > max_y:
                        continue
                    
                    poly = get_rotated_polygon(width_s, height_s, cx, cy, rotation)
                    bx1, by1, bx2, by2 = get_polygon_bounds(poly)
                    if bx1 < edge_margin or by1 < edge_margin or bx2 > canvas_w - edge_margin or by2 > canvas_h - edge_margin:
                        continue
                    
                    collision = False
                    for existing in placements:
                        if polygons_overlap(poly, existing['polygon']):
                            collision = True
                            break
                    
                    if not collision:
                        placements.append({
                            'x': cx - width_s / 2,
                            'y': cy - height_s / 2,
                            'width': width_s,
                            'height': height_s,
                            'rotation': rotation,
                            'center_x': cx,
                            'center_y': cy,
                            'polygon': poly
                        })
                        placed = True
                        break
                
                if placed:
                    break
        
        # Final fallback: try with much smaller size anywhere
        if not placed:
            width_s = int(width * 0.5)
            height_s = int(height * 0.5)
            
            # Try random positions
            for _ in range(50):
                cx = random.uniform(edge_margin + width_s/2, canvas_w - edge_margin - width_s/2)
                cy = random.uniform(edge_margin + height_s/2, canvas_h - edge_margin - height_s/2)
                
                poly = get_rotated_polygon(width_s, height_s, cx, cy, rotation)
                
                collision = False
                for existing in placements:
                    if polygons_overlap(poly, existing['polygon']):
                        collision = True
                        break
                
                if not collision:
                    placements.append({
                        'x': cx - width_s / 2,
                        'y': cy - height_s / 2,
                        'width': width_s,
                        'height': height_s,
                        'rotation': rotation,
                        'center_x': cx,
                        'center_y': cy,
                        'polygon': poly
                    })
                    placed = True
                    break
    
    return placements


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

def apply_global_perspective(canvas, canvas_w, canvas_h, photo_corners=None, crop_margin=100):
    """
    Apply a SINGLE perspective warp to the entire canvas composite.
    
    This simulates the camera viewing the scene at an angle.
    After warp, crops to content bounds with clean margin.
    
    Args:
        canvas: The flat composite image (BGR)
        canvas_w, canvas_h: Original canvas dimensions
        photo_corners: List of photo corner arrays (for cropping)
        crop_margin: Margin around content after crop (default 25px)
    
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
    perspective_strength = random.uniform(0.06, 0.12)
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
    
    v_tilt = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
    
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
    
    # First, crop to canvas bounds + padding to remove perspective black borders
    # The canvas corners map to dst_offset, so use them with padding
    crop_padding = crop_margin
    
    crop_x1 = max(0, int(min(dst_offset[:, 0])) - crop_padding)
    crop_y1 = max(0, int(min(dst_offset[:, 1])) - crop_padding)
    crop_x2 = min(out_w, int(max(dst_offset[:, 0])) + crop_padding)
    crop_y2 = min(out_h, int(max(dst_offset[:, 1])) + crop_padding)
    
    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        warped = warped[crop_y1:crop_y2, crop_x1:crop_x2]
        dst_offset[:, 0] -= crop_x1
        dst_offset[:, 1] -= crop_y1
    
    # Then, trim any remaining dark edges aggressively
    warped = trim_dark_edges(warped, min_luma=10, max_dark_ratio=0.15)
    
    # Final cleanup: crop tighter to content (removes perspective artifacts)
    warped = crop_to_content_bounds(warped, margin=50, min_luma=10, max_dark_ratio=0.05)
    
    global_corners = dst_offset
    content_bounds = (warped.shape[1], warped.shape[0])
    
    return warped, global_corners, M, content_bounds


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
    """Add 1-3 gradient overlays."""
    h, w = img.shape[:2]
    num_gradients = random.randint(1, 3)
    
    for _ in range(num_gradients):
        grad_type = random.choice(['radial', 'horizontal', 'vertical', 'diagonal'])
        alpha = random.uniform(0.05, 0.20)
        direction = random.choice([-1, 1])
        
        if grad_type == 'radial':
            cx, cy = w/2, h/2
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
            max_dist = np.sqrt(cx**2 + cy**2)
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


def crop_to_content_bounds(img, margin=50, min_luma=10, max_dark_ratio=0.05):
    """
    Crop image to content bounds, removing dark edge artifacts from perspective.
    
    Only trims edges that are predominantly BLACK (luma < 10), not just dark.
    This avoids trimming into photo content which may have legitimate dark areas.
    
    Args:
        img: Input image (BGR)
        margin: Minimum margin around content (default 50px)
        min_luma: Pixels below this luma are considered black border (default 10)
        max_dark_ratio: Edge is trimmed if >5% of edge pixels are near-black
    
    Returns:
        Cropped image with clean edges
    """
    h, w = img.shape[:2]
    if h <= 2 * margin + 100 or w <= 2 * margin + 100:
        return img
    
    # Calculate luma for edge detection
    if len(img.shape) == 3:
        luma = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]
    else:
        luma = img.astype(float)
    
    # Iteratively trim dark edges
    crop_top = 0
    crop_bottom = h
    crop_left = 0
    crop_right = w
    
    max_iterations = 300
    for _ in range(max_iterations):
        changed = False
        
        current_h = crop_bottom - crop_top
        current_w = crop_right - crop_left
        
        # Safety: don't crop too small (keep at least 600px in each dimension)
        if current_h < 600 or current_w < 600:
            break
        
        # Check top edge - only trim if predominantly near-black
        if current_h > 2 * margin:
            top_row = luma[crop_top, crop_left:crop_right]
            # Count pixels that are NEAR-BLACK (luma < min_luma)
            top_black = np.sum(top_row < min_luma)
            # Only trim if >5% of edge is near-black
            if top_black > current_w * max_dark_ratio:
                crop_top += 1
                changed = True
        
        # Check bottom edge
        if current_h > 2 * margin and crop_bottom > crop_top + margin:
            bottom_row = luma[crop_bottom - 1, crop_left:crop_right]
            bottom_black = np.sum(bottom_row < min_luma)
            if bottom_black > current_w * max_dark_ratio:
                crop_bottom -= 1
                changed = True
        
        # Check left edge
        if current_w > 2 * margin:
            left_col = luma[crop_top:crop_bottom, crop_left]
            left_black = np.sum(left_col < min_luma)
            if left_black > current_h * max_dark_ratio:
                crop_left += 1
                changed = True
        
        # Check right edge
        if current_w > 2 * margin and crop_right > crop_left + margin:
            right_col = luma[crop_top:crop_bottom, crop_right - 1]
            right_black = np.sum(right_col < min_luma)
            if right_black > current_h * max_dark_ratio:
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
    
    if final_h < 600 or final_w < 600:
        # Too small after trimming - return with padding instead
        pad = 50
        padded = np.full((h + 2*pad, w + 2*pad, 3), 128, dtype=np.uint8)
        padded[pad:pad+h, pad:pad+w] = img
        return padded
    
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
    """Add screen-mode glare."""
    if random.random() < 0.35:
        h, w = img.shape[:2]
        num_flares = random.randint(1, 3)
        for _ in range(num_flares):
            cx = random.uniform(w * 0.2, w * 0.8)
            cy = random.uniform(h * 0.2, h * 0.8)
            rx = random.uniform(w * 0.05, w * 0.15)
            ry = random.uniform(h * 0.05, h * 0.15)
            angle = random.uniform(0, math.pi)
            
            y, x = np.ogrid[:h, :w]
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            xr = (x - cx) * cos_a - (y - cy) * sin_a
            yr = (x - cx) * sin_a + (y - cy) * cos_a
            flare = np.maximum(0, 1 - (xr/rx)**2 - (yr/ry)**2)
            flare = cv2.GaussianBlur(flare.astype(np.float32), (31, 31), 0)
            
            img_f = img.astype(np.float32) / 255.0
            flare_bgr = np.stack([flare * 255] * 3, axis=-1)
            flare_f = flare_bgr / 255.0
            img_f = 1 - (1 - img_f) * (1 - flare_f * 0.3)
            img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return img


def create_drop_shadow(photo):
    """
    Create a drop shadow for a photo.
    
    Uses screen-mode compositing for realistic soft shadows.
    """
    h, w = photo.shape[:2]
    
    # Shadow parameters
    shadow_opacity = random.uniform(0.15, 0.30)
    blur_sigma = random.uniform(8, 20)
    shadow_offset = random.randint(10, 25)
    
    # Direction (lower-right is most common for natural light)
    if random.random() < 0.7:
        offset_x = shadow_offset
        offset_y = shadow_offset
    else:
        angle = random.uniform(0, 360)
        offset_x = int(math.cos(math.radians(angle)) * shadow_offset)
        offset_y = int(math.sin(math.radians(angle)) * shadow_offset)
    
    # Create shadow mask (slightly larger than photo)
    shadow_h = h + abs(offset_y) + 20
    shadow_w = w + abs(offset_x) + 20
    shadow_mask = np.zeros((shadow_h, shadow_w), dtype=np.float32)
    
    # Photo area in shadow space
    px1 = max(0, offset_x) + 10
    py1 = max(0, offset_y) + 10
    px2 = px1 + w
    py2 = py1 + h
    
    # Fill with photo shape
    cv2.rectangle(shadow_mask, (px1, py1), (px2, py2), 1.0, -1)
    
    # Blur for soft edges
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), blur_sigma)
    
    # Apply shadow using screen blend mode
    photo_f = photo.astype(np.float32) / 255.0
    
    if photo.shape[2] == 4:
        alpha = photo_f[:, :, 3:4]
        
        # Clip shadow to photo bounds
        sx1 = max(0, -offset_x)
        sy1 = max(0, -offset_y)
        sx2 = min(w, w - offset_x)
        sy2 = min(h, h - offset_y)
        
        shadow = shadow_mask[sy1:sy2+abs(offset_y), sx1:sx2+abs(offset_x)]
        
        # Screen blend: 1 - (1 - a) * (1 - b)
        darkened = 1 - (1 - photo_f[:, :, :3]) * (1 - shadow[:, :, np.newaxis] * shadow_opacity)
        photo_f[:, :, :3] = darkened
        
        # Slight alpha reduction at shadow edges
        photo_f[:, :, 3] *= (1 - shadow * shadow_opacity * 0.2)
    else:
        # No alpha - just darken
        shadow = shadow_mask[abs(offset_y):abs(offset_y)+h, abs(offset_x):abs(offset_x)+w]
        photo_f = 1 - (1 - photo_f) * (1 - shadow[:, :, np.newaxis] * shadow_opacity)
    
    return np.clip(photo_f * 255, 0, 255).astype(np.uint8)


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
    
    output_dir = Path("../data/examples_v5")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    CANVAS_W, CANVAS_H = 2400, 1350
    
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
            
            # Add drop shadow (uses screen blend mode)
            photo = create_drop_shadow(photo)
            
            # Rotate photo
            rotation = placement['rotation']
            photo_rotated = rotate_photo(photo, rotation)
            
            # Composite photo at center position
            center_x = placement['center_x']
            center_y = placement['center_y']
            canvas = composite_photo_at_center(canvas, photo_rotated, center_x, center_y)
            
            # Store polygon for global warp transformation
            placed_photos.append({
                'polygon': placement['polygon'].copy(),
                'rotation': rotation,
            })
            
            verification_stats['total_photos'] += 1
            if abs(rotation) > 5:
                verification_stats['photos_with_rotation'] += 1
        
        # Apply ONE global perspective warp to entire composite
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        
        photo_corners_list = [photo_data['polygon'] for photo_data in placed_photos]
        
        warped_canvas, global_corners, transform_matrix, content_bounds = apply_global_perspective(
            canvas_bgr, CANVAS_W, CANVAS_H,
            photo_corners=photo_corners_list,
            crop_margin=400  # Large margin to keep photos well inside image bounds
        )
        
        # Transform all photo corners through the global warp
        final_photos = []
        out_w, out_h = warped_canvas.shape[1], warped_canvas.shape[0]
        
        for photo_data in placed_photos:
            polygon = photo_data['polygon']
            
            # Transform through global perspective
            warped_corners = calculate_warped_photo_corners(polygon, transform_matrix)
            
            # Don't clip - corners may legitimately extend beyond cropped bounds
            # This represents the actual perspective transformation
            
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
