#!/usr/bin/env python3
"""
Photo Pose Detector - Synthetic Training Data Generator (v34)
===============================================================

This module generates synthetic training images for TWO YOLO models:
1. Detection Model: Axis-aligned bounding boxes around photos
2. Pose Model: 4 corner keypoints per photo (LL, UL, UR, LR)

KEY IMPROVEMENTS IN v34:
- LARGER CANVAS PADDING: 300px padding to prevent black edges after perspective warp
- SCALED DOWN SHADOWS: Max blur reduced to 2 for more subtle effects
- VERIFIED CORNER TRACKING: Color-based verification confirms corner accuracy

Author: Photo Pose Detector Project
Version: 34 - Fixed canvas size and shadow scaling
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
# CONFIGURATION CONSTANTS (v34)
# =============================================================================

CONFIG = {
    # Canvas settings (640x640 matches YOLO input)
    'CANVAS_SIZE': 640,
    
    # EXTRA PADDING: Much larger to prevent black edges after perspective warp
    # Photos placed in center of padded canvas, then cropped after warp
    'CANVAS_PADDING': 300,  # 300px on each side (total: 640 + 600 = 1240px)
    
    # Photo packing - 1-4 photos with PROPER SIZING
    'NUM_PHOTOS_MIN': 1,
    'NUM_PHOTOS_MAX': 4,
    
    # Photo sizes: SMALLER to fit multiple photos with margins
    'PHOTO_SIZE_MIN': 180,
    'PHOTO_SIZE_MAX': 480,
    
    # Margin from canvas edge (pixels)
    'EDGE_MARGIN': 40,
    
    # Spacing between photos
    'PHOTO_SPACING': 30,
    
    # Photo rotation
    'ROTATION_RANGE': 30,
    
    # Photo edges (focus simulation) - SMALLER
    'EDGE_BLUR_MIN': 0,
    'EDGE_BLUR_MAX': 2,
    
    # Drop shadows - MUCH SMALLER for 640x640
    'SHADOW_OFFSET_MAX': 2,   # Reduced from 3 to 2
    'SHADOW_BLUR_MIN': 1,    # Kept at 1
    'SHADOW_BLUR_MAX': 2,    # Reduced from 4 to 2
    
    # Glare effects
    'GLARE_PROBABILITY': 0.50,
    'GLARE_COUNT_MIN': 2,
    'GLARE_COUNT_MAX': 4,
    'GLARE_SIZE_MIN': 0.20,
    'GLARE_SIZE_MAX': 0.40,
    'GLARE_OPACITY_MIN': 0.60,
    'GLARE_OPACITY_MAX': 1.00,
    'GLARE_BLUR': 15,
    
    # Global perspective warp - VARIABLE (5-20%)
    'PERSPECTIVE_STRENGTH_MIN': 0.05,
    'PERSPECTIVE_STRENGTH_MAX': 0.20,
    
    # Output - Margin around content
    'CROP_MARGIN': 60,
}


# =============================================================================
# POLYGON UTILITIES - VERIFIED CORRECT
# =============================================================================

def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """
    Calculate the 4 corner coordinates of a rotated rectangle.
    
    This must match where rotate_photo() places corners when the photo
    is centered at (center_x, center_y) in placement space.
    
    The correct formula is:
    polygon = (center_x - width/2, center_y - height/2) + M_raw @ corner
    
    Where M_raw is cv2.getRotationMatrix2D WITHOUT the canvas expansion offset.
    The rotate_photo() function adds canvas expansion to M, but when compositing,
    the canvas top-left is (center_x - new_w/2, center_y - new_h/2), which cancels out
    the canvas expansion offset.
    """
    import cv2
    import numpy as np
    
    if abs(rotation) < 1:
        hw, hh = width / 2, height / 2
        return np.array([
            [center_x - hw, center_y - hh],
            [center_x + hw, center_y - hh],
            [center_x + hw, center_y + hh],
            [center_x - hw, center_y + hh]
        ], dtype=np.float32)
    
    # Build rotation matrix around photo center (WITHOUT canvas expansion offset)
    photo_center = (width / 2, height / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    # Canvas offset: this is where the photo center is relative to origin
    canvas_offset = (center_x - width / 2, center_y - height / 2)
    
    # Corners in PHOTO SPACE (at the edges of the original photo)
    corners_photo = np.array([
        [0, 0],              # TL
        [width, 0],          # TR
        [width, height],     # BR
        [0, height]          # BL
    ], dtype=np.float32)
    
    # Transform corners: polygon = canvas_offset + M_raw @ corner
    corners_final = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        corners_final[i, 0] = canvas_offset[0] + rotated[0]
        corners_final[i, 1] = canvas_offset[1] + rotated[1]
    
    return corners_final


def reorder_corners_to_llulur(corners):
    """
    Reorder 4 corners to LL, UL, UR, LR based on position.
    
    For a photo in the image:
    - LL (Lower-Left): left side, bottom (small x, large y)
    - UL (Upper-Left): left side, top (small x, small y)
    - UR (Upper-Right): right side, top (large x, small y)
    - LR (Lower-Right): right side, bottom (large x, large y)
    
    Polygon order: TL=0, TR=1, BR=2, BL=3
    YOLO order:     LL=0, UL=1, UR=2, LR=3
    
    Mapping: [BL, TL, TR, BR] -> [LL, UL, UR, LR]
    """
    corners = np.array(corners)
    
    # Sort by x to determine left/right
    sorted_by_x = sorted(enumerate(corners), key=lambda i_c: corners[i_c[0]][0])
    
    # First two = left side (small x), Last two = right side (large x)
    left_indices = [i for i, c in sorted_by_x[:2]]
    right_indices = [i for i, c in sorted_by_x[2:]]
    
    # Within left side: sort by y (small y = UL, large y = LL)
    left_corners = [(corners[i][0], corners[i][1]) for i in left_indices]
    left_corners.sort(key=lambda c: c[1])
    ul = np.array(left_corners[0])
    ll = np.array(left_corners[1])
    
    # Within right side: sort by y (small y = UR, large y = LR)
    right_corners = [(corners[i][0], corners[i][1]) for i in right_indices]
    right_corners.sort(key=lambda c: c[1])
    ur = np.array(right_corners[0])
    lr = np.array(right_corners[1])
    
    return np.array([ll, ul, ur, lr], dtype=np.float32)


def polygon_to_yolo_order(polygon_corners):
    """
    Convert polygon corners (TL, TR, BR, BL) to YOLO order (LL, UL, UR, LR).
    
    Mapping: polygon[0]=TL->UL, polygon[1]=TR->UR, polygon[2]=BR->LR, polygon[3]=BL->LL
    YOLO order: [LL, UL, UR, LR]
    Polygon:    [TL, TR, BR, BL]
    Mapping:    [3,  0,  1,  2]  (BL->LL, TL->UL, TR->UR, BR->LR)
    """
    yolo_order = [3, 0, 1, 2]
    return polygon_corners[yolo_order]


# =============================================================================
# PHOTO PACKING - SIMPLE GRID APPROACH (v33)
# =============================================================================

def pack_photos_grid(canvas_w, canvas_h):
    """
    Pack photos using a simple grid/random approach.
    Ensures photos stay within bounds with proper margins.
    """
    import random as rnd
    rnd.seed(None)
    
    num_photos = rnd.randint(CONFIG['NUM_PHOTOS_MIN'], CONFIG['NUM_PHOTOS_MAX'])
    
    edge_margin = CONFIG['EDGE_MARGIN']
    spacing = CONFIG['PHOTO_SPACING']
    usable_w = canvas_w - 2 * edge_margin
    usable_h = canvas_h - 2 * edge_margin
    
    placements = []
    
    if num_photos == 1:
        # Single centered photo
        size = rnd.randint(CONFIG['PHOTO_SIZE_MIN'], int(CONFIG['PHOTO_SIZE_MAX'] * 0.8))
        height = int(size * rnd.uniform(0.75, 0.85))
        
        cx = canvas_w / 2 + rnd.uniform(-50, 50)
        cy = canvas_h / 2 + rnd.uniform(-50, 50)
        rotation = rnd.uniform(-CONFIG['ROTATION_RANGE'], CONFIG['ROTATION_RANGE']) if rnd.random() < 0.15 else 0
        
        placements.append({
            'width': size,
            'height': height,
            'center_x': cx,
            'center_y': cy,
            'rotation': rotation,
            'polygon': get_rotated_polygon(size, height, cx, cy, rotation)
        })
    
    elif num_photos == 2:
        layout = rnd.choice(['horizontal', 'vertical'])
        
        if layout == 'horizontal':
            photo_w = int((usable_w - spacing) / 2)
            photo_h = int(photo_w * rnd.uniform(0.75, 0.85))
            
            total_h = max(photo_h, int(usable_h * 0.4))
            photo_h = min(photo_h, usable_h - 20)
            
            cx1 = edge_margin + photo_w / 2 + rnd.uniform(-20, 20)
            cx2 = canvas_w - edge_margin - photo_w / 2 + rnd.uniform(-20, 20)
            cy = canvas_h / 2 + rnd.uniform(-40, 40)
            
            rot1 = rnd.uniform(-15, 15) if rnd.random() < 0.3 else 0
            rot2 = rnd.uniform(-15, 15) if rnd.random() < 0.3 else 0
            
            placements.append({
                'width': photo_w, 'height': photo_h,
                'center_x': cx1, 'center_y': cy,
                'rotation': rot1,
                'polygon': get_rotated_polygon(photo_w, photo_h, cx1, cy, rot1)
            })
            placements.append({
                'width': photo_w, 'height': photo_h,
                'center_x': cx2, 'center_y': cy,
                'rotation': rot2,
                'polygon': get_rotated_polygon(photo_w, photo_h, cx2, cy, rot2)
            })
        else:
            photo_h = int((usable_h - spacing) / 2)
            photo_w = int(photo_h / rnd.uniform(0.75, 0.85))
            photo_w = min(photo_w, usable_w - 20)
            
            cx = canvas_w / 2 + rnd.uniform(-40, 40)
            cy1 = edge_margin + photo_h / 2 + rnd.uniform(-20, 20)
            cy2 = canvas_h - edge_margin - photo_h / 2 + rnd.uniform(-20, 20)
            
            rot1 = rnd.uniform(-15, 15) if rnd.random() < 0.3 else 0
            rot2 = rnd.uniform(-15, 15) if rnd.random() < 0.3 else 0
            
            placements.append({
                'width': photo_w, 'height': photo_h,
                'center_x': cx, 'center_y': cy1,
                'rotation': rot1,
                'polygon': get_rotated_polygon(photo_w, photo_h, cx, cy1, rot1)
            })
            placements.append({
                'width': photo_w, 'height': photo_h,
                'center_x': cx, 'center_y': cy2,
                'rotation': rot2,
                'polygon': get_rotated_polygon(photo_w, photo_h, cx, cy2, rot2)
            })
    
    elif num_photos == 3:
        layout = rnd.choice(['triangle', 'L', 'row'])
        
        if layout == 'triangle':
            center_size = int(min(usable_w, usable_h) * 0.45)
            center_h = int(center_size * rnd.uniform(0.75, 0.85))
            
            cx_c = canvas_w / 2 + rnd.uniform(-30, 30)
            cy_c = canvas_h / 2 + rnd.uniform(-30, 30)
            
            small_size = int(center_size * 0.6)
            small_h = int(small_size * rnd.uniform(0.75, 0.85))
            
            placements.append({
                'width': center_size, 'height': center_h,
                'center_x': cx_c, 'center_y': cy_c,
                'rotation': 0,
                'polygon': get_rotated_polygon(center_size, center_h, cx_c, cy_c, 0)
            })
            
            cx_l = edge_margin + small_size / 2 + rnd.uniform(5, 15)
            cy_l = canvas_h - edge_margin - small_h / 2 + rnd.uniform(-20, 20)
            rot_l = rnd.uniform(-20, 20) if rnd.random() < 0.4 else 0
            placements.append({
                'width': small_size, 'height': small_h,
                'center_x': cx_l, 'center_y': cy_l,
                'rotation': rot_l,
                'polygon': get_rotated_polygon(small_size, small_h, cx_l, cy_l, rot_l)
            })
            
            cx_r = canvas_w - edge_margin - small_size / 2 + rnd.uniform(-15, -5)
            cy_r = canvas_h - edge_margin - small_h / 2 + rnd.uniform(-20, 20)
            rot_r = rnd.uniform(-20, 20) if rnd.random() < 0.4 else 0
            placements.append({
                'width': small_size, 'height': small_h,
                'center_x': cx_r, 'center_y': cy_r,
                'rotation': rot_r,
                'polygon': get_rotated_polygon(small_size, small_h, cx_r, cy_r, rot_r)
            })
        
        elif layout == 'row':
            photo_w = int((usable_w - 2 * spacing) / 3)
            photo_h = int(photo_w / rnd.uniform(0.75, 0.85))
            photo_h = min(photo_h, usable_h - 20)
            
            cy = canvas_h / 2 + rnd.uniform(-30, 30)
            
            for i in range(3):
                cx = edge_margin + photo_w / 2 + i * (photo_w + spacing) + rnd.uniform(-10, 10)
                rot = rnd.uniform(-15, 15) if rnd.random() < 0.3 else 0
                placements.append({
                    'width': photo_w, 'height': photo_h,
                    'center_x': cx, 'center_y': cy,
                    'rotation': rot,
                    'polygon': get_rotated_polygon(photo_w, photo_h, cx, cy, rot)
                })
        else:
            top_w = int((usable_w - spacing) / 2)
            top_h = int(top_w / rnd.uniform(0.75, 0.85))
            
            bottom_w = int(top_w * 0.8)
            bottom_h = int(bottom_w / rnd.uniform(0.75, 0.85))
            
            cx = canvas_w / 2 + rnd.uniform(-20, 20)
            
            cx_tl = edge_margin + top_w / 2 + rnd.uniform(-10, 10)
            cy_t = edge_margin + top_h / 2 + rnd.uniform(-10, 10)
            placements.append({
                'width': top_w, 'height': top_h,
                'center_x': cx_tl, 'center_y': cy_t,
                'rotation': rnd.uniform(-10, 10) if rnd.random() < 0.3 else 0,
                'polygon': get_rotated_polygon(top_w, top_h, cx_tl, cy_t, 
                    rnd.uniform(-10, 10) if rnd.random() < 0.3 else 0)
            })
            
            cx_tr = canvas_w - edge_margin - top_w / 2 + rnd.uniform(-10, 10)
            placements.append({
                'width': top_w, 'height': top_h,
                'center_x': cx_tr, 'center_y': cy_t,
                'rotation': rnd.uniform(-10, 10) if rnd.random() < 0.3 else 0,
                'polygon': get_rotated_polygon(top_w, top_h, cx_tr, cy_t,
                    rnd.uniform(-10, 10) if rnd.random() < 0.3 else 0)
            })
            
            cy_b = canvas_h - edge_margin - bottom_h / 2 + rnd.uniform(-10, 10)
            placements.append({
                'width': bottom_w, 'height': bottom_h,
                'center_x': cx, 'center_y': cy_b,
                'rotation': rnd.uniform(-20, 20) if rnd.random() < 0.4 else 0,
                'polygon': get_rotated_polygon(bottom_w, bottom_h, cx, cy_b,
                    rnd.uniform(-20, 20) if rnd.random() < 0.4 else 0)
            })
    
    else:  # num_photos == 4
        layout = rnd.choice(['grid', 'diamond', 'row'])
        
        if layout == 'grid':
            cell_w = int((usable_w - spacing) / 2)
            cell_h = int((usable_h - spacing) / 2)
            photo_w = int(min(cell_w, cell_h) * 0.9)
            photo_h = int(photo_w / rnd.uniform(0.75, 0.85))
            
            cx = canvas_w / 2 + rnd.uniform(-20, 20)
            cy = canvas_h / 2 + rnd.uniform(-20, 20)
            
            positions = [
                (-photo_w/2 - spacing/2, -photo_h/2 - spacing/2),
                (photo_w/2 + spacing/2, -photo_h/2 - spacing/2),
                (-photo_w/2 - spacing/2, photo_h/2 + spacing/2),
                (photo_w/2 + spacing/2, photo_h/2 + spacing/2),
            ]
            
            for ox, oy in positions:
                px = cx + ox + rnd.uniform(-5, 5)
                py = cy + oy + rnd.uniform(-5, 5)
                rot = rnd.uniform(-20, 20) if rnd.random() < 0.4 else rnd.uniform(-10, 10)
                placements.append({
                    'width': photo_w, 'height': photo_h,
                    'center_x': px, 'center_y': py,
                    'rotation': rot,
                    'polygon': get_rotated_polygon(photo_w, photo_h, px, py, rot)
                })
        
        elif layout == 'diamond':
            center_size = int(min(usable_w, usable_h) * 0.35)
            center_h = int(center_size * rnd.uniform(0.75, 0.85))
            
            small_size = int(center_size * 0.65)
            small_h = int(small_size * rnd.uniform(0.75, 0.85))
            
            cx = canvas_w / 2 + rnd.uniform(-20, 20)
            cy = canvas_h / 2 + rnd.uniform(-20, 20)
            
            placements.append({
                'width': center_size, 'height': center_h,
                'center_x': cx, 'center_y': cy,
                'rotation': 0,
                'polygon': get_rotated_polygon(center_size, center_h, cx, cy, 0)
            })
            
            offsets = [
                (0, -center_h * 0.8),
                (center_size * 0.7, 0),
                (0, center_h * 0.8),
                (-center_size * 0.7, 0),
            ]
            
            for ox, oy in offsets:
                px = cx + ox + rnd.uniform(-10, 10)
                py = cy + oy + rnd.uniform(-10, 10)
                rot = rnd.uniform(-25, 25) if rnd.random() < 0.5 else 0
                placements.append({
                    'width': small_size, 'height': small_h,
                    'center_x': px, 'center_y': py,
                    'rotation': rot,
                    'polygon': get_rotated_polygon(small_size, small_h, px, py, rot)
                })
        
        else:
            photo_w = int((usable_w - 3 * spacing) / 4)
            photo_h = int(photo_w / rnd.uniform(0.75, 0.85))
            photo_h = min(photo_h, int(usable_h * 0.5))
            
            cy = canvas_h / 2 + rnd.uniform(-30, 30)
            
            for i in range(4):
                cx = edge_margin + photo_w / 2 + i * (photo_w + spacing) + rnd.uniform(-8, 8)
                rot = rnd.uniform(-20, 20) if rnd.random() < 0.4 else rnd.uniform(-10, 10)
                placements.append({
                    'width': photo_w, 'height': photo_h,
                    'center_x': cx, 'center_y': cy,
                    'rotation': rot,
                    'polygon': get_rotated_polygon(photo_w, photo_h, cx, cy, rot)
                })
    
    # Add shadow parameters
    for p in placements:
        p['shadow_params'] = {
            'offset_x': rnd.choice([-1, 0, 1]) * rnd.randint(0, CONFIG['SHADOW_OFFSET_MAX']),
            'offset_y': rnd.choice([-1, 0, 1]) * rnd.randint(0, CONFIG['SHADOW_OFFSET_MAX']),
            'blur_sigma': rnd.uniform(CONFIG['SHADOW_BLUR_MIN'], CONFIG['SHADOW_BLUR_MAX']),
            'opacity': rnd.choice([rnd.uniform(0.10, 0.20), rnd.uniform(0.30, 0.45)])
        }
        p['circle_radius'] = math.sqrt(p['width']**2 + p['height']**2) / 2 + 5
    
    return placements


def spiral_pack_photos(canvas_w, canvas_h):
    """Pack photos - just calls the grid function."""
    return pack_photos_grid(canvas_w, canvas_h)


# =============================================================================
# PHOTO EFFECTS
# =============================================================================

def rotate_photo(photo, angle):
    """Rotate a photo by specified degrees."""
    h, w = photo.shape[:2]
    
    if abs(angle) < 1:
        return photo
    
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(
        photo, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128, 0)
    )
    
    return rotated


def composite_photo_at_center(canvas, photo, cx, cy):
    """Composite photo onto canvas with center at (cx, cy)."""
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    
    top_left_x = int(cx - pw / 2)
    top_left_y = int(cy - ph / 2)
    
    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = pw, ph
    dst_x1, dst_y1 = top_left_x, top_left_y
    dst_x2, dst_y2 = top_left_x + pw, top_left_y + ph
    
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


def apply_photo_shadow(canvas, photo, cx, cy, offset_x, offset_y, blur_sigma, opacity, rotation=0):
    """Render a drop shadow onto the canvas - SMALLER VERSION."""
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    num_channels = canvas.shape[2]
    
    rot_rad = math.radians(rotation)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    
    rotated_offset_x = offset_x * cos_r - offset_y * sin_r
    rotated_offset_y = offset_x * sin_r + offset_y * cos_r
    
    shadow_cx = cx + rotated_offset_x * 0.5
    shadow_cy = cy + rotated_offset_y * 0.5
    
    # Smaller blur pad
    blur_pad = int(blur_sigma * 2) + 3
    shadow_w = pw + blur_pad * 2
    shadow_h = ph + blur_pad * 2
    shadow_mask = np.zeros((shadow_h, shadow_w), dtype=np.float32)
    
    shadow_mask[blur_pad:blur_pad+ph, blur_pad:blur_pad+pw] = 1.0
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=blur_sigma)
    
    center_rot = (shadow_w / 2, shadow_h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center_rot, rotation, 1.0)
    shadow_mask = cv2.warpAffine(shadow_mask, rot_matrix, (shadow_w, shadow_h), 
                                   borderValue=0, flags=cv2.INTER_LINEAR)
    
    if shadow_mask.max() > 0:
        shadow_mask = shadow_mask / shadow_mask.max()
    
    shadow_top_left_x = int(shadow_cx - shadow_w / 2)
    shadow_top_left_y = int(shadow_cy - shadow_h / 2)
    
    mask_h, mask_w = shadow_mask.shape
    
    canvas_f = canvas.astype(np.float32) / 255.0
    
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
        
        for c in range(num_channels):
            canvas_f[clip_y1:clip_y2, clip_x1:clip_x2, c] *= (1 - shadow_vals)
    
    canvas[:, :, :num_channels] = np.clip(canvas_f * 255, 0, 255).astype(np.uint8)
    return canvas


def fast_photo_manipulation(img):
    """Apply brightness/contrast/saturation/gamma adjustments."""
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
    """Add glare highlights using screen blend."""
    if random.random() < 0.5:
        h, w = img.shape[:2]
        
        num_flares = random.randint(CONFIG['GLARE_COUNT_MIN'], CONFIG['GLARE_COUNT_MAX'])
        for _ in range(num_flares):
            img_f = img.astype(np.float32) / 255.0
            
            cx = random.uniform(w * 0.15, w * 0.85)
            cy = random.uniform(h * 0.1, h * 0.7)
            
            rx = random.uniform(w * 0.20, w * 0.40)
            ry = random.uniform(h * 0.20, h * 0.40)
            
            y, x = np.ogrid[:h, :w]
            
            flare = np.maximum(0, 1 - (x - cx)**2 / (rx**2) - (y - cy)**2 / (ry**2))
            flare = cv2.GaussianBlur(flare.astype(np.float32), 
                                     (CONFIG['GLARE_BLUR'], CONFIG['GLARE_BLUR']), 0)
            
            opacity = random.uniform(CONFIG['GLARE_OPACITY_MIN'], CONFIG['GLARE_OPACITY_MAX'])
            flare_f = flare[:, :, np.newaxis]
            img_f = 1 - (1 - img_f) * (1 - flare_f * opacity)
            
            img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    
    return img


def add_rgba_alpha(img):
    """Convert BGR image to BGRA with alpha channel."""
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = 255
    return bgra


def blur_alpha_edges(photo, edge_effect=None):
    """Create soft/blurred edges on ALL photo edges."""
    if photo.shape[2] != 4:
        return photo
    
    h, w = photo.shape[:2]
    
    if edge_effect is None:
        edge_effect = random.uniform(CONFIG['EDGE_BLUR_MIN'], CONFIG['EDGE_BLUR_MAX'])
    
    if edge_effect < 1:
        return photo
    
    alpha = photo[:, :, 3].astype(np.float32) / 255.0
    rgb = photo[:, :, :3].astype(np.float32)
    
    y_dist = np.minimum(np.arange(h)[:, np.newaxis], np.arange(h)[::-1, np.newaxis])
    x_dist = np.minimum(np.arange(w), np.arange(w)[::-1])
    edge_dist = np.minimum(y_dist, x_dist)
    
    fade_dist = edge_effect * 2.5
    soft_alpha = np.clip(edge_dist / fade_dist, 0, 1)
    
    blur_size = int(edge_effect * 2) * 2 + 1
    blur_size = max(blur_size, 5)
    rgb_blurred = cv2.GaussianBlur(rgb, (blur_size, blur_size), 0)
    
    blend_factor = 1.0 - soft_alpha
    rgb_soft = rgb * (1 - blend_factor)[:, :, np.newaxis] + rgb_blurred * blend_factor[:, :, np.newaxis]
    
    soft_alpha_final = alpha * soft_alpha
    
    photo[:, :, :3] = np.clip(rgb_soft, 0, 255).astype(np.uint8)
    photo[:, :, 3] = np.clip(soft_alpha_final * 255, 0, 255).astype(np.uint8)
    
    return photo


# =============================================================================
# BACKGROUND GENERATION
# =============================================================================

def random_base_background(w, h):
    """Generate a random background with controlled brightness and saturation."""
    import colorsys
    
    rand_val = random.random()
    
    if rand_val < CONFIG.get('BG_DARK_RATIO', 0.30):
        lightness = random.uniform(0.04, 0.28)
        saturation = random.uniform(0, 0.04)
    elif rand_val < CONFIG.get('BG_DARK_RATIO', 0.30) + CONFIG.get('BG_LIGHT_RATIO', 0.30):
        lightness = random.uniform(0.69, 0.96)
        saturation = random.uniform(0, 0.04)
    else:
        lightness = random.uniform(0.19, 0.86)
        saturation = random.uniform(0.04, 0.40)
    
    hue = random.uniform(0, 1)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    color = (int(r * 255), int(g * 255), int(b * 255))
    
    img = np.ones((h, w, 3), dtype=np.float32) * np.array(color, dtype=np.float32)
    
    noise_sigma = random.uniform(1, 4)
    noise = np.random.normal(0, noise_sigma, (h, w, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    img = apply_3_linear_gradients(img)
    
    return img


def apply_3_linear_gradients(img):
    """Apply 3 random linear gradients with screen blend."""
    h, w = img.shape[:2]
    img_f = img.astype(np.float32) / 255.0
    
    for _ in range(3):
        direction = random.choice(['horizontal', 'vertical', 'diagonal_tl', 'diagonal_tr'])
        
        if direction == 'horizontal':
            x = np.linspace(0, 1, w)
            gradient = np.tile(x, (h, 1))
        elif direction == 'vertical':
            y = np.linspace(0, 1, h)[:, np.newaxis]
            gradient = np.tile(y, (1, w))
        elif direction == 'diagonal_tl':
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)[:, np.newaxis]
            gradient = x + y
            gradient = gradient / gradient.max()
        else:
            x = np.linspace(1, 0, w)
            y = np.linspace(0, 1, h)[:, np.newaxis]
            gradient = x + y
            gradient = gradient / gradient.max()
        
        opacity = random.uniform(0, 0.20)
        overlay = gradient[:, :, np.newaxis] * opacity
        result = 1.0 - (1.0 - img_f) * (1.0 - overlay)
        img_f = result
    
    return np.clip(img_f * 255, 0, 255).astype(np.uint8)


def apply_texture_overlay(canvas):
    """Apply a random texture overlay to the background."""
    textures_dir = Path("/Users/krys.petrie/dev/photo-pose-detector/textures")
    
    if not textures_dir.exists():
        return canvas
    
    textures = list(textures_dir.glob("*.jpg")) + list(textures_dir.glob("*.png"))
    if not textures:
        return canvas
    
    texture_path = random.choice(textures)
    texture = cv2.imread(str(texture_path))
    
    if texture is None:
        return canvas
    
    texture = cv2.resize(texture, (canvas.shape[1], canvas.shape[0]))
    
    flip_code = random.choice([-1, 0, 1, None])
    if flip_code is not None:
        texture = cv2.flip(texture, flip_code)
    
    opacity = random.uniform(0, 0.40)
    use_screen = random.choice([True, False])
    
    canvas_f = canvas.astype(np.float32) / 255.0
    texture_f = texture.astype(np.float32) / 255.0
    
    if use_screen:
        blended = 1.0 - (1.0 - canvas_f) * (1.0 - texture_f)
    else:
        blended = canvas_f * texture_f
    
    result = canvas_f * (1 - opacity) + blended * opacity
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)


# =============================================================================
# GLOBAL PERSPECTIVE WARP - MORE VARIABLE (5-25%)
# =============================================================================

def apply_global_perspective(canvas, canvas_w, canvas_h, photo_corners=None, crop_margin=60):
    """
    Apply a SINGLE perspective warp to the entire canvas composite.
    
    VARIABLE PERSPECTIVE: 5-25% strength with MUCH greater variety.
    """
    src_corners = np.array([
        [0, 0],
        [canvas_w - 1, 0],
        [canvas_w - 1, canvas_h - 1],
        [0, canvas_h - 1]
    ], dtype=np.float32)
    
    perspective_strength = random.uniform(
        CONFIG['PERSPECTIVE_STRENGTH_MIN'],
        CONFIG['PERSPECTIVE_STRENGTH_MAX']
    )
    direction = random.randint(0, 7)
    max_offset_x = canvas_w * perspective_strength
    max_offset_y = canvas_h * perspective_strength
    
    # Initialize ALL offsets to zero
    tl_offset_x = tr_offset_x = bl_offset_x = br_offset_x = 0.0
    tl_offset_y = tr_offset_y = bl_offset_y = br_offset_y = 0.0
    
    # 8 different perspective directions
    if direction == 0:
        tl_offset_x = random.uniform(max_offset_x * 0.6, max_offset_x)
        tr_offset_x = random.uniform(max_offset_x * 0.3, max_offset_x * 0.8)
        bl_offset_x = random.uniform(max_offset_x * 0.6, max_offset_x)
        br_offset_x = random.uniform(max_offset_x * 0.3, max_offset_x * 0.8)
    elif direction == 1:
        tl_offset_x = random.uniform(-max_offset_x * 0.8, -max_offset_x * 0.3)
        tr_offset_x = random.uniform(-max_offset_x, -max_offset_x * 0.6)
        bl_offset_x = random.uniform(-max_offset_x * 0.8, -max_offset_x * 0.3)
        br_offset_x = random.uniform(-max_offset_x, -max_offset_x * 0.6)
    elif direction == 2:
        tl_offset_y = random.uniform(max_offset_y * 0.6, max_offset_y)
        tr_offset_y = random.uniform(max_offset_y * 0.6, max_offset_y)
        bl_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
        br_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
    elif direction == 3:
        tl_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
        tr_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
        bl_offset_y = random.uniform(max_offset_y * 0.6, max_offset_y)
        br_offset_y = random.uniform(max_offset_y * 0.6, max_offset_y)
    elif direction == 4:
        tl_offset_x = random.uniform(max_offset_x * 0.5, max_offset_x)
        tl_offset_y = random.uniform(max_offset_y * 0.5, max_offset_y)
        tr_offset_x = random.uniform(-max_offset_x * 0.5, max_offset_x * 0.5)
        tr_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
        bl_offset_x = random.uniform(-max_offset_x * 0.5, max_offset_x * 0.5)
        bl_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
        br_offset_x = random.uniform(-max_offset_x, -max_offset_x * 0.5)
        br_offset_y = random.uniform(-max_offset_y, -max_offset_y * 0.5)
    elif direction == 5:
        tl_offset_x = random.uniform(-max_offset_x * 0.5, max_offset_x * 0.5)
        tl_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
        tr_offset_x = random.uniform(max_offset_x * 0.5, max_offset_x)
        tr_offset_y = random.uniform(max_offset_y * 0.5, max_offset_y)
        bl_offset_x = random.uniform(-max_offset_x, -max_offset_x * 0.5)
        bl_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
        br_offset_x = random.uniform(-max_offset_x * 0.5, max_offset_x * 0.5)
        br_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
    elif direction == 6:
        tl_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
        tl_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
        tr_offset_x = random.uniform(-max_offset_x, -max_offset_x * 0.5)
        tr_offset_y = random.uniform(-max_offset_y, -max_offset_y * 0.5)
        bl_offset_x = random.uniform(max_offset_x * 0.5, max_offset_x)
        bl_offset_y = random.uniform(max_offset_y * 0.5, max_offset_y)
        br_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
        br_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
    else:
        tl_offset_x = random.uniform(-max_offset_x, -max_offset_x * 0.5)
        tl_offset_y = random.uniform(-max_offset_y, -max_offset_y * 0.5)
        tr_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
        tr_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
        bl_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
        bl_offset_y = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
        br_offset_x = random.uniform(max_offset_x * 0.5, max_offset_x)
        br_offset_y = random.uniform(max_offset_y * 0.5, max_offset_y)
    
    v_tilt = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
    
    dst_corners = np.array([
        [tl_offset_x, tl_offset_y + v_tilt],
        [canvas_w - 1 + tr_offset_x, tr_offset_y + v_tilt],
        [canvas_w - 1 + br_offset_x, canvas_h - 1 + br_offset_y - v_tilt],
        [bl_offset_x, canvas_h - 1 + bl_offset_y - v_tilt]
    ], dtype=np.float32)
    
    min_x = min(c[0] for c in dst_corners)
    max_x = max(c[0] for c in dst_corners)
    min_y = min(c[1] for c in dst_corners)
    max_y = max(c[1] for c in dst_corners)
    
    out_w = int(max_x - min_x) + 1
    out_h = int(max_y - min_y) + 1
    
    offset_x = -min_x
    offset_y = -min_y
    dst_offset = dst_corners.copy()
    dst_offset[:, 0] += offset_x
    dst_offset[:, 1] += offset_y
    
    M = cv2.getPerspectiveTransform(src_corners, dst_offset)
    
    warped = cv2.warpPerspective(
        canvas, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Transform photo corners through perspective
    warped_photo_corners = []
    if photo_corners is not None:
        for corners in photo_corners:
            ones = np.ones((len(corners), 1))
            corners_h = np.hstack([corners, ones])
            warped_corners = corners_h @ M.T
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:3]
            warped_photo_corners.append(warped_corners)
    
    # Crop to content bounds with margin
    warped_h, warped_w = warped.shape[:2]
    
    crop_margin_px = crop_margin
    crop_x1 = crop_margin_px
    crop_y1 = crop_margin_px
    crop_x2 = warped_w - crop_margin_px
    crop_y2 = warped_h - crop_margin_px
    
    if crop_x2 > crop_x1 + 200 and crop_y2 > crop_y1 + 200:
        warped = warped[crop_y1:crop_y2, crop_x1:crop_x2]
        
        crop_offset_x = float(crop_margin_px)
        crop_offset_y = float(crop_margin_px)
        
        for corners in warped_photo_corners:
            corners[:, 0] -= crop_offset_x
            corners[:, 1] -= crop_offset_y
        
        dst_offset[:, 0] -= crop_offset_x
        dst_offset[:, 1] -= crop_offset_y
    
    global_corners = dst_offset
    content_bounds = (warped.shape[1], warped.shape[0])
    
    return warped, global_corners, M, content_bounds, warped_photo_corners


# =============================================================================
# LABEL GENERATION (BOTH DETECTION AND POSE FORMATS)
# =============================================================================

def generate_detection_label(photos, out_w, out_h):
    """Generate detection label (5 columns): class x_center y_center width height."""
    lines = []
    for p in photos:
        corners = p['corners']
        
        x_min = min(c[0] for c in corners)
        x_max = max(c[0] for c in corners)
        y_min = min(c[1] for c in corners)
        y_max = max(c[1] for c in corners)
        
        x_center = (x_min + x_max) / 2 / out_w
        y_center = (y_min + y_max) / 2 / out_h
        width = (x_max - x_min) / out_w
        height = (y_max - y_min) / out_h
        
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return '\n'.join(lines) + '\n' if lines else ''


def generate_pose_label(photos, out_w, out_h):
    """Generate pose label (13 columns): class x_center y_center width height kp0x kp0y kpc0 ..."""
    lines = []
    for p in photos:
        corners = p['corners']
        
        x_min = min(c[0] for c in corners)
        x_max = max(c[0] for c in corners)
        y_min = min(c[1] for c in corners)
        y_max = max(c[1] for c in corners)
        
        x_center = (x_min + x_max) / 2 / out_w
        y_center = (y_min + y_max) / 2 / out_h
        width = (x_max - x_min) / out_w
        height = (y_max - y_min) / out_h
        
        # Reorder corners to LL, UL, UR, LR
        ordered_corners = reorder_corners_to_llulur(corners)
        
        line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        for kp in ordered_corners:
            kx = max(0, min(1, kp[0] / out_w))
            ky = max(0, min(1, kp[1] / out_h))
            line += f" {kx:.6f} {ky:.6f} 2"
        line += "\n"
        lines.append(line)
    
    return ''.join(lines)


# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================

def timeout_handler(signum, frame):
    print("\n⏱️ TIMEOUT: Generation took too long, stopping...")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)

print("🖼️  Generating 10 example images (v34 - Fixed canvas size and shadows)")
print("   Canvas: 640x640 + 600px padding | Photos: 1-4 | Perspective: 5-20%")
print("   Shadows: max blur=2, max offset=2")
print("⏱️  Timeout: 300 seconds")
print()

start_time = time.time()

try:
    source_dir = Path("./images")
    sources = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.jpeg')) + \
              list(source_dir.glob('*.png')) + list(source_dir.glob('*.webp'))
    
    print(f"📁 Found {len(sources)} source images")
    
    if len(sources) < 3:
        print("ERROR: Need at least 3 source images")
        sys.exit(1)
    
    output_dir = Path("../data/examples_v34")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    CANVAS_SIZE = CONFIG['CANVAS_SIZE']
    PADDING = CONFIG['CANVAS_PADDING']
    
    random.seed(None)
    np.random.seed(None)
    
    print("\n📸 Generating images...\n")
    
    verification_stats = {
        'total_photos': 0,
        'photos_with_rotation': 0,
        'photos_in_bounds': 0,
        'valid_quadrilaterals': 0,
        'perspective_displacements': [],
    }
    
    for i in range(10):
        img_start = time.time()
        
        # Generate background
        bg = random_base_background(CANVAS_SIZE, CANVAS_SIZE)
        bg = apply_texture_overlay(bg)
        canvas = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
        
        # Add LARGE padding around canvas to prevent black edges after perspective warp
        # Total size: 640 + 2*300 = 1240px
        padded_canvas = np.ones((CANVAS_SIZE + 2 * PADDING, CANVAS_SIZE + 2 * PADDING, 4), dtype=np.uint8) * 180
        padded_canvas[:, :, 3] = 0  # Transparent alpha
        padded_canvas[PADDING:PADDING+CANVAS_SIZE, PADDING:PADDING+CANVAS_SIZE] = canvas
        
        # Use working_canvas that we'll add photos to
        working_canvas = padded_canvas.copy()
        
        # Pack photos (1-4) - placed in padded canvas coordinates
        placements = spiral_pack_photos(CANVAS_SIZE, CANVAS_SIZE)
        placed_photos = []
        
        for placement in placements:
            photo = cv2.imread(str(random.choice(sources)))
            if photo is None:
                continue
            
            target_w, target_h = placement['width'], placement['height']
            h_orig, w_orig = photo.shape[:2]
            scale = min(target_w / w_orig, target_h / h_orig)
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            photo = cv2.resize(photo, (new_w, new_h))
            
            # Apply effects
            photo = fast_photo_manipulation(photo)
            photo = fast_glare(photo)
            photo = add_rgba_alpha(photo)
            photo = blur_alpha_edges(photo)
            photo = rotate_photo(photo, placement['rotation'])
            
            # Photos are in 0-640 coordinates, offset by PADDING for padded canvas space
            center_x = placement['center_x'] + PADDING
            center_y = placement['center_y'] + PADDING
            
            # Shadow
            shadow_params = placement.get('shadow_params', {})
            if shadow_params:
                working_canvas = apply_photo_shadow(
                    working_canvas, photo, center_x, center_y,
                    shadow_params['offset_x'], shadow_params['offset_y'],
                    shadow_params['blur_sigma'], shadow_params['opacity'],
                    rotation=placement['rotation']
                )
            
            working_canvas = composite_photo_at_center(working_canvas, photo, center_x, center_y)
            
            # Store polygon in PADDED coordinates (for perspective transform)
            placed_photos.append({
                'polygon': placement['polygon'].copy() + np.array([PADDING, PADDING]),
                'rotation': placement['rotation'],
            })
            
            verification_stats['total_photos'] += 1
            if abs(placement['rotation']) > 5:
                verification_stats['photos_with_rotation'] += 1
        
        # Apply global perspective warp on working_canvas
        canvas_bgr = cv2.cvtColor(working_canvas, cv2.COLOR_BGR2BGRA)
        photo_corners_list = [p['polygon'] for p in placed_photos]
        
        warped_canvas, global_corners, transform_matrix, content_bounds, warped_photo_corners = apply_global_perspective(
            canvas_bgr, CANVAS_SIZE + 2*PADDING, CANVAS_SIZE + 2*PADDING,
            photo_corners=photo_corners_list,
            crop_margin=CONFIG['CROP_MARGIN']
        )
        
        out_w, out_h = warped_canvas.shape[1], warped_canvas.shape[0]
        
        # Resize to 640x640
        if out_w != CANVAS_SIZE or out_h != CANVAS_SIZE:
            scale_x = CANVAS_SIZE / warped_canvas.shape[1] if warped_canvas.shape[1] > 0 else 1
            scale_y = CANVAS_SIZE / warped_canvas.shape[0] if warped_canvas.shape[0] > 0 else 1
            
            warped_canvas = cv2.resize(warped_canvas, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_LINEAR)
            out_w, out_h = CANVAS_SIZE, CANVAS_SIZE
            
            # Scale corners
            warped_photo_corners = [
                np.array([[kp[0] * scale_x, kp[1] * scale_y] for kp in corners])
                for corners in warped_photo_corners
            ]
        
        final_photos = []
        for idx, photo_data in enumerate(placed_photos):
            warped_corners = warped_photo_corners[idx]
            
            final_photos.append({
                'corners': warped_corners,
                'rotation': photo_data['rotation']
            })
            
            in_bounds = all(
                -10 <= warped_corners[j, 0] < out_w + 10 and
                -10 <= warped_corners[j, 1] < out_h + 10
                for j in range(4)
            )
            if in_bounds:
                verification_stats['photos_in_bounds'] += 1
        
        # Calculate perspective displacement
        src = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32)
        displacements = [np.linalg.norm(global_corners[j] - src[j]) for j in range(4)]
        verification_stats['perspective_displacements'].append(max(displacements))
        
        # Save image
        pil_img = Image.fromarray(cv2.cvtColor(warped_canvas, cv2.COLOR_BGR2RGB))
        img_path = output_dir / f"example_{i+1:02d}.jpg"
        pil_img.save(img_path, quality=90)
        
        # Debug: save annotated image with corners for verification
        debug_img = warped_canvas.copy()
        yolo_names = ['LL', 'UL', 'UR', 'LR']
        for idx, photo_data in enumerate(final_photos):
            corners = photo_data['corners']
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][idx % 4]
            pts = corners.astype(np.int32)
            cv2.polylines(debug_img, [pts], True, color, 2)
            for j, pt in enumerate(corners):
                cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 5, color, -1)
                cv2.putText(debug_img, yolo_names[j], (int(pt[0])+8, int(pt[1])-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        debug_path = output_dir / f"example_{i+1:02d}_debug.jpg"
        cv2.imwrite(str(debug_path), debug_img)
        
        # Generate BOTH label formats
        det_label = generate_detection_label(final_photos, out_w, out_h)
        pose_label = generate_pose_label(final_photos, out_w, out_h)
        
        det_lbl_path = output_dir / f"example_{i+1:02d}_det.txt"
        with open(det_lbl_path, 'w') as f:
            f.write(det_label)
        
        pose_lbl_path = output_dir / f"example_{i+1:02d}_pose.txt"
        with open(pose_lbl_path, 'w') as f:
            f.write(pose_label)
        
        img_time = time.time() - img_start
        print(f"  {i+1:2d}/10: {img_time:4.1f}s ({len(final_photos)} photos)")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("📊 VERIFICATION RESULTS (v34)")
    print(f"{'='*60}")
    
    total = verification_stats['total_photos']
    print(f"\n  Total photos: {total}")
    print(f"  Photos with rotation: {verification_stats['photos_with_rotation']}/{total}")
    print(f"  Photos in bounds: {verification_stats['photos_in_bounds']}/{total}")
    
    if verification_stats['perspective_displacements']:
        avg_disp = sum(verification_stats['perspective_displacements']) / len(verification_stats['perspective_displacements'])
        min_disp = min(verification_stats['perspective_displacements'])
        max_disp = max(verification_stats['perspective_displacements'])
        print(f"  Perspective displacement: {min_disp:.0f}-{max_disp:.0f}px (avg: {avg_disp:.0f}px)")
        print(f"    Target: 32-128px for 5-20% of 640")
    
    print(f"\n✅ Done! 10 images in {total_time:.1f}s")
    print(f"📂 {output_dir.absolute()}")
    
    for f in sorted(output_dir.glob("example_*.jpg")):
        print(f"   - {f.name}")
    
    signal.alarm(0)

except Exception as e:
    signal.alarm(0)
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
    sys.exit(1)
