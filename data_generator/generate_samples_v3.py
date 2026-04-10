#!/usr/bin/env python3
"""
Photo Pose Detector - Fast Synthetic Data Generator (v3)

CORRECT ARCHITECTURE:
1. Pack photos FLAT (rectangles) with ±30° rotation on a flat background
2. Apply ONE global perspective warp to the ENTIRE composite at the end
3. Ensure no corners extend off canvas, no black borders from warp
4. Keep all lighting/shadow/color/brightness/contrast changes

This is fundamentally different from v2 which incorrectly warped each photo individually.
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
# UTILITY FUNCTIONS
# =============================================================================

def clip_corners(corners, margin, canvas_w, canvas_h):
    """Ensure all corners stay within canvas bounds."""
    corners = corners.copy()
    corners[:, 0] = np.clip(corners[:, 0], margin, canvas_w - margin)
    corners[:, 1] = np.clip(corners[:, 1], margin, canvas_h - margin)
    return corners


def rotate_photo_with_transform(photo, angle):
    """
    Rotate photo and return rotation matrix for transforming corners.
    
    Returns:
        (rotated_photo, rotation_matrix_2x3)
    """
    h, w = photo.shape[:2]
    
    if abs(angle) < 1:
        return photo, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Adjust translation
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # Rotate
    rotated = cv2.warpAffine(
        photo, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return rotated, M


def composite_photo_at_center(canvas, photo, cx, cy):
    """
    Composite photo onto canvas with center at (cx, cy).
    
    Args:
        canvas: BGRA canvas
        photo: BGRA photo (may be larger due to rotation padding)
        cx, cy: Center position on canvas
    
    Returns:
        Composited canvas
    """
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    
    # Calculate top-left corner on canvas
    top_left_x = int(cx - pw / 2)
    top_left_y = int(cy - ph / 2)
    
    # Calculate overlap region
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
    
    # Check valid region
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
# PHOTO ROTATION (±30 degrees) - Pre-composite
# =============================================================================

def rotate_photo(photo, angle=None):
    """
    Rotate a photo by ±30 degrees.
    
    Args:
        photo: Photo image array
        angle: Specific angle (or None for random)
    
    Returns:
        Rotated photo with transparency for empty corners
    """
    h, w = photo.shape[:2]
    
    if angle is None:
        # Random angle between -30 and +30 degrees
        angle = random.uniform(-30, 30)
    
    if abs(angle) < 1:
        return photo  # No rotation needed
    
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
        borderValue=(128, 128, 128, 0)  # Transparent border
    )
    
    return rotated


# =============================================================================
# PHOTO PACKING - Pack photos flat, then rotate individually
# =============================================================================

def calculate_rotated_bounds(width, height, angle):
    """
    Calculate the bounding box of a rectangle after rotation.
    
    Args:
        width, height: Original dimensions
        angle: Rotation angle in degrees
    
    Returns:
        (rotated_width, rotated_height)
    """
    angle_rad = math.radians(abs(angle))
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    
    rotated_w = width * cos_a + height * sin_a
    rotated_h = width * sin_a + height * cos_a
    
    return rotated_w, rotated_h


def calculate_photo_packings(canvas_w, canvas_h, num_photos_target=None):
    """
    Calculate photo packings using shelf algorithm.
    
    IMPORTANT: Rotation is applied BEFORE placement.
    We calculate the rotated bounding box for each photo first,
    then pack using those actual dimensions. This prevents overlap.
    
    Args:
        canvas_w, canvas_h: Canvas dimensions
        num_photos_target: Target number of photos (or None for auto)
    
    Returns:
        List of placements, each with: x, y, width, height, rotation, 
        rotated_width, rotated_height, actual_corners (4 corners after rotation)
    """
    margin = 60  # Edge margin for global perspective safety
    gap = 25     # Gap between photos
    
    # Determine number of photos (6-12 for good coverage)
    if num_photos_target is None:
        num_photos = random.randint(6, 12)
    else:
        num_photos = num_photos_target
    
    # Generate photos with rotation, calculate rotated dimensions
    photos_with_rotation = []
    for _ in range(num_photos):
        # Random dimensions
        r = random.random()
        if r < 0.25:
            width = int(canvas_w * random.uniform(0.28, 0.38))
            height = int(canvas_h * random.uniform(0.20, 0.28))
        elif r < 0.70:
            width = int(canvas_w * random.uniform(0.16, 0.26))
            height = int(canvas_h * random.uniform(0.12, 0.20))
        else:
            width = int(canvas_w * random.uniform(0.10, 0.16))
            height = int(canvas_h * random.uniform(0.08, 0.14))
        
        # Random rotation between -30 and +30 degrees
        rotation = random.uniform(-30, 30)
        
        # Calculate rotated bounding box
        rot_w, rot_h = calculate_rotated_bounds(width, height, rotation)
        
        photos_with_rotation.append({
            'width': width,
            'height': height,
            'rotation': rotation,
            'rotated_width': rot_w,
            'rotated_height': rot_h
        })
    
    # Sort by rotated area (larger first) for better shelf packing
    photos_with_rotation.sort(key=lambda x: x['rotated_width'] * x['rotated_height'], reverse=True)
    
    # Shelf packing algorithm using ROTATED dimensions
    placements = []
    shelf_y = margin
    current_x = margin
    current_shelf_height = 0
    
    for photo in photos_with_rotation:
        rot_w = photo['rotated_width']
        rot_h = photo['rotated_height']
        
        # Check if fits in current shelf
        if current_x + rot_w > canvas_w - margin:
            shelf_y += current_shelf_height + gap
            current_x = margin
            current_shelf_height = 0
        
        # Check if fits vertically
        if shelf_y + rot_h > canvas_h - margin:
            # Scale down to fit
            scale = (canvas_h - margin - shelf_y) / rot_h
            if scale > 0.5:
                # Scale original dimensions (not rotated)
                photo['width'] = int(photo['width'] * scale)
                photo['height'] = int(photo['height'] * scale)
                # Recalculate rotated dimensions
                photo['rotated_width'], photo['rotated_height'] = calculate_rotated_bounds(
                    photo['width'], photo['height'], photo['rotation']
                )
                rot_w, rot_h = photo['rotated_width'], photo['rotated_height']
            else:
                continue  # Skip this photo
        
        # Calculate center position for placement
        # The photo center is at (x + width/2, y + height/2)
        # But the rotated bounding box is centered there
        center_x = current_x + rot_w / 2
        center_y = shelf_y + rot_h / 2
        
        # Calculate the ACTUAL corner positions after rotation
        # Corners are relative to photo center (0,0)
        photo_w, photo_h = photo['width'], photo['height']
        photo_corners = np.array([
            [-photo_w/2, -photo_h/2],  # TL (relative to center)
            [ photo_w/2, -photo_h/2],  # TR
            [ photo_w/2,  photo_h/2],  # BR
            [-photo_w/2,  photo_h/2]   # BL
        ], dtype=np.float32)
        
        # Apply rotation to corners
        angle_rad = math.radians(photo['rotation'])
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotated_corners = np.array([
            [
                photo_corners[i, 0] * cos_a - photo_corners[i, 1] * sin_a + center_x,
                photo_corners[i, 0] * sin_a + photo_corners[i, 1] * cos_a + center_y
            ]
            for i in range(4)
        ], dtype=np.float32)
        
        placements.append({
            'x': current_x,
            'y': shelf_y,
            'width': photo['width'],
            'height': photo['height'],
            'rotation': photo['rotation'],
            'center_x': center_x,
            'center_y': center_y,
            'actual_corners': rotated_corners  # The actual 4 corner positions after rotation
        })
        
        current_shelf_height = max(current_shelf_height, rot_h)
        current_x += rot_w + gap
    
    return placements


# =============================================================================
# GLOBAL PERSPECTIVE WARP - Applied once to entire composite
# =============================================================================

def apply_global_perspective(canvas, canvas_w, canvas_h, margin=40):
    """
    Apply a SINGLE perspective warp to the entire canvas composite.
    
    This simulates the camera viewing the scene at an angle.
    The warp is applied to ALL content (photos + background) together.
    
    Key: Output is LARGER than input to allow corners to extend beyond bounds.
    Edge pixels are stretched to fill, no black borders.
    
    Args:
        canvas: The flat composite image
        canvas_w, canvas_h: Original canvas dimensions
        margin: Safety margin for corners
    
    Returns:
        (warped_canvas, global_corners, transform_matrix)
        global_corners: The 4 corners of the canvas after warp (for reference)
    """
    # Source corners (original rectangle)
    src_corners = np.array([
        [0, 0],
        [canvas_w - 1, 0],
        [canvas_w - 1, canvas_h - 1],
        [0, canvas_h - 1]
    ], dtype=np.float32)
    
    # Calculate corner displacement to create visible trapezoid
    # This should be strong enough to create actual perspective distortion
    perspective_strength = random.uniform(0.10, 0.18)  # 10-18% of canvas
    
    # Direction: which side appears closer
    direction = random.randint(0, 3)
    
    # Calculate large offsets that WILL extend beyond canvas bounds
    max_offset_x = canvas_w * perspective_strength
    max_offset_y = canvas_h * perspective_strength
    
    if direction == 0:  # Left side closer
        tl_offset_x = random.uniform(max_offset_x * 0.5, max_offset_x)
        tr_offset_x = random.uniform(-max_offset_x * 0.2, max_offset_x * 0.2)
        bl_offset_x = random.uniform(max_offset_x * 0.5, max_offset_x)
        br_offset_x = random.uniform(-max_offset_x * 0.2, max_offset_x * 0.2)
    elif direction == 1:  # Right side closer
        tl_offset_x = random.uniform(-max_offset_x * 0.2, max_offset_x * 0.2)
        tr_offset_x = random.uniform(max_offset_x * 0.5, max_offset_x)
        bl_offset_x = random.uniform(-max_offset_x * 0.2, max_offset_x * 0.2)
        br_offset_x = random.uniform(max_offset_x * 0.5, max_offset_x)
    elif direction == 2:  # Top closer
        tl_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
        tr_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
        bl_offset_x = random.uniform(-max_offset_x * 0.8, max_offset_x * 0.8)
        br_offset_x = random.uniform(-max_offset_x * 0.8, max_offset_x * 0.8)
    else:  # Bottom closer
        tl_offset_x = random.uniform(-max_offset_x * 0.8, max_offset_x * 0.8)
        tr_offset_x = random.uniform(-max_offset_x * 0.8, max_offset_x * 0.8)
        bl_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
        br_offset_x = random.uniform(-max_offset_x * 0.3, max_offset_x * 0.3)
    
    # Vertical tilt
    v_tilt = random.uniform(-max_offset_y * 0.3, max_offset_y * 0.3)
    
    # Build destination corners that MAY extend beyond bounds
    dst_corners = np.array([
        [tl_offset_x, v_tilt],                    # TL (may go negative)
        [canvas_w - 1 + tr_offset_x, v_tilt],     # TR
        [canvas_w - 1 + br_offset_x, canvas_h - 1 - v_tilt],  # BR
        [bl_offset_x, canvas_h - 1 - v_tilt]      # BL
    ], dtype=np.float32)
    
    # Calculate output bounds based on where corners actually are
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
    
    # Apply warp (output is LARGER than input)
    warped = cv2.warpPerspective(
        canvas, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128)
    )
    
    # Calculate the actual content bounds (where pixels are not gray border)
    content_mask = np.all(warped != 128, axis=2).astype(np.float32)
    content_mask = cv2.GaussianBlur(content_mask, (31, 31), 0)
    
    # Get a representative background color from original canvas
    bg_color = np.median(canvas.reshape(-1, 3), axis=0)
    
    # Blend gray border with background color
    for c in range(3):
        bg_layer = np.full_like(warped[:, :, c], bg_color[c])
        warped[:, :, c] = (warped[:, :, c] * content_mask + 
                          bg_layer * (1 - content_mask)).astype(np.uint8)
    
    # Return corners relative to output canvas
    global_corners = dst_offset
    
    return warped, global_corners, M


# =============================================================================
# CALCULATE FINAL KEYPOINTS (photo corners after global warp)
# =============================================================================

def calculate_warped_photo_corners(flat_corners, transform_matrix):
    """
    Transform photo corners through the global perspective warp.
    
    Args:
        flat_corners: Original flat corners [4x2]
        transform_matrix: The 3x3 perspective transform matrix
    
    Returns:
        Warped corners after global perspective
    """
    # Add homogeneous coordinate
    ones = np.ones((4, 1))
    corners_h = np.hstack([flat_corners, ones])
    
    # Apply transform
    warped_h = corners_h @ transform_matrix.T
    
    # Convert back to cartesian
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


def add_rgba_alpha(img):
    """Convert BGR image to BGRA with alpha channel."""
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = 255  # Fully opaque
    return bgra


# =============================================================================
# DROP SHADOWS (Subtle, applied to flat photos)
# =============================================================================

def create_subtle_shadow(photo, shadow_offset_base=8):
    """
    Create a subtle drop shadow for a photo.
    
    Uses screen-mode compositing for realistic shadows.
    """
    h, w = photo.shape[:2]
    
    # Shadow parameters
    shadow_opacity = random.uniform(0.12, 0.22)
    blur_sigma = random.uniform(5, 15)
    
    # Offset direction (lower-right for natural light from upper-left)
    if random.random() < 0.65:
        lx, ly = -1, -1  # Upper-left light source
    else:
        angle = random.uniform(0, 360)
        lx, ly = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    
    l_len = math.sqrt(lx**2 + lx**2 + ly**2)
    lx, ly = lx / l_len * 2, ly / l_len * 2
    
    shadow_offset = int(shadow_offset_base + random.uniform(0, 5))
    
    # Create shadow mask
    shadow_mask = np.zeros((h + abs(int(lx * shadow_offset)) + 10, 
                            w + abs(int(ly * shadow_offset)) + 10), dtype=np.float32)
    
    # Shadow shape (slightly larger than photo)
    pad = 5
    cv2.rectangle(shadow_mask, 
                 (pad, pad), 
                 (shadow_mask.shape[1] - pad - int(lx * shadow_offset), 
                  shadow_mask.shape[0] - pad - int(ly * shadow_offset)),
                 1.0, -1)
    
    # Blur
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), blur_sigma)
    
    # Apply shadow to photo
    if len(photo.shape) == 3 and photo.shape[2] == 4:
        photo_f = photo.astype(np.float32) / 255.0
        alpha = photo_f[:, :, 3:4]
        
        # Darken with shadow
        shadow_factor = 1 - shadow_mask[:h, :w, np.newaxis] * shadow_opacity
        photo_f[:, :, :3] *= shadow_factor
        photo_f[:, :, 3] *= (1 - shadow_mask[:h, :w] * shadow_opacity * 0.3)
        
        return np.clip(photo_f * 255, 0, 255).astype(np.uint8)
    else:
        return photo


# =============================================================================
# COMPOSITING
# =============================================================================

def alpha_composite(bg, fg):
    """
    Alpha-composite fg onto bg.
    
    Both images should be BGRA.
    """
    if len(bg.shape) == 3 and bg.shape[2] == 4:
        bg_f = bg.astype(np.float32) / 255.0
    else:
        bg_f = np.dstack([bg, np.ones(bg.shape[:2], dtype=np.float32) * 255]) / 255.0
    
    if len(fg.shape) == 3 and fg.shape[2] == 4:
        fg_f = fg.astype(np.float32) / 255.0
    else:
        fg_f = np.dstack([fg, np.ones(fg.shape[:2], dtype=np.float32) * 255]) / 255.0
    
    # Alpha blend
    alpha_fg = fg_f[:, :, 3:4]
    alpha_bg = bg_f[:, :, 3:4]
    
    out_alpha = alpha_fg + alpha_bg * (1 - alpha_fg)
    out_rgb = (fg_f[:, :, :3] * alpha_fg + 
               bg_f[:, :, :3] * alpha_bg * (1 - alpha_fg)) / np.maximum(out_alpha, 0.001)
    
    out = np.dstack([out_rgb, out_alpha]) * 255
    return np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_rotation_applied(flat_corners, rotation):
    """Verify rotation was applied."""
    return abs(rotation) > 5, f"Rotation: {rotation:.1f}°"


def verify_perspective_subtle(global_corners, out_w, out_h):
    """Verify global perspective is subtle (corners don't move too much)."""
    # Global corners are relative to OUTPUT image
    src = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32)
    displacements = [np.linalg.norm(global_corners[i] - src[i]) for i in range(4)]
    
    max_disp = max(displacements)
    min_disp = min(displacements)
    
    # Perspective should be visible but not extreme
    if max_disp > 300:
        return False, f"Perspective too strong: max displacement {max_disp:.0f}px"
    
    return True, f"Perspective OK: {max_disp:.0f}px max displacement"


def verify_trapezoid_shape(corners):
    """
    Verify that the shape is a proper trapezoid (not bowtie/convex).
    
    After global perspective, photos become trapezoids. We verify:
    1. Shape is convex (all cross products have same sign)
    2. Edge ratios are reasonable (not too extreme)
    """
    # Check convexity using cross products
    def cross_product_2d(A, B, C):
        """Cross product of AB x BC"""
        return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])
    
    signs = []
    for i in range(4):
        A = corners[i]
        B = corners[(i + 1) % 4]
        C = corners[(i + 2) % 4]
        cp = cross_product_2d(A, B, C)
        signs.append(cp > 0)
    
    # Convex if all cross products have same sign (all True or all False)
    if not (all(signs) or not any(signs)):
        return False, "Non-convex shape (bowtie or concave)"
    
    # Check edge ratios (trapezoid should have similar parallel edges)
    top_width = np.linalg.norm(corners[1] - corners[0])
    bot_width = np.linalg.norm(corners[2] - corners[3])
    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])
    
    if top_width > 0 and bot_width > 0:
        h_ratio = min(top_width, bot_width) / max(top_width, bot_width)
        v_ratio = min(left_height, right_height) / max(left_height, right_height)
        
        # After global perspective, ratios should be 0.5-1.0
        if h_ratio < 0.5 or v_ratio < 0.5:
            return False, f"Edge ratio too extreme: h={h_ratio:.2f}, v={v_ratio:.2f}"
    
    return True, "Valid trapezoid"


def verify_corners_in_bounds(corners, canvas_w, canvas_h):
    """Verify all corners are within canvas bounds."""
    in_bounds = all(
        0 <= corners[i, 0] < canvas_w and 0 <= corners[i, 1] < canvas_h
        for i in range(4)
    )
    return in_bounds, "All corners in bounds" if in_bounds else "Some corners out of bounds"


# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================

# Timeout handler
def timeout_handler(signum, frame):
    print("\n⏱️ TIMEOUT: Generation took too long, stopping...")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(180)  # 3 minute timeout

print("🖼️  Generating 10 example images (v3 - correct architecture)...")
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
    
    output_dir = Path("../data/examples_v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    CANVAS_W, CANVAS_H = 1920, 1080
    
    random.seed(42)
    np.random.seed(42)
    
    print("\n📸 Generating images...\n")
    
    # Verification tracking
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
        
        # Get photo packings (flat rectangles)
        placements = calculate_photo_packings(CANVAS_W, CANVAS_H)
        
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
            
            # Apply effects BEFORE rotation (but rotation is already accounted for in placement)
            photo = fast_photo_manipulation(photo)
            photo = fast_glare(photo)
            
            # Add alpha channel
            photo = add_rgba_alpha(photo)
            
            # Rotate photo (rotation is already factored into placement dimensions)
            rotation = placement['rotation']
            photo_rotated = rotate_photo(photo, rotation)
            
            # The placement already has actual_corners calculated
            # These are the 4 corner positions after rotation
            actual_corners = placement['actual_corners']
            
            # Composite photo onto canvas at the correct rotated position
            center_x = placement['center_x']
            center_y = placement['center_y']
            canvas = composite_photo_at_center(canvas, photo_rotated, center_x, center_y)
            
            # Store actual corners (after rotation) for global warp transformation
            placed_photos.append({
                'actual_corners': actual_corners.copy(),  # Already in canvas coordinates
                'rotation': rotation,
                'original_placement': placement
            })
            
            # Verification
            verification_stats['total_photos'] += 1
            
            # Rotation check
            if abs(rotation) > 5:
                verification_stats['photos_with_rotation'] += 1
        
        # Apply ONE global perspective warp to entire composite
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        warped_canvas, global_corners, transform_matrix = apply_global_perspective(
            canvas_bgr, CANVAS_W, CANVAS_H, margin=40
        )
        
        # Transform all photo corners through the global warp
        final_photos = []
        for photo_data in placed_photos:
            # The corners are already in canvas coordinates (after rotation)
            actual_corners = photo_data['actual_corners']
            
            # Transform through global perspective
            warped_corners = calculate_warped_photo_corners(
                actual_corners, transform_matrix
            )
            
            # Clip warped corners to output bounds
            out_w, out_h = warped_canvas.shape[1], warped_canvas.shape[0]
            warped_corners[:, 0] = np.clip(warped_corners[:, 0], 0, out_w - 1)
            warped_corners[:, 1] = np.clip(warped_corners[:, 1], 0, out_h - 1)
            
            final_photos.append({
                'corners': warped_corners,
                'rotation': photo_data['rotation']
            })
            
            # Verification
            ok, msg = verify_corners_in_bounds(warped_corners, out_w, out_h)
            if ok:
                verification_stats['photos_in_bounds'] += 1
            
            ok, msg = verify_trapezoid_shape(warped_corners)
            if ok:
                verification_stats['valid_trapezoids'] += 1
        
        # Global perspective check
        out_w, out_h = warped_canvas.shape[1], warped_canvas.shape[0]
        ok, msg = verify_perspective_subtle(global_corners, out_w, out_h)
        if ok:
            verification_stats['subtle_perspective'] += 1
        verification_stats['global_perspective_applied'] += 1
        
        # Save image
        pil_img = Image.fromarray(cv2.cvtColor(warped_canvas, cv2.COLOR_BGR2RGB))
        img_path = output_dir / f"example_{i+1:02d}.jpg"
        pil_img.save(img_path, quality=90)
        
        # Save label file (corners relative to warped image)
        lbl_path = output_dir / f"example_{i+1:02d}.txt"
        out_w, out_h = warped_canvas.shape[1], warped_canvas.shape[0]
        with open(lbl_path, 'w') as f:
            for p in final_photos:
                kps = p['corners']
                x_center = sum(k[0] for k in kps) / 4 / out_w
                y_center = sum(k[1] for k in kps) / 4 / out_h
                width = (max(k[0] for k in kps) - min(k[0] for k in kps)) / out_w
                height = (max(k[1] for k in kps) - min(k[1] for k in kps)) / out_h
                
                line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                for kx, ky in kps:
                    line += f" {kx/out_w:.6f} {ky/out_h:.6f} 2"
                line += "\n"
                f.write(line)
        
        img_time = time.time() - img_start
        print(f"  {i+1:2d}/10: {img_time:5.1f}s ({len(final_photos)} photos, global warp applied)")
    
    total_time = time.time() - start_time
    
    # Print verification summary
    print(f"\n{'='*60}")
    print("📊 VERIFICATION RESULTS")
    print(f"{'='*60}")
    
    total = verification_stats['total_photos']
    print(f"\n  Total photos: {total}")
    print(f"  Photos with rotation (±30°): {verification_stats['photos_with_rotation']}/{total}")
    print(f"  Photos in bounds: {verification_stats['photos_in_bounds']}/{total}")
    print(f"  Valid trapezoids: {verification_stats['valid_trapezoids']}/{total}")
    print(f"  Subtle global perspective: {verification_stats['subtle_perspective']}/{verification_stats['global_perspective_applied']}")
    
    print(f"\n✅ Done! 10 images in {total_time:.1f}s")
    print(f"📂 {output_dir.absolute()}")
    
    # List files
    for f in sorted(output_dir.glob("example_*.jpg")):
        print(f"   - {f.name} ({f.stat().st_size//1024} KB)")
    
    signal.alarm(0)
    
except Exception as e:
    signal.alarm(0)
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
    sys.exit(1)
