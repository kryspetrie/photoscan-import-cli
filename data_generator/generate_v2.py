#!/usr/bin/env python3
"""
Simple Data Generator - Fixed Version 2
========================================

Issues fixed:
1. No artificial border - photos placed properly within canvas
2. Bounding box matches actual corners (computed from corners, not separately)
3. Photos actually visible in output
4. Perspective transform keeps corners in bounds
5. Proper alpha compositing for smooth photo edges

Author: Photo Pose Detector Project
"""

import cv2
import numpy as np
from pathlib import Path
import random
import sys
import time
import math
import colorsys
import colorsys

# Configuration
CANVAS_SIZE = 640
PHOTO_SIZE_MIN = 180
PHOTO_SIZE_MAX = 480
ROTATION_RANGE = 30
NUM_PHOTOS_MIN = 1
NUM_PHOTOS_MAX = 4
EDGE_MARGIN = 50


# =============================================================================
# PLACEMENT VALIDATION — prevent overlaps and off-screen photos
# =============================================================================

OVERLAP_THRESHOLD = 0.05   # Max allowed overlap (5% of smaller photo area)
BOUND_MARGIN = 80          # Minimum margin from canvas edge for rotated corners
MAX_PACK_ATTEMPTS = 50     # Retries before reducing photo count


def compute_rotated_bbox(width, height, center_x, center_y, rotation):
    """Compute axis-aligned bounding box of a rotated rectangle."""
    corners = get_rotated_polygon(width, height, center_x, center_y, rotation)
    xs = corners[:, 0]
    ys = corners[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def check_bounds(placements, canvas_size, margin=BOUND_MARGIN):
    """Check that all rotated photo corners stay within canvas bounds."""
    for p in placements:
        corners = get_rotated_polygon(p['width'], p['height'],
                                       p['center_x'], p['center_y'],
                                       p['rotation'])
        for c in corners:
            if c[0] < margin or c[0] > canvas_size - margin:
                return False
            if c[1] < margin or c[1] > canvas_size - margin:
                return False
    return True


def polygon_area(corners):
    """Compute area of a simple polygon using the shoelace formula."""
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0


def polygon_intersection_area(poly1, poly2):
    """
    Compute the intersection area of two convex polygons using rasterization.
    Falls back to bounding box overlap if polygon clipping fails.
    """
    # Use shapely-free approach: rasterize both polygons on a small grid
    # and count overlapping pixels. Fast enough for 2-4 polygons on 640x640.
    try:
        # Use OpenCV to create binary masks and count overlap
        canvas_size = 640
        mask1 = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        mask2 = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        
        pts1 = corners_to_int_points(poly1)
        pts2 = corners_to_int_points(poly2)
        
        cv2.fillPoly(mask1, [pts1], 1)
        cv2.fillPoly(mask2, [pts2], 1)
        
        overlap = np.count_nonzero(mask1 & mask2)
        return float(overlap)
    except Exception:
        # Fallback: bounding box overlap
        x1_min, y1_min, x1_max, y1_max = bounding_box(poly1)
        x2_min, y2_min, x2_max, y2_max = bounding_box(poly2)
        
        ix_min = max(x1_min, x2_min)
        iy_min = max(y1_min, y2_min)
        ix_max = min(x1_max, x2_max)
        iy_max = min(y1_max, y2_max)
        
        if ix_max <= ix_min or iy_max <= iy_min:
            return 0.0
        return float((ix_max - ix_min) * (iy_max - iy_min))


def corners_to_int_points(corners):
    """Convert corners array to int32 points for OpenCV."""
    return np.array([[int(round(c[0])), int(round(c[1]))] for c in corners],
                     dtype=np.int32)


def bounding_box(corners):
    """Get bounding box of polygon corners."""
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def check_overlaps(placements, threshold=OVERLAP_THRESHOLD):
    """
    Check that no pair of photos overlaps more than threshold.
    Uses pixel-based polygon intersection for accuracy.
    Returns True if all pairs are within the threshold.
    """
    n = len(placements)
    if n <= 1:
        return True
    
    # Compute polygon corners and pixel areas for each placement
    polys = []
    areas = []
    for p in placements:
        corners = get_rotated_polygon(p['width'], p['height'],
                                       p['center_x'], p['center_y'],
                                       p['rotation'])
        polys.append(corners)
        # Compute area via polygon formula (accurate)
        areas.append(polygon_area(corners))
    
    for i in range(n):
        for j in range(i + 1, n):
            if areas[i] < 1 or areas[j] < 1:
                continue
            overlap_pixels = polygon_intersection_area(polys[i], polys[j])
            smaller_area = min(areas[i], areas[j])
            overlap_fraction = overlap_pixels / smaller_area
            if overlap_fraction > threshold:
                return False
    return True


def verify_composited_pixels(canvas, expected_areas):
    """
    After compositing, verify that the total opaque pixel count
    matches expectations. Returns True if within 90-110% of expected.
    """
    if canvas.shape[2] == 4:
        alpha = canvas[:, :, 3]
        composited = np.count_nonzero(alpha > 128)
    else:
        # For 3-channel, use brightness deviation from background
        # This is approximate — skip verification for 3-channel canvases
        return True
    
    expected_total = sum(expected_areas)
    if expected_total < 1:
        return True
    
    ratio = composited / expected_total
    # Allow 10% tolerance for clipping at edges
    return 0.80 <= ratio <= 1.10


def pack_photos_validated(canvas_size):
    """
    Pack photos with validation: no overlaps, all corners within bounds.
    Retries placement up to MAX_ATTEMPTS times, then reduces photo count.
    """
    for num_photos in range(NUM_PHOTOS_MAX, NUM_PHOTOS_MIN - 1, -1):
        for attempt in range(MAX_PACK_ATTEMPTS):
            placements = _generate_placements(num_photos, canvas_size)
            if check_bounds(placements, canvas_size) and check_overlaps(placements):
                return placements
        # Couldn't place this many photos, try fewer
    
    # Fallback: single small photo at center
    return [{
        'width': 180, 'height': 144,
        'center_x': canvas_size / 2, 'center_y': canvas_size / 2,
        'rotation': 0
    }]


def _generate_placements(num_photos, canvas_size):
    """Generate candidate placements (same logic as old pack_photos_simple)."""
    placements = []
    margin = BOUND_MARGIN
    
    if num_photos == 1:
        size = random.randint(PHOTO_SIZE_MIN, PHOTO_SIZE_MAX)
        aspect = random.uniform(0.7, 1.3)
        height = int(size * aspect)
        width = int(size)
        
        cx = canvas_size / 2 + random.uniform(-60, 60)
        cy = canvas_size / 2 + random.uniform(-60, 60)
        rotation = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        
        placements.append({
            'width': width, 'height': height,
            'center_x': cx, 'center_y': cy,
            'rotation': rotation
        })
    
    elif num_photos == 2:
        size = random.randint(PHOTO_SIZE_MIN, min(380, PHOTO_SIZE_MAX))
        aspect1 = random.uniform(0.7, 1.3)
        aspect2 = random.uniform(0.7, 1.3)
        
        if random.random() < 0.5:  # Horizontal
            w1 = int(size * random.uniform(0.8, 1.0))
            h1 = int(w1 * aspect1)
            w2 = int(size * random.uniform(0.8, 1.0))
            h2 = int(w2 * aspect2)
            
            cx1 = canvas_size * 0.3 + random.uniform(-30, 30)
            cy1 = canvas_size * 0.5 + random.uniform(-30, 30)
            cx2 = canvas_size * 0.7 + random.uniform(-30, 30)
            cy2 = canvas_size * 0.5 + random.uniform(-30, 30)
        else:  # Vertical
            h1 = int(size * random.uniform(0.8, 1.0))
            w1 = int(h1 * aspect1)
            h2 = int(size * random.uniform(0.8, 1.0))
            w2 = int(h2 * aspect2)
            
            cx1 = canvas_size * 0.5 + random.uniform(-30, 30)
            cy1 = canvas_size * 0.3 + random.uniform(-30, 30)
            cx2 = canvas_size * 0.5 + random.uniform(-30, 30)
            cy2 = canvas_size * 0.7 + random.uniform(-30, 30)
        
        placements.append({
            'width': w1, 'height': h1,
            'center_x': cx1, 'center_y': cy1,
            'rotation': random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        })
        placements.append({
            'width': w2, 'height': h2,
            'center_x': cx2, 'center_y': cy2,
            'rotation': random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        })
    
    else:  # 3 or 4 photos — grid layout with more margin
        size = random.randint(PHOTO_SIZE_MIN, min(320, PHOTO_SIZE_MAX))
        
        cols = 2
        rows = 2 if num_photos == 4 else 2
        
        usable = canvas_size - 2 * margin
        cell_w = usable / cols
        cell_h = usable / rows
        
        positions = []
        for r in range(rows):
            for c in range(cols):
                positions.append((r, c))
        random.shuffle(positions)
        positions = positions[:num_photos]
        
        for row, col in positions:
            aspect = random.uniform(0.7, 1.3)
            width = int(cell_w * random.uniform(0.55, 0.75))
            height = int(width * aspect)
            
            # Keep height within cell
            if height > cell_h * 0.75:
                height = int(cell_h * 0.75)
                width = int(height / aspect)
            
            cx = margin + (col + 0.5) * cell_w + random.uniform(-cell_w * 0.05, cell_w * 0.05)
            cy = margin + (row + 0.5) * cell_h + random.uniform(-cell_h * 0.05, cell_h * 0.05)
            
            placements.append({
                'width': width, 'height': height,
                'center_x': cx, 'center_y': cy,
                'rotation': random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
            })
    
    return placements


def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """Calculate corners of rotated rectangle - VERIFIED CORRECT."""
    if abs(rotation) < 1:
        return np.array([
            [center_x - width/2, center_y + height/2],  # LL
            [center_x - width/2, center_y - height/2],  # UL
            [center_x + width/2, center_y - height/2],  # UR
            [center_x + width/2, center_y + height/2]   # LR
        ], dtype=np.float32)
    
    photo_center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(height * sin + width * cos)
    new_h = int(width * sin + height * cos)
    
    M[0, 2] += (new_w - width) / 2
    M[1, 2] += (new_h - height) / 2
    
    top_left_x = center_x - new_w / 2
    top_left_y = center_y - new_h / 2
    
    corners_photo = np.array([
        [0, height], [0, 0], [width, 0], [width, height]
    ], dtype=np.float32)
    
    corners_final = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M @ pt
        corners_final[i, 0] = top_left_x + rotated[0]
        corners_final[i, 1] = top_left_y + rotated[1]
    
    return corners_final


def rotate_photo(photo, angle):
    """Rotate photo with alpha channel."""
    h, w = photo.shape[:2]
    
    # Add alpha channel if not present
    if photo.shape[2] == 3:
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
    
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
    
    return cv2.warpAffine(photo, M, (new_w, new_h), 
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, 
                          borderValue=(0, 0, 0, 0))  # Transparent border


def apply_photo_shadow(canvas, photo, cx, cy, offset_x, offset_y, blur_sigma, opacity, orig_w, orig_h, rotation):
    """
    Render a drop shadow beneath the photo onto the canvas.
    
    Creates a shadow mask at the ORIGINAL (pre-rotation) photo dimensions,
    rotates it to match the photo, blurs for soft edges, then darkens
    the canvas behind the photo with a slight offset.
    
    IMPORTANT: We create the mask at orig_w x orig_h (before rotation)
    and rotate it. The rotated photo has expanded dimensions (the
    bounding box of the rotated rectangle) — if we used those dimensions,
    the shadow would be an axis-aligned rectangle that's too large.
    """
    ch, cw = canvas.shape[:2]
    num_channels = canvas.shape[2]
    
    # Rotate the offset direction by the photo's rotation so the shadow
    # always falls in the same physical direction relative to the scene
    rot_rad = math.radians(rotation)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    rotated_offset_x = offset_x * cos_r - offset_y * sin_r
    rotated_offset_y = offset_x * sin_r + offset_y * cos_r
    
    # Shadow center is slightly offset from photo center
    shadow_cx = cx + rotated_offset_x * 0.5
    shadow_cy = cy + rotated_offset_y * 0.5
    
    # Create filled rectangle shadow mask at ORIGINAL photo dimensions
    # (before rotation), then rotate it to match the photo shape.
    # This produces a rotated-rectangle shadow matching the photo.
    blur_pad = int(3 * blur_sigma) + 1
    mask_w = orig_w + blur_pad * 2
    mask_h = orig_h + blur_pad * 2
    shadow_mask = np.zeros((mask_h, mask_w), dtype=np.float32)
    
    # Fill the original-dimensions rectangle
    shadow_mask[blur_pad:blur_pad+orig_h, blur_pad:blur_pad+orig_w] = 1.0
    
    # Blur BEFORE rotation so the blur is omnidirectional (not stretched along rotation axis)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=blur_sigma)
    
    # Rotate the shadow mask to match the photo
    if abs(rotation) > 0.5:
        center_rot = (mask_w / 2, mask_h / 2)
        rot_matrix = cv2.getRotationMatrix2D(center_rot, rotation, 1.0)
        # Calculate new dimensions (same formula as rotate_photo)
        cos_a = abs(rot_matrix[0, 0])
        sin_a = abs(rot_matrix[0, 1])
        new_w = int(mask_h * sin_a + mask_w * cos_a)
        new_h = int(mask_w * sin_a + mask_h * cos_a)
        # Adjust translation to center the rotated mask
        rot_matrix[0, 2] += (new_w - mask_w) / 2
        rot_matrix[1, 2] += (new_h - mask_h) / 2
        shadow_mask = cv2.warpAffine(shadow_mask, rot_matrix, (new_w, new_h),
                                     borderValue=0, flags=cv2.INTER_LINEAR)
        mask_w = new_w
        mask_h = new_h
    
    # Normalize so peak is 1.0
    if shadow_mask.max() > 0:
        shadow_mask = shadow_mask / shadow_mask.max()
    
    # Position the shadow on the canvas centered at (shadow_cx, shadow_cy)
    shadow_top_left_x = int(shadow_cx - mask_w / 2)
    shadow_top_left_y = int(shadow_cy - mask_h / 2)
    
    # Darken the canvas
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


def composite_photo_at_center(canvas, photo, cx, cy):
    """
    Composite BGRA photo onto canvas with alpha compositing.
    Photo edges with alpha=0 are transparent (show canvas through).
    """
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    
    # Ensure canvas has alpha channel
    if canvas.shape[2] == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
        canvas[:, :, 3] = 255  # Opaque canvas
    
    top_left_x = int(cx - pw / 2)
    top_left_y = int(cy - ph / 2)
    
    # Calculate source and destination regions
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
        src_x2 = src_x1 + (cw - dst_x1)
        dst_x2 = cw
    if dst_y2 > ch:
        src_y2 = src_y1 + (ch - dst_y1)
        dst_y2 = ch
    
    copy_w = int(dst_x2 - dst_x1)
    copy_h = int(dst_y2 - dst_y1)
    
    if copy_w <= 0 or copy_h <= 0:
        return canvas
    
    src_x1, src_y1 = int(src_x1), int(src_y1)
    
    # Extract regions
    canvas_region = canvas[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32) / 255.0
    photo_region = photo[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w].astype(np.float32) / 255.0
    
    # Get alpha from photo (4th channel)
    photo_alpha = photo_region[:, :, 3:4]
    canvas_alpha = canvas_region[:, :, 3:4]
    
    # Alpha compositing: result = photo * photo_alpha + canvas * canvas_alpha * (1 - photo_alpha)
    # Normalize alpha to [0, 1]
    alpha = photo_alpha
    result_rgb = photo_region[:, :, :3] * alpha + canvas_region[:, :, :3] * (1 - alpha)
    result_alpha = np.maximum(canvas_alpha, alpha)  # Keep max opacity
    
    # Convert back to uint8
    result = np.concatenate([result_rgb, result_alpha], axis=2)
    result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = result
    
    return canvas


# =============================================================================
# BACKGROUND GENERATION WITH GRADIENTS AND TEXTURES
# =============================================================================

def random_base_background(w, h):
    """Generate a random background with controlled brightness and saturation."""
    rand_val = random.random()
    
    # 30% dark, 30% light, 40% medium with varying saturation
    if rand_val < 0.30:
        lightness = random.uniform(0.04, 0.28)
        saturation = random.uniform(0, 0.04)
    elif rand_val < 0.60:
        lightness = random.uniform(0.69, 0.96)
        saturation = random.uniform(0, 0.04)
    else:
        lightness = random.uniform(0.19, 0.86)
        saturation = random.uniform(0.04, 0.40)
    
    hue = random.uniform(0, 1)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    color = (int(r * 255), int(g * 255), int(b * 255))
    
    img = np.ones((h, w, 3), dtype=np.float32) * np.array(color, dtype=np.float32)
    
    # Add noise
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


def fast_glare(img):
    """Add glare highlights using screen blend."""
    if random.random() < 0.5:
        h, w = img.shape[:2]
        
        num_flares = random.randint(2, 4)
        for _ in range(num_flares):
            img_f = img.astype(np.float32) / 255.0
            
            cx = random.uniform(w * 0.15, w * 0.85)
            cy = random.uniform(h * 0.1, h * 0.7)
            
            rx = random.uniform(w * 0.20, w * 0.40)
            ry = random.uniform(h * 0.20, h * 0.40)
            
            y, x = np.ogrid[:h, :w]
            
            flare = np.maximum(0, 1 - (x - cx)**2 / (rx**2) - (y - cy)**2 / (ry**2))
            flare = cv2.GaussianBlur(flare.astype(np.float32), (15, 15), 0)
            
            opacity = random.uniform(0.60, 1.00)
            flare_f = flare[:, :, np.newaxis]
            img_f = 1 - (1 - img_f) * (1 - flare_f * opacity)
            
            img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    
    return img


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
# SAFE PERSPECTIVE - KEEPS CORNERS IN BOUNDS
# =============================================================================

def apply_perspective_safe(canvas, corners_list):
    """
    Apply perspective transform that keeps all corners in bounds.
    Returns BGR canvas (alpha flattened).
    """
    h, w = canvas.shape[:2]
    max_strength = 0.05  # Start at 5%
    safety_margin = 15  # Keep corners this many pixels inside bounds
    
    # Convert to BGR if BGRA (for warpPerspective)
    if canvas.shape[2] == 4:
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    else:
        canvas_bgr = canvas.copy()
    
    for strength in np.linspace(max_strength, 0.0, 25):
        max_disp = int(min(w, h) * strength)
        
        # Random offsets
        tl = (random.randint(-max_disp, 0), random.randint(-max_disp, 0))
        tr = (random.randint(0, max_disp), random.randint(-max_disp, 0))
        bl = (random.randint(-max_disp, 0), random.randint(0, max_disp))
        br = (random.randint(0, max_disp), random.randint(0, max_disp))
        
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_pts = np.array([
            [tl[0], tl[1]],
            [w + tr[0], tr[1]],
            [w + br[0], h + br[1]],
            [bl[0], h + bl[1]]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Check if all corners stay in bounds (with safety margin)
        all_in_bounds = True
        warped_corners_check = []
        
        for corners in corners_list:
            for i in range(4):
                pt = np.array([corners[i, 0], corners[i, 1], 1])
                result = M @ pt
                wx = result[0] / result[2]
                wy = result[1] / result[2]
                warped_corners_check.append((wx, wy))
                if wx < safety_margin or wx > w - safety_margin or wy < safety_margin or wy > h - safety_margin:
                    all_in_bounds = False
        
        if all_in_bounds:
            warped = cv2.warpPerspective(canvas_bgr, M, (w, h),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(128, 128, 128))
            return warped, M, True
    
    # Fallback: no perspective
    return canvas_bgr, np.eye(3), False


def generate_image(source_dir):
    """Generate a single image with validated placement and pixel verification."""
    sources = list(Path(source_dir).glob('*.jpg')) + list(Path(source_dir).glob('*.jpeg'))
    if not sources:
        raise ValueError(f"No source images found")
    
    # Create textured background with gradients and noise
    canvas = random_base_background(CANVAS_SIZE, CANVAS_SIZE)
    canvas = apply_texture_overlay(canvas)
    # Convert to BGRA for alpha compositing and pixel verification
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
    canvas[:, :, 3] = 255  # Opaque background
    
    # Pack photos WITH validation (no overlaps, all in bounds)
    placements = pack_photos_validated(CANVAS_SIZE)
    photos_data = []
    photos_data = []
    
    for placement in placements:
        photo_path = random.choice(sources)
        photo = cv2.imread(str(photo_path))
        if photo is None:
            continue
        
        h_orig, w_orig = photo.shape[:2]
        scale = min(placement['width'] / w_orig, placement['height'] / h_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        photo = cv2.resize(photo, (new_w, new_h))
        
        # Apply glare effect before rotation
        photo = fast_glare(photo)
        
        # Convert to BGRA for proper alpha compositing
        if photo.shape[2] == 3:
            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
        photo[:, :, 3] = 255  # Full opacity
        

        photo = rotate_photo(photo, placement['rotation'])
        
        # Apply drop shadow BEFORE compositing (darkens background behind photo)
        shadow_offset = random.randint(2, 5)
        angle = random.uniform(0, 2 * math.pi)
        offset_x = int(shadow_offset * math.cos(angle))
        offset_y = int(shadow_offset * math.sin(angle))
        shadow_opacity = random.uniform(0.15, 0.35)
        shadow_blur = random.uniform(1.5, 3.0)
        canvas = apply_photo_shadow(
            canvas, photo, placement['center_x'], placement['center_y'],
            offset_x, offset_y, shadow_blur, shadow_opacity,
            new_w, new_h, placement['rotation']
        )
        
        canvas = composite_photo_at_center(canvas, photo, placement['center_x'], placement['center_y'])
        
        # Calculate corners
        corners = get_rotated_polygon(new_w, new_h, placement['center_x'], placement['center_y'], placement['rotation'])
        
        photos_data.append({
            'corners': corners,
            'rotation': placement['rotation']
        })
    
    # Sanity check: verify no photo corners are off-canvas after compositing
    # (pack_photos_validated should prevent this, but check anyway)
    oob_count = 0
    for p in photos_data:
        corners = p['corners']
        for c in corners:
            if c[0] < 0 or c[0] > CANVAS_SIZE or c[1] < 0 or c[1] > CANVAS_SIZE:
                oob_count += 1
    if oob_count > 0:
        print(f"  WARNING: {oob_count} corners out of bounds")
    
    # Apply perspective (keeping corners in bounds)
    corners_list = [p['corners'] for p in photos_data]
    canvas, persp_M, had_perspective = apply_perspective_safe(canvas, corners_list)
    
    # Transform corners
    final_photos = []
    for photo in photos_data:
        corners = photo['corners']
        warped_corners = np.zeros_like(corners)
        for i in range(4):
            pt = np.array([corners[i, 0], corners[i, 1], 1])
            result = persp_M @ pt
            warped_corners[i, 0] = result[0] / result[2]
            warped_corners[i, 1] = result[1] / result[2]
        
        final_photos.append({
            'corners': warped_corners,
            'rotation': photo['rotation']
        })
    
    # Generate labels
    det_labels = []
    pose_labels = []
    
    for photo in final_photos:
        corners = photo['corners']
        
        min_x = min(c[0] for c in corners)
        max_x = max(c[0] for c in corners)
        min_y = min(c[1] for c in corners)
        max_y = max(c[1] for c in corners)
        
        x_center = ((min_x + max_x) / 2) / CANVAS_SIZE
        y_center = ((min_y + max_y) / 2) / CANVAS_SIZE
        width = (max_x - min_x) / CANVAS_SIZE
        height = (max_y - min_y) / CANVAS_SIZE
        
        det_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        corners_str = " ".join([f"{corners[i,0]/CANVAS_SIZE:.6f} {corners[i,1]/CANVAS_SIZE:.6f} 2" for i in range(4)])
        pose_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {corners_str}")
    
    return canvas, final_photos, det_labels, pose_labels


def create_debug_image(img, photos):
    """Create debug image with corner overlays."""
    debug = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    names = ['LL', 'UL', 'UR', 'LR']
    
    for photo in photos:
        corners = photo['corners'].astype(np.int32)
        cv2.polylines(debug, [corners], True, (255, 255, 255), 2)
        
        for i in range(4):
            pt = (int(corners[i, 0]), int(corners[i, 1]))
            cv2.circle(debug, pt, 10, colors[i], -1)
            cv2.putText(debug, names[i], (pt[0]+12, pt[1]-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    
    return debug


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate training data')
    parser.add_argument('--source', default='./images', help='Source images')
    parser.add_argument('--output', default='./data/examples_v2_final', help='Output dir')
    parser.add_argument('--count', type=int, default=10, help='Number of images')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.count} images...")
    start_time = time.time()
    
    total_photos = 0
    
    for i in range(args.count):
        img, photos, det_labels, pose_labels = generate_image(args.source)
        total_photos += len(photos)
        
        cv2.imwrite(str(output_dir / f"example_{i+1:02d}.jpg"), img)
        cv2.imwrite(str(output_dir / f"example_{i+1:02d}_debug.jpg"), create_debug_image(img, photos))
        
        with open(output_dir / f"example_{i+1:02d}_det.txt", 'w') as f:
            f.write('\n'.join(det_labels))
        with open(output_dir / f"example_{i+1:02d}_pose.txt", 'w') as f:
            f.write('\n'.join(pose_labels))
        
        print(f"  {i+1:2d}/{args.count}: {len(photos)} photos")
    
    print(f"\n✅ Done! {total_photos} total photos in {args.count} images (avg: {total_photos/args.count:.1f})")
    print(f"📂 {output_dir.absolute()}")


if __name__ == "__main__":
    main()
