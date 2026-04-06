#!/usr/bin/env python3
"""
Photo Pose Detector - Synthetic Training Data Generator (v12)

Uses OpenCV for proper image transformations with alpha channel support.
Keypoints are placed at actual warped corners, not bounding box.
"""

import random
import argparse
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import math

DEFAULT_CONFIG = {
    "num_train_images": 800,
    "num_val_images": 200,
    "image_width": 1920,
    "image_height": 1080,
    "num_photos_min": 4,
    "num_photos_max": 9,
    "min_photo_size": 200,
    "max_photo_size": 550,
    "source_images": "./images",
    "output_dir": "../data",
    "seed": 42,
}


class Config:
    def __init__(self, **kwargs):
        for k, v in DEFAULT_CONFIG.items():
            setattr(self, k, kwargs.get(k, v))


def cv2_perspective_warp(img_rgba, strength=0.1):
    """Apply perspective warp with alpha based on warped quadrilateral.
    
    Both RGB and alpha are warped through the perspective transform.
    Output canvas is sized to contain all warped corners.
    Returns: (warped_image, corners) where corners are the 4 corner positions
    """
    h, w = img_rgba.shape[:2]
    
    # Source corners (image rectangle)
    src = np.array([
        [0.0, 0.0],
        [w - 1.0, 0.0],
        [w - 1.0, h - 1.0],
        [0.0, h - 1.0]
    ], dtype=np.float32)
    
    # Random corner displacements
    tl = random.uniform(-w * strength, w * strength)
    tr = random.uniform(-w * strength, w * strength)
    bl = random.uniform(-w * strength, w * strength)
    br = random.uniform(-w * strength, w * strength)
    ty_top = random.uniform(-h * strength * 0.5, h * strength * 0.5)
    ty_bottom = random.uniform(-h * strength * 0.5, h * strength * 0.5)
    
    # Destination corners (warped quadrilateral)
    dst = np.array([
        [tl, ty_top],
        [w - 1.0 + tr, ty_top],
        [w - 1.0 + br, h - 1.0 + ty_bottom],
        [bl, h - 1.0 + ty_bottom]
    ], dtype=np.float32)
    
    # Calculate bounding box of warped corners
    min_x = min(c[0] for c in dst)
    max_x = max(c[0] for c in dst)
    min_y = min(c[1] for c in dst)
    max_y = max(c[1] for c in dst)
    
    # Add margin for anti-aliasing
    margin = 5
    
    # Output size must fit the warped corners
    out_w = int(max_x - min_x) + 1 + margin * 2
    out_h = int(max_y - min_y) + 1 + margin * 2
    
    # Offset to shift corners to positive coordinates
    offset_x = -min_x + margin
    offset_y = -min_y + margin
    
    # Adjusted destination corners (all positive)
    dst_offset = dst.copy()
    dst_offset[:, 0] += offset_x
    dst_offset[:, 1] += offset_y
    
    # Get perspective transform
    M = cv2.getPerspectiveTransform(src, dst_offset)
    
    # Warp RGB channels
    warped_rgb = cv2.warpPerspective(
        img_rgba[:, :, :3],
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # Warp alpha channel - fully opaque source alpha
    alpha_channel = img_rgba[:, :, 3] if img_rgba.shape[2] == 4 else np.full((h, w), 255, dtype=np.uint8)
    warped_alpha = cv2.warpPerspective(
        alpha_channel,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Combine RGB and alpha
    warped = np.dstack([warped_rgb, warped_alpha])
    
    # Return corners in new local coordinates (all positive, within warped bounds)
    corners = dst_offset.copy()
    
    return warped, corners


def add_drop_shadow(img_rgba, corners, offset=2, spread=25, opacity=0.5):
    """Add a realistic drop shadow for photos lying flat.
    
    Creates a soft, spread shadow around the photo - the shadow will be
    composited onto the background when placed on the scene.
    
    Args:
        img_rgba: Warped photo with alpha channel
        corners: 4 corner positions of the warped photo
        offset: How far shadow starts from photo edge (default 2px)
        spread: How wide the shadow spreads (default 25px)
        opacity: Shadow intensity (default 0.5)
    
    Returns:
        (photo_with_shadow_canvas, new_corners, shadow_mask)
        The shadow_mask should be composited onto background, then photo on top.
    """
    h, w = img_rgba.shape[:2]
    
    # Calculate bounds with spread
    min_x = min(c[0] for c in corners) - offset - spread
    max_x = max(c[0] for c in corners) + offset + spread
    min_y = min(c[1] for c in corners) - offset - spread
    max_y = max(c[1] for c in corners) + offset + spread
    
    # Output canvas size
    out_w = int(max_x - min_x) + 1
    out_h = int(max_y - min_y) + 1
    offset_x = -min_x
    offset_y = -min_y
    
    # Photo corners in output coordinates
    photo_corners = corners.copy().astype(np.float32)
    photo_corners[:, 0] += offset_x
    photo_corners[:, 1] += offset_y
    
    # Create shadow mask
    shadow_mask = np.zeros((out_h, out_w), dtype=np.float32)
    cv2.fillPoly(shadow_mask, [photo_corners.astype(np.int32)], 255)
    
    # Dilate for offset + spread
    kernel_size = max(3, (offset + spread) // 2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(shadow_mask.astype(np.uint8), kernel, iterations=1)
    
    # Subtract photo to get shadow ring
    shadow_mask = cv2.subtract(expanded, shadow_mask.astype(np.uint8)).astype(np.float32)
    
    # Blur for soft edges
    blur_size = max(7, (spread // 2) | 1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_size, blur_size), 0)
    
    # Normalize to opacity
    shadow_mask = shadow_mask / 255.0 * opacity
    
    # Create output canvas with photo placed correctly
    output = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    
    # Place photo onto output
    px1, py1 = int(offset_x), int(offset_y)
    px2, py2 = int(offset_x + w), int(offset_y + h)
    
    # Clamp
    ox1, oy1 = max(0, px1), max(0, py1)
    ox2, oy2 = min(out_w, px2), min(out_h, py2)
    sx1, sy1 = ox1 - px1, oy1 - py1
    sx2, sy2 = sx1 + (ox2 - ox1), sy1 + (oy2 - oy1)
    
    # Copy photo to output (no RGB modification)
    output[oy1:oy2, ox1:ox2] = img_rgba[sy1:sy2, sx1:sx2]
    
    return output, photo_corners, shadow_mask


def add_variable_blur(img_rgba, min_blur=0, max_blur=8, edge_focus=0.5):
    """Add variable blur simulating wavy/lens out-of-focus effect.
    
    Creates a non-uniform blur where some areas are more blurred than others,
    with optional focus on edges for realistic curled/warped photo effect.
    Both RGB and alpha channels are blurred for proper edge blending.
    
    Args:
        img_rgba: Image in BGRA format (with alpha channel)
        min_blur: Minimum blur kernel size (0 = no blur)
        max_blur: Maximum blur kernel size
        edge_focus: How much to focus blur on edges (0 = uniform pattern, 1 = edge-heavy)
    
    Returns:
        Image with variable blur applied (BGRA)
    """
    h, w = img_rgba.shape[:2]
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Create base sine wave pattern for organic variation
    freq1 = random.uniform(0.005, 0.015)
    freq2 = random.uniform(0.008, 0.025)
    phase_x = random.uniform(0, 2 * np.pi)
    phase_y = random.uniform(0, 2 * np.pi)
    wave = np.sin(x * freq1 + phase_x) * np.sin(y * freq2 + phase_y)
    
    # Normalize wave to 0-1
    wave_norm = (wave - wave.min()) / (wave.max() - wave.min())
    
    # Create edge distance map (normalized 0-1)
    edge_x = np.minimum(x / w, (w - x) / w)
    edge_y = np.minimum(y / h, (h - y) / h)
    edge_dist = np.minimum(edge_x, edge_y) * 2
    
    # Combine wave pattern with edge focus
    pattern = wave_norm * (1 - edge_focus) + edge_dist * edge_focus
    
    # Add some random variation
    if random.random() < 0.5:
        pattern = pattern * random.uniform(0.8, 1.2)
    
    pattern = np.clip(pattern, 0, 1)
    
    # Scale to blur range
    blur_map = pattern * (max_blur - min_blur) + min_blur
    blur_map = np.maximum(blur_map, 1)
    
    # Create blurred versions of RGB at different levels
    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]
    
    levels = [min_blur, (min_blur + max_blur) / 2, max_blur]
    blurred_rgb = []
    blurred_alpha = []
    
    for level in levels:
        kernel = max(3, int(level * 2) | 1)
        if kernel > 1:
            blurred_rgb.append(cv2.GaussianBlur(rgb, (kernel, kernel), 0))
            blurred_alpha.append(cv2.GaussianBlur(alpha, (kernel, kernel), 0))
        else:
            blurred_rgb.append(rgb.copy())
            blurred_alpha.append(alpha.copy())
    
    # Interpolate based on blur map
    result_rgb = np.zeros_like(rgb, dtype=np.float32)
    result_alpha = np.zeros_like(alpha, dtype=np.float32)
    
    pattern_norm = (blur_map - min_blur) / (max_blur - min_blur) if max_blur > min_blur else np.zeros_like(blur_map)
    pattern_norm = np.clip(pattern_norm, 0, 1)
    
    # Level transitions
    mask1 = np.clip(pattern_norm * 2, 0, 1)
    mask2 = np.clip(pattern_norm * 2 - 1, 0, 1)
    
    for c in range(3):
        result_rgb[:, :, c] = (
            blurred_rgb[0][:, :, c].astype(np.float32) * (1 - mask1) +
            blurred_rgb[1][:, :, c].astype(np.float32) * (mask1 * (1 - mask2)) +
            blurred_rgb[2][:, :, c].astype(np.float32) * mask2
        )
    
    result_alpha = (
        blurred_alpha[0].astype(np.float32) * (1 - mask1) +
        blurred_alpha[1].astype(np.float32) * (mask1 * (1 - mask2)) +
        blurred_alpha[2].astype(np.float32) * mask2
    )
    
    # Combine RGB and alpha
    result = np.dstack([result_rgb.astype(np.uint8), result_alpha.astype(np.uint8)])
    
    return result


def add_glare(img_rgb, radius_ratio=0.4, intensity=0.65, cx=None, cy=None):
    """Add simulated glare using screen blend mode with radial gradient.
    
    Args:
        img_rgb: Image in BGR format (no alpha)
        radius_ratio: Size of glare spot relative to image (0.1-1.5)
        intensity: Brightness of glare center (0.1-1.0)
        cx, cy: Position override (None for random)
    
    Returns:
        Image with glare applied
    """
    h, w = img_rgb.shape[:2]
    
    # Random position - can be anywhere including well outside image
    if cx is None:
        cx = random.randint(int(w * -0.8), int(w * 1.8))
    if cy is None:
        cy = random.randint(int(h * -0.8), int(h * 1.8))
    
    radius = int(min(w, h) * radius_ratio)
    
    # Create radial gradient
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    gradient = np.clip(1 - dist / radius, 0, 1)
    
    # Apply falloff shaping (sharper falloff for more realistic glare)
    gradient = gradient ** 1.5
    
    # Screen blend mode: result = 1 - (1-a)*(1-b)
    gradient_float = (gradient * intensity).astype(np.float32)
    img_float = img_rgb.astype(np.float32) / 255.0
    
    result = 1.0 - (1.0 - img_float) * (1.0 - gradient_float[..., np.newaxis])
    result = (result * 255).astype(np.uint8)
    
    return result


def composite_with_shadow(bg, photo, shadow_mask, pos):
    """Composite photo onto background, applying shadow first.
    
    The shadow is applied to the background before the photo is placed.
    """
    bx, by = pos
    bh, bw = bg.shape[:2]
    ph, pw = photo.shape[:2]
    sh, sw = shadow_mask.shape[:2]
    
    # Calculate output bounds
    x1 = max(0, bx)
    y1 = max(0, by)
    x2 = min(bw, bx + sw)
    y2 = min(bh, by + sh)
    
    if x1 >= x2 or y1 >= y2:
        return bg
    
    # Calculate offsets
    ox1 = x1 - bx
    oy1 = y1 - by
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)
    
    # Apply shadow to background (darken where shadow exists)
    shadow_region = shadow_mask[oy1:oy2, ox1:ox2]
    bg_region = bg[y1:y2, x1:x2].astype(np.float32)
    
    # Darken background by shadow amount
    darkening = np.stack([shadow_region] * 3, axis=-1) * 0.4
    bg[y1:y2, x1:x2] = np.clip(bg_region - darkening * 255 * 0.5, 0, 255).astype(np.uint8)
    
    # Now composite photo on top
    return composite_overlay(bg, photo, (bx, by))


class BackgroundGenerator:
    """Generates simple backgrounds."""
    
    def __init__(self):
        self.types = ['wood', 'marble', 'solid', 'fabric', 'laminate']
    
    def generate(self, width, height):
        bg_type = random.choice(self.types)
        
        if bg_type == 'wood':
            return self._generate_wood(width, height)
        elif bg_type == 'marble':
            return self._generate_marble(width, height)
        elif bg_type == 'solid':
            return self._generate_solid(width, height)
        elif bg_type == 'fabric':
            return self._generate_fabric(width, height)
        else:
            return self._generate_laminate(width, height)
    
    def _generate_wood(self, w, h):
        base_r = random.randint(130, 170)
        base_g = random.randint(95, 125)
        base_b = random.randint(65, 90)
        
        arr = np.full((h, w, 3), [base_r, base_g, base_b], dtype=np.float32)
        
        for y in range(0, h, random.randint(10, 25)):
            offset = random.randint(-10, 10)
            arr[y, :] = np.clip(arr[y, :] + offset, 0, 255)
        
        img = Image.fromarray(arr.astype(np.uint8))
        return np.array(img.filter(ImageFilter.GaussianBlur(radius=2)))
    
    def _generate_marble(self, w, h):
        base = random.randint(200, 230)
        arr = np.full((h, w, 3), base, dtype=np.float32)
        
        for _ in range(random.randint(2, 4)):
            x0, y0 = random.randint(0, w), random.randint(0, h)
            length = random.randint(h // 4, h)
            angle = random.uniform(0, 2 * math.pi)
            
            for i in range(0, length, 3):
                x, y = int(x0 + math.cos(angle) * i), int(y0 + math.sin(angle) * i)
                if 0 <= x < w and 0 <= y < h:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                arr[ny, nx] = base - 30
        
        img = Image.fromarray(arr.astype(np.uint8))
        return np.array(img.filter(ImageFilter.GaussianBlur(radius=3)))
    
    def _generate_solid(self, w, h):
        c = random.randint(160, 220)
        return np.full((h, w, 3), [c, c, c], dtype=np.uint8)
    
    def _generate_fabric(self, w, h):
        base = random.randint(150, 190)
        arr = np.full((h, w, 3), base, dtype=np.float32)
        arr += np.random.randint(-10, 10, (h, w, 3))
        arr = np.clip(arr, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
        return np.array(img.filter(ImageFilter.GaussianBlur(radius=1)))
    
    def _generate_laminate(self, w, h):
        base_r = random.randint(160, 180)
        base_g = random.randint(130, 150)
        base_b = random.randint(90, 110)
        
        arr = np.full((h, w, 3), [base_r, base_g, base_b], dtype=np.float32)
        ph = random.randint(70, 100)
        for y in range(0, h, ph):
            arr[y, :] = np.clip(arr[y, :] - 15, 0, 255)
        
        img = Image.fromarray(arr.astype(np.uint8))
        return np.array(img.filter(ImageFilter.GaussianBlur(radius=1)))


def apply_luma_gradient(bg_arr):
    """Apply luma gradient to background."""
    if random.random() > 0.7:
        return bg_arr
    
    h, w = bg_arr.shape[:2]
    grad_type = random.choice(['linear_h', 'linear_v', 'corner', 'radial'])
    
    overlay = np.zeros((h, w), dtype=np.float32)
    
    if grad_type == 'linear_h':
        for x in range(w):
            overlay[:, x] = (x / w) * random.randint(20, 35)
    elif grad_type == 'linear_v':
        for y in range(h):
            overlay[y, :] = (y / h) * random.randint(20, 35)
    elif grad_type == 'corner':
        max_d = math.sqrt(w**2 + h**2)
        for y in range(h):
            for x in range(w):
                d = math.sqrt(x**2 + y**2)
                overlay[y, x] = (d / max_d) * random.randint(30, 45)
    else:
        cx, cy = w // 2, h // 2
        max_d = math.sqrt(cx**2 + cy**2)
        for y in range(h):
            for x in range(w):
                d = math.sqrt((x - cx)**2 + (y - cy)**2)
                overlay[y, x] = (d / max_d) * random.randint(25, 40)
    
    result = bg_arr.astype(np.float32)
    result[:, :, 0] = np.clip(result[:, :, 0] - overlay, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] - overlay, 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] - overlay, 0, 255)
    
    return result.astype(np.uint8)


def process_photo(img_path, min_size, max_size):
    """Process source image into photo instance with perspective transform.
    
    Returns: (warped_photo, corners, shadow_mask) 
        corners are the 4 corner positions
        shadow_mask is the drop shadow (or None)
    """
    try:
        # Use cv2 to load consistently in BGR format
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None, None, None
        # Convert BGR to RGBA
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    except:
        return None, None, None
    
    # Resize to target size
    ih, iw = img.shape[:2]
    if iw > ih:
        target_w = random.randint(min_size, max_size)
        target_h = int(ih * (target_w / iw))
    else:
        target_h = random.randint(min_size, max_size)
        target_w = int(iw * (target_h / ih))
    
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Extract RGB for processing
    rgb = img[:, :, :3]
    
    # Apply color effects (increase AND decrease variations)
    # Saturation adjustment
    if random.random() < 0.5:
        # Convert to HSV, adjust S channel, convert back
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_factor = random.uniform(0.5, 1.8)  # 0.5 = half saturation, 1.8 = 80% more
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Contrast adjustment
    if random.random() < 0.5:
        contrast_factor = random.uniform(0.7, 1.5)  # 0.7 = less contrast, 1.5 = more
        rgb = cv2.convertScaleAbs(rgb, alpha=contrast_factor, beta=0)
    
    # Brightness adjustment
    if random.random() < 0.5:
        brightness_offset = random.uniform(-40, 40)  # Negative = darker, positive = brighter
        rgb = cv2.convertScaleAbs(rgb, alpha=1.0, beta=brightness_offset)
    
    # Gamma curve adjustment
    if random.random() < 0.4:
        gamma = random.uniform(0.6, 1.6)  # < 1 = brighten shadows, > 1 = darken shadows
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        rgb = cv2.LUT(rgb, table)
    
    # Simulated glare (screen blend with radial gradient)
    if random.random() < 0.3:  # 30% chance
        radius_ratio = random.uniform(0.1, 1.5)  # Tiny to massive
        intensity = random.uniform(0.1, 1.0)  # Very subtle to full white
        rgb = add_glare(rgb, radius_ratio=radius_ratio, intensity=intensity)
    
    # Re-add alpha channel (fully opaque)
    img = np.dstack([rgb, np.full((target_h, target_w), 255, dtype=np.uint8)])
    
    # Variable blur (wavy/fluid lens effect) - applied after effects
    if random.random() < 0.3:  # 30% chance
        min_blur = random.uniform(0, 2)  # No blur to slight blur
        max_blur = random.uniform(3, 10)  # Up to quite blurred
        edge_focus = random.uniform(0.3, 0.8)  # 30-80% edge focus
        img = add_variable_blur(img, min_blur=min_blur, max_blur=max_blur, edge_focus=edge_focus)
    
    # Apply perspective warp - returns corners for keypoint tracking
    warped, corners = cv2_perspective_warp(img, strength=random.uniform(0.08, 0.16))
    
    # Optionally add drop shadow
    shadow_mask = None
    if random.random() < 0.7:  # 70% chance of shadow
        # Parameters for realistic flat photo shadows
        offset = random.randint(0, 15)  # How far shadow starts from edge (0-15px)
        spread = random.randint(20, 40)  # How wide the shadow spreads (20-40px)
        opacity = random.uniform(0.15, 0.50)  # Shadow intensity
        warped, corners, shadow_mask = add_drop_shadow(
            warped, corners,
            offset=offset,
            spread=spread,
            opacity=opacity
        )
    
    return warped, corners, shadow_mask


def convert_bgra_to_rgba(img):
    """Convert BGRA image (from cv2) to RGBA for consistent compositing."""
    if img.shape[2] == 4:
        # Swap R and B channels: BGRA -> RGBA
        return img[:, :, [2, 1, 0, 3]]
    elif img.shape[2] == 3:
        # Swap R and B channels: BGR -> RGB
        return img[:, :, [2, 1, 0]]
    return img


def find_position_with_corners(photo_w, photo_h, placed_list, local_corners, margin=8, canvas_w=4000, canvas_h=3000):
    """Find non-overlapping position using actual warped corner coordinates.
    
    Args:
        photo_w, photo_h: Photo dimensions (fallback bounds)
        placed_list: List of placed photos as dicts with 'corners' key
        local_corners: 4 corner points in photo-local coordinates (numpy array or list)
        margin: Minimum gap between photos in pixels
        canvas_w, canvas_h: Canvas dimensions
    
    Returns:
        (x, y) tuple if position found, None otherwise
    """
    candidates = []
    max_attempts = 2000
    
    # Precompute corner bounds for quick reject
    corners = np.array(local_corners)
    cx, cy = corners[:, 0], corners[:, 1]
    corner_min_x, corner_max_x = cx.min(), cx.max()
    corner_min_y, corner_max_y = cy.min(), cy.max()
    corner_w = corner_max_x - corner_min_x
    corner_h = corner_max_y - corner_min_y
    
    for attempt in range(max_attempts):
        # Random position within canvas bounds
        x = random.randint(margin, canvas_w - int(corner_w) - margin)
        y = random.randint(margin, canvas_h - int(corner_h) - margin)
        
        # Compute absolute corners for this position
        abs_corners = corners + np.array([x, y])
        
        # Check for overlap with all placed photos
        overlaps = False
        for placed in placed_list:
            placed_corners = placed['corners']  # Already in absolute coords
            if polygons_overlap(abs_corners, placed_corners, margin):
                overlaps = True
                break
        
        if not overlaps:
            # Prefer positions closer to edges for denser packing
            edge_dist = min(x, y, canvas_w - x - corner_w, canvas_h - y - corner_h)
            candidates.append((x, y, edge_dist))
    
    if candidates:
        candidates.sort(key=lambda c: c[2])
        return candidates[0][0], candidates[0][1]
    
    return None


def polygons_overlap(poly1, poly2, margin=0):
    """Check if two convex polygons overlap using separating axis theorem.
    
    Args:
        poly1, poly2: Nx2 numpy arrays of polygon vertices in absolute coords
        margin: Extra buffer space
    Returns:
        True if polygons overlap, False otherwise
    """
    # Quick bounding box reject
    min1_x, max1_x = poly1[:, 0].min(), poly1[:, 0].max()
    min1_y, max1_y = poly1[:, 1].min(), poly1[:, 1].max()
    min2_x, max2_x = poly2[:, 0].min(), poly2[:, 0].max()
    min2_y, max2_y = poly2[:, 1].min(), poly2[:, 1].max()
    
    # Expand by margin
    if max1_x + margin < min2_x or min1_x - margin > max2_x:
        return False
    if max1_y + margin < min2_y or min1_y - margin > max2_y:
        return False
    
    # SAT with all edge normals from both polygons
    for poly in [poly1, poly2]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            
            # Edge vector
            edge = p2 - p1
            if np.abs(edge).sum() < 0.001:  # Skip degenerate edges
                continue
            
            # Normal (perpendicular to edge)
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            
            # Project both polygons onto normal
            proj1 = poly1 @ normal
            proj2 = poly2 @ normal
            
            # Check for gap
            if proj1.max() + margin < proj2.min() or proj2.max() + margin < proj1.min():
                return False  # Separating axis found
    
    return True  # No separating axis found, polygons overlap

def composite_overlay(bg, overlay, pos):
    """Composite overlay onto background - PIXEL PERFECT alpha compositing."""
    bx, by = pos
    
    # Get bounds
    bh, bw = bg.shape[:2]
    oh, ow = overlay.shape[:2]
    
    # Calculate overlap region
    x1 = max(0, bx)
    y1 = max(0, by)
    x2 = min(bw, bx + ow)
    y2 = min(bh, by + oh)
    
    if x1 >= x2 or y1 >= y2:
        return bg
    
    # Calculate offset into overlay
    ox1 = x1 - bx
    oy1 = y1 - by
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)
    
    # Extract regions
    bg_region = bg[y1:y2, x1:x2].astype(np.float32)
    ov_region = overlay[oy1:oy2, ox1:ox2].astype(np.float32)
    
    # Get alpha (4th channel)
    if ov_region.shape[2] == 4:
        alpha = ov_region[:, :, 3].astype(np.float32) / 255.0
        ov_rgb = ov_region[:, :, :3]
    else:
        alpha = np.ones((ov_region.shape[0], ov_region.shape[1]), dtype=np.float32)
        ov_rgb = ov_region[:, :, :3]
    
    # Ensure bg is RGB only (strip alpha if present)
    if bg_region.shape[2] == 4:
        bg_rgb = bg_region[:, :, :3]
    else:
        bg_rgb = bg_region
    
    # Reshape for broadcasting
    h_reg, w_reg = bg_rgb.shape[:2]
    alpha = alpha.reshape(h_reg, w_reg, 1)
    
    # Alpha compositing: result = bg * (1 - alpha) + overlay * alpha
    result = bg_rgb * (1 - alpha) + ov_rgb * alpha
    
    # Write back to bg
    bg[y1:y2, x1:x2, :3] = result.astype(np.uint8)
    
    return bg


def get_warped_bounds(corners):
    """Calculate bounding box of warped corners."""
    min_x = min(c[0] for c in corners)
    max_x = max(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_y = max(c[1] for c in corners)
    return min_x, min_y, max_x - min_x, max_y - min_y


def render_scene(config, sources, requested_photos=None):
    """Render scene with densely placed photos.
    
    Args:
        config: Configuration object
        sources: List of source image paths
        requested_photos: Number of photos to place (or None for random)
    
    Returns: (image, infos, canvas_size)
    """
    # Determine canvas size - must be at least 2000x3000 (4:3)
    # 4:3 = width : height
    canvas_h = 3000
    canvas_w = int(canvas_h * 4 / 3)  # 4000
    
    # Create background
    bg_gen = BackgroundGenerator()
    bg = bg_gen.generate(canvas_w, canvas_h)
    bg = apply_luma_gradient(bg)
    
    # Number of photos
    if requested_photos is not None:
        num_photos = requested_photos
    else:
        num_photos = random.randint(config.num_photos_min, config.num_photos_max)
    num_photos = min(num_photos, len(sources))
    
    # Sample sources
    if len(sources) >= num_photos:
        sampled = random.sample(sources, num_photos)
    else:
        sampled = (sources * (num_photos // len(sources) + 1))[:num_photos]
    
    # Calculate photo sizes based on number of photos
    # Goal: fill canvas well with appropriate photo sizes
    edge_margin = 15
    photo_gap = 8
    
    if num_photos == 1:
        # Single photo: fill most of the canvas
        # Target: fill ~70% of the smaller dimension
        target_size = min(canvas_w, canvas_h) * 0.7
        min_size = int(target_size * 0.8)
        max_size = int(target_size * 1.1)
    elif num_photos <= 3:
        # 2-3 photos: fill width with 1-2 rows
        cols = num_photos
        rows = 1
        avail_w = canvas_w - 2 * edge_margin - (cols - 1) * photo_gap
        target_size = avail_w // cols
        min_size = int(target_size * 0.8)
        max_size = int(target_size * 1.1)
    else:
        # 4+ photos: use grid layout
        cols = int(math.ceil(math.sqrt(num_photos)))
        rows = int(math.ceil(num_photos / cols))
        avail_w = canvas_w - 2 * edge_margin - (cols - 1) * photo_gap
        avail_h = canvas_h - 2 * edge_margin - (rows - 1) * photo_gap
        target_size = min(avail_w // cols, avail_h // rows)
        min_size = int(target_size * 0.8)
        max_size = int(target_size * 1.1)
    
    # Clamp to reasonable range
    min_size = max(200, min(min_size, 2000))
    max_size = max(min_size + 50, min(max_size, 2500))
    
    # Process photos - convert from BGR to RGB for consistent color space
    photos = []
    for path in sampled:
        photo, corners, shadow_mask = process_photo(str(path), min_size, max_size)
        if photo is not None and photo.shape[0] > 50 and photo.shape[1] > 50:
            # Convert from BGRA to RGBA for consistent compositing
            photo = convert_bgra_to_rgba(photo)
            photos.append((photo, corners, shadow_mask))
    
    # Sort by area (larger first) - helps with packing
    photos.sort(key=lambda p: p[0].shape[0] * p[0].shape[1], reverse=True)
    
    # Try to place photos, reducing sizes if needed
    max_retries = 5
    scale_factor = 0.85  # Reduce size by 15% each retry
    
    for retry in range(max_retries):
        # Try to place all photos
        placed = []
        infos = []
        
        for photo, corners, shadow_mask in photos:
            # Use tight bounding box from warped corners for collision detection
            pos = find_position_with_corners(
                photo.shape[1], photo.shape[0], placed, corners, 
                margin=8, canvas_w=canvas_w, canvas_h=canvas_h
            )
            if pos is None:
                continue
            
            px, py = pos
            
            # Apply shadow first if exists, then composite photo
            if shadow_mask is not None:
                bg = composite_with_shadow(bg, photo, shadow_mask, (px, py))
            else:
                bg = composite_overlay(bg, photo, (px, py))
            
            # Calculate absolute corners
            abs_corners = np.array([
                [corners[0][0] + px, corners[0][1] + py],
                [corners[1][0] + px, corners[1][1] + py],
                [corners[2][0] + px, corners[2][1] + py],
                [corners[3][0] + px, corners[3][1] + py],
            ])
            
            # Store with corners for accurate collision detection
            placed.append({'x': px, 'y': py, 'w': photo.shape[1], 'h': photo.shape[0], 'corners': abs_corners})
            
            # Keypoints at actual warped corners
            kps = [
                (px + corners[0][0], py + corners[0][1]),  # TL
                (px + corners[1][0], py + corners[1][1]),  # TR
                (px + corners[2][0], py + corners[2][1]),  # BR
                (px + corners[3][0], py + corners[3][1]),  # BL
            ]
            
            # Calculate center from warped corners
            xc = sum(c[0] for c in kps) / 4
            yc = sum(c[1] for c in kps) / 4
            
            # Calculate bounding box for the box format
            min_x = min(k[0] for k in kps)
            max_x = max(k[0] for k in kps)
            min_y = min(k[1] for k in kps)
            max_y = max(k[1] for k in kps)
            
            bw_box = (max_x - min_x) / canvas_w
            bh_box = (max_y - min_y) / canvas_h
            
            infos.append({
                'x_center': xc / canvas_w,
                'y_center': yc / canvas_h,
                'width': bw_box,
                'height': bh_box,
                'keypoints': kps
            })
        
        # Check if we placed enough photos
        if len(placed) >= num_photos:
            break
        
        # If not, regenerate photos at smaller size
        if retry < max_retries - 1:
            new_min = int(min_size * scale_factor)
            new_max = int(max_size * scale_factor)
            
            # Ensure minimum size
            if new_min < 150:
                break  # Can't go smaller
            
            min_size = new_min
            max_size = new_max
            
            # Regenerate photos at smaller size
            photos = []
            for path in sampled:
                photo, corners, shadow_mask = process_photo(str(path), min_size, max_size)
                if photo is not None and photo.shape[0] > 50 and photo.shape[1] > 50:
                    photo = convert_bgra_to_rgba(photo)
                    photos.append((photo, corners, shadow_mask))
            
            photos.sort(key=lambda p: p[0].shape[0] * p[0].shape[1], reverse=True)
    
    return bg, infos, (canvas_w, canvas_h)


class DatasetGenerator:
    def __init__(self, config, sources):
        self.config = config
        self.sources = [f for f in sources if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
        
        if len(self.sources) < 5:
            raise ValueError(f"Need at least 5 source images")
    
    def generate(self):
        out_dir = Path(self.config.output_dir)
        
        for d in ['images/train', 'images/val', 'labels/train', 'labels/val']:
            (out_dir / d).mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {self.config.num_train_images} training images...")
        self._generate_images(
            out_dir / 'images/train',
            out_dir / 'labels/train',
            self.config.num_train_images,
            'train'
        )
        
        print(f"\nGenerating {self.config.num_val_images} validation images...")
        self._generate_images(
            out_dir / 'images/val',
            out_dir / 'labels/val',
            self.config.num_val_images,
            'val'
        )
        
        self._save_yaml(out_dir)
        print(f"\nDone!")
    
    def _generate_images(self, img_dir, lbl_dir, num, prefix):
        for i in range(num):
            bg, infos, canvas_size = render_scene(self.config, self.sources)
            canvas_w, canvas_h = canvas_size
            
            # Background is already RGB (from BackgroundGenerator)
            # Convert to PIL format for saving
            pil_img = Image.fromarray(bg.astype('uint8'))
            
            img_path = img_dir / f"{prefix}_{i:05d}.jpg"
            pil_img.save(img_path, quality=95)
            
            lbl_path = lbl_dir / f"{prefix}_{i:05d}.txt"
            self._save_labels(lbl_path, infos, canvas_w, canvas_h)
            
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num}")
    
    def _save_labels(self, path, infos, canvas_w, canvas_h):
        with open(path, 'w') as f:
            for p in infos:
                line = f"0 {p['x_center']:.6f} {p['y_center']:.6f} {p['width']:.6f} {p['height']:.6f}"
                for kx, ky in p['keypoints']:
                    line += f" {kx / canvas_w:.6f} {ky / canvas_h:.6f} 2"
                line += "\n"
                f.write(line)
    
    def _save_yaml(self, out_dir):
        yaml = f"""# YOLO Pose Dataset
path: {out_dir.absolute()}
train: images/train
val: images/val

kpt_shape: [4, 2]
flip_idx: [1, 0, 3, 2]

nc: 1
names:
  0: photo_corner

pose:
  flip_idx: [1, 0, 3, 2]
"""
        p = Path("../training/dataset.yaml")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml)
        print(f"  Config: {p.absolute()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=DEFAULT_CONFIG["num_train_images"])
    parser.add_argument("--num-val", type=int, default=DEFAULT_CONFIG["num_val_images"])
    parser.add_argument("--source-images", type=str, default=DEFAULT_CONFIG["source_images"])
    parser.add_argument("--output", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--min-photos", type=int, default=DEFAULT_CONFIG["num_photos_min"])
    parser.add_argument("--max-photos", type=int, default=DEFAULT_CONFIG["num_photos_max"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    
    args = parser.parse_args()
    
    src_dir = Path(args.source_images)
    if not src_dir.exists():
        print(f"Error: {src_dir} not found")
        return
    
    sources = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg")) + \
              list(src_dir.glob("*.png")) + list(src_dir.glob("*.webp"))
    
    print(f"Found {len(sources)} source images")
    
    config = Config(
        num_train_images=args.num_train,
        num_val_images=args.num_val,
        output_dir=args.output,
        num_photos_min=args.min_photos,
        num_photos_max=args.max_photos,
        seed=args.seed,
    )
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    DatasetGenerator(config, sources).generate()


if __name__ == "__main__":
    main()
