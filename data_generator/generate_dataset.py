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


def find_position(w, h, placed, margin=8):
    """Find non-overlapping position."""
    candidates = []
    
    for _ in range(100):
        x = random.randint(margin, 1920 - w - margin)
        y = random.randint(margin, 1080 - h - margin)
        
        if not _overlaps(x, y, w, h, placed, margin):
            edge_score = min(x, y, 1920 - x - w, 1080 - y - h)
            candidates.append((x, y, edge_score))
    
    if candidates:
        candidates.sort(key=lambda c: c[2])
        return candidates[0][0], candidates[0][1]
    
    return None


def _overlaps(x, y, w, h, placed, margin):
    for px, py, pw, ph in placed:
        if (x < px + pw + margin and x + w + margin > px and
            y < py + ph + margin and y + h + margin > py):
            return True
    return False


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


def render_scene(config, sources):
    """Render scene with densely placed photos."""
    w, h = 1920, 1080
    
    # Create background
    bg_gen = BackgroundGenerator()
    bg = bg_gen.generate(w, h)
    bg = apply_luma_gradient(bg)
    
    # Number of photos
    num_photos = random.randint(config.num_photos_min, config.num_photos_max)
    
    # Sample sources
    if len(sources) >= num_photos:
        sampled = random.sample(sources, num_photos)
    else:
        sampled = (sources * (num_photos // len(sources) + 1))[:num_photos]
    
    # Process photos
    photos = []
    for path in sampled:
        photo, corners, shadow_mask = process_photo(path, config.min_photo_size, config.max_photo_size)
        if photo is not None and photo.shape[0] > 50 and photo.shape[1] > 50:
            photos.append((photo, corners, shadow_mask))
    
    # Sort by area (larger first)
    photos.sort(key=lambda p: p[0].shape[0] * p[0].shape[1], reverse=True)
    
    # Place photos
    placed = []
    infos = []
    
    for photo, corners, shadow_mask in photos:
        ph, pw = photo.shape[:2]
        
        # Use photo size for collision detection
        pos = find_position(pw, ph, placed, margin=6)
        if pos is None:
            continue
        
        px, py = pos
        
        # Apply shadow first if exists, then composite photo
        if shadow_mask is not None:
            bg = composite_with_shadow(bg, photo, shadow_mask, (px, py))
        else:
            bg = composite_overlay(bg, photo, (px, py))
        
        placed.append((px, py, pw, ph))
        
        # Keypoints at actual warped corners (corners are in local coords)
        # corners order: [TL, TR, BR, BL]
        kps = [
            (px + corners[0][0], py + corners[0][1]),  # TL
            (px + corners[1][0], py + corners[1][1]),  # TR
            (px + corners[2][0], py + corners[2][1]),  # BR
            (px + corners[3][0], py + corners[3][1]),  # BL
        ]
        
        # Calculate center from warped corners
        xc = sum(c[0] for c in kps) / 4 + px
        yc = sum(c[1] for c in kps) / 4 + py
        
        # Calculate bounding box for the box format
        min_x = min(k[0] for k in kps)
        max_x = max(k[0] for k in kps)
        min_y = min(k[1] for k in kps)
        max_y = max(k[1] for k in kps)
        
        bw_box = (max_x - min_x) / 1920
        bh_box = (max_y - min_y) / 1080
        
        infos.append({
            'x_center': xc / 1920,
            'y_center': yc / 1080,
            'width': bw_box,
            'height': bh_box,
            'keypoints': kps
        })
    
    return bg, infos


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
            bg, infos = render_scene(self.config, self.sources)
            
            pil_img = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
            
            img_path = img_dir / f"{prefix}_{i:05d}.jpg"
            pil_img.save(img_path, quality=95)
            
            lbl_path = lbl_dir / f"{prefix}_{i:05d}.txt"
            self._save_labels(lbl_path, infos)
            
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num}")
    
    def _save_labels(self, path, infos):
        with open(path, 'w') as f:
            for p in infos:
                line = f"0 {p['x_center']:.6f} {p['y_center']:.6f} {p['width']:.6f} {p['height']:.6f}"
                for kx, ky in p['keypoints']:
                    line += f" {kx / 1920:.6f} {ky / 1080:.6f} 2"
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
