#!/usr/bin/env python3
"""
Photo Pose Detector - Synthetic Training Data Generator (v14)

Uses OpenCV for proper image transformations with alpha channel support.
Keypoints are placed at actual warped corners, not bounding box.
Implements binary search for optimal fill percentage.
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
    
    Returns:
        warped_rgba: The warped image
        corners: Corner coordinates relative to warped image
        M: The perspective transform matrix (for reverse mapping)
        src: Source points
        dst: Destination points (before offset)
    """
    h, w = img_rgba.shape[:2]
    
    src = np.array([
        [0.0, 0.0],
        [w - 1.0, 0.0],
        [w - 1.0, h - 1.0],
        [0.0, h - 1.0]
    ], dtype=np.float32)
    
    tl = random.uniform(-w * strength, w * strength)
    tr = random.uniform(-w * strength, w * strength)
    bl = random.uniform(-w * strength, w * strength)
    br = random.uniform(-w * strength, w * strength)
    ty_top = random.uniform(-h * strength * 0.5, h * strength * 0.5)
    ty_bottom = random.uniform(-h * strength * 0.5, h * strength * 0.5)
    
    dst = np.array([
        [tl, ty_top],
        [w - 1.0 + tr, ty_top],
        [w - 1.0 + br, h - 1.0 + ty_bottom],
        [bl, h - 1.0 + ty_bottom]
    ], dtype=np.float32)
    
    min_x = min(c[0] for c in dst)
    max_x = max(c[0] for c in dst)
    min_y = min(c[1] for c in dst)
    max_y = max(c[1] for c in dst)
    
    margin = 5
    out_w = int(max_x - min_x) + 1 + margin * 2
    out_h = int(max_y - min_y) + 1 + margin * 2
    
    offset_x = -min_x + margin
    offset_y = -min_y + margin
    
    dst_offset = dst.copy()
    dst_offset[:, 0] += offset_x
    dst_offset[:, 1] += offset_y
    
    M = cv2.getPerspectiveTransform(src, dst_offset)
    
    warped_rgb = cv2.warpPerspective(
        img_rgba[:, :, :3],
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    warped_alpha = cv2.warpPerspective(
        img_rgba[:, :, 3],
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    warped_rgba = np.dstack([warped_rgb, warped_alpha])
    
    # Return corners relative to warped image, plus transform info
    corners = dst_offset.copy()
    
    return warped_rgba, corners, M, src, dst


def find_actual_visual_bounds(img_rgba):
    """Find the actual quadrilateral corners of visible pixels after perspective warp.
    
    Returns the 4 corner points of the visible photo region, ordered as:
    0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
    """
    if img_rgba.shape[2] == 4:
        alpha = img_rgba[:, :, 3]
    else:
        alpha = np.ones(img_rgba.shape[:2], dtype=np.uint8) * 255
    
    # Threshold alpha to find visible pixels
    _, thresh = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours of the visible region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # No visible pixels
        return np.array([[0, 0], [img_rgba.shape[2]-1, 0], 
                        [img_rgba.shape[2]-1, img_rgba.shape[0]-1], [0, img_rgba.shape[0]-1]], dtype=np.float32)
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate to a polygon with 4 points
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) >= 4:
        points = approx.reshape(-1, 2).astype(np.float32)
    else:
        # Fallback to convex hull
        hull = cv2.convexHull(largest_contour)
        points = hull.reshape(-1, 2).astype(np.float32)
        
        # If we have more than 4 points, we need to find the 4 extreme corners
        if len(points) > 4:
            # Simple approach: find bounding box corners
            min_x_idx = np.argmin(points[:, 0])
            max_x_idx = np.argmax(points[:, 0])
            min_y_idx = np.argmin(points[:, 1])
            max_y_idx = np.argmax(points[:, 1])
            
            # Get these 4 points
            candidates = [points[min_x_idx], points[max_x_idx], 
                         points[min_y_idx], points[max_y_idx]]
            # Remove duplicates
            unique_points = []
            for p in candidates:
                is_dup = False
                for up in unique_points:
                    if np.allclose(p, up, atol=5):
                        is_dup = True
                        break
                if not is_dup:
                    unique_points.append(p)
            points = np.array(unique_points, dtype=np.float32)
    
    if len(points) < 4:
        # Fallback to bounding box
        rows = np.any(thresh > 0, axis=1)
        cols = np.any(thresh > 0, axis=0)
        if not rows.any() or not cols.any():
            return np.array([[0, 0], [img_rgba.shape[2]-1, 0], 
                            [img_rgba.shape[2]-1, img_rgba.shape[0]-1], [0, img_rgba.shape[0]-1]], dtype=np.float32)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return np.array([[cmin, rmin], [cmax, rmin], [cmax, rmax], [cmin, rmax]], dtype=np.float32)
    
    # Now order the points: TL, TR, BR, BL
    # Top-left has minimum x+y
    # Bottom-right has maximum x+y
    # Top-right has minimum x-y (most negative)
    # Bottom-left has maximum x-y (most positive)
    
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]
    
    tl_idx = np.argmin(sums)
    br_idx = np.argmax(sums)
    tr_idx = np.argmin(diffs)
    bl_idx = np.argmax(diffs)
    
    # Handle case where some indices are the same
    indices = [tl_idx, tr_idx, br_idx, bl_idx]
    unique_indices = []
    for idx in indices:
        if idx not in unique_indices:
            unique_indices.append(idx)
        else:
            # Find another point
            for i, p in enumerate(points):
                if i not in unique_indices:
                    unique_indices.append(i)
                    break
    
    while len(unique_indices) < 4:
        for i in range(len(points)):
            if i not in unique_indices:
                unique_indices.append(i)
                break
    
    corners = np.array([
        points[unique_indices[0]],  # TL
        points[unique_indices[1]],  # TR
        points[unique_indices[2]],  # BR
        points[unique_indices[3]],  # BL
    ], dtype=np.float32)
    
    return corners


def add_drop_shadow(img_rgba, corners, offset=2, spread=25, opacity=0.5):
    """Add a realistic drop shadow."""
    h, w = img_rgba.shape[:2]
    
    min_x = min(c[0] for c in corners) - offset - spread
    max_x = max(c[0] for c in corners) + offset + spread
    min_y = min(c[1] for c in corners) - offset - spread
    max_y = max(c[1] for c in corners) + offset + spread
    
    out_w = max(1, int(max_x - min_x) + 1)
    out_h = max(1, int(max_y - min_y) + 1)
    offset_x = -min_x
    offset_y = -min_y
    
    photo_corners = corners.copy().astype(np.float32)
    photo_corners[:, 0] += offset_x
    photo_corners[:, 1] += offset_y
    
    shadow_mask = np.zeros((out_h, out_w), dtype=np.float32)
    cv2.fillPoly(shadow_mask, [photo_corners.astype(np.int32)], 255)
    
    kernel_size = max(3, (offset + spread) // 2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(shadow_mask.astype(np.uint8), kernel, iterations=1)
    shadow_mask = cv2.subtract(expanded, shadow_mask.astype(np.uint8)).astype(np.float32)
    
    blur_size = max(7, (spread // 2) | 1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_size, blur_size), 0)
    shadow_mask = shadow_mask / 255.0 * opacity
    
    output = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    
    px1, py1 = int(offset_x), int(offset_y)
    px2, py2 = int(offset_x + w), int(offset_y + h)
    
    ox1, oy1 = max(0, px1), max(0, py1)
    ox2, oy2 = min(out_w, px2), min(out_h, py2)
    
    dst_h = oy2 - oy1
    dst_w = ox2 - ox1
    
    sx1 = ox1 - px1
    sy1 = oy1 - py1
    sx2 = sx1 + dst_w
    sy2 = sy1 + dst_h
    
    if dst_h > 0 and dst_w > 0 and sy1 >= 0 and sx1 >= 0:
        src = img_rgba[sy1:sy2, sx1:sx2]
        if src.shape[0] == dst_h and src.shape[1] == dst_w:
            output[oy1:oy2, ox1:ox2] = src
        else:
            output[oy1:oy2, ox1:ox2] = cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    
    return output, photo_corners, shadow_mask


def add_variable_blur(img_rgba, min_blur=0, max_blur=8, edge_focus=0.5):
    """Add variable blur simulating wavy/lens out-of-focus effect."""
    if min_blur >= max_blur:
        return img_rgba
    
    h, w = img_rgba.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGBA))
    
    y_coords, x_coords = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
    positions = np.sqrt((x_coords - 0.5)**2 + (y_coords - 0.5)**2)
    
    noise = np.random.randn(h, w) * 0.2
    positions = np.clip(positions + noise * edge_focus, 0, 1)
    
    result = pil_img.copy()
    
    num_regions = 5
    for i in range(num_regions):
        mask = (positions >= i/num_regions) & (positions < (i+1)/num_regions)
        if not mask.any():
            continue
        
        blur_amt = min_blur + (max_blur - min_blur) * (i / num_regions)
        if blur_amt > 0.5:
            region = pil_img.copy()
            region_pixels = np.array(region)
            row_mask = np.any(mask, axis=1)
            col_mask = np.any(mask, axis=0)
            
            if row_mask.any() and col_mask.any():
                r_start = np.argmax(row_mask)
                r_end = len(row_mask) - np.argmax(row_mask[::-1])
                c_start = np.argmax(col_mask)
                c_end = len(col_mask) - np.argmax(col_mask[::-1])
                
                region_crop = region_pixels[r_start:r_end, c_start:c_end]
                if region_crop.size > 0:
                    blur_size = int(blur_amt * 2) | 1
                    blurred = cv2.GaussianBlur(region_crop, (blur_size, blur_size), 0)
                    result_pixels = np.array(result)
                    result_pixels[r_start:r_end, c_start:c_end] = blurred
                    result = Image.fromarray(result_pixels)
    
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGRA)


    return cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGRA)


def manipulate_photo(img_rgba, p=0.8):
    """Apply random contrast, gamma, saturation, brightness to a photo.

    Called on the original rectangular photo before perspective warp.
    This simulates real-world photo quality variation: different printing
    conditions, aging, exposure differences.

    Args:
        img_rgba: BGRA image
        p: probability of applying any manipulation

    Returns:
        Manipulated BGRA image (or original if no manipulation applied)
    """
    if random.random() > p:
        return img_rgba

    # Convert BGRA -> RGBA for PIL
    pil_img = Image.fromarray(cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGBA))

    # Brightness: factor in [0.7, 1.3]
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)

    # Contrast: factor in [0.7, 1.5]
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.5)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)

    # Saturation: factor in [0.4, 1.6]
    if random.random() < 0.5:
        factor = random.uniform(0.4, 1.6)
        pil_img = ImageEnhance.Color(pil_img).enhance(factor)

    # Gamma: exponent in [0.65, 1.5]
    if random.random() < 0.5:
        gamma = random.uniform(0.65, 1.5)
        arr = np.array(pil_img).astype(np.float32) / 255.0
        arr = np.clip(arr ** gamma, 0, 1)
        pil_img = Image.fromarray((arr * 255).astype(np.uint8))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)


def add_fuzzy_drop_shadow(img_rgba, corners, offset=(3, 3), blur_radius=18, opacity=0.45):
    """Add a fuzzy, dispersed drop shadow using pure Gaussian blurring.

    Simulates realistic soft shadows cast by photos on a table surface.
    The shadow is a filled shape at the photo's position, shifted by offset,
    then blurred to create a soft, dispersed edge. Composited under the photo.

    Args:
        img_rgba: BGRA image (rectangular)
        corners: 4x2 array of corner coordinates
        offset: (x, y) pixel displacement of shadow relative to photo
        blur_radius: std dev for Gaussian blur — larger = more diffuse
        opacity: shadow darkness (0.0–1.0)

    Returns:
        Composite image with soft shadow baked in (BGRA)
    """
    h, w = img_rgba.shape[:2]

    if img_rgba.shape[2] == 4:
        alpha = img_rgba[:, :, 3]
    else:
        alpha = np.ones((h, w), dtype=np.uint8) * 255

    # Extended canvas to accommodate blurred shadow
    pad = int(blur_radius * 3)
    ext_h = h + pad * 2
    ext_w = w + pad * 2

    ox, oy = offset

    # Build filled shadow shape: full photo area shifted by (ox, oy)
    # This is the key difference from the ring approach: shadow fills ENTIRE
    # area under the photo (not just the edge ring)
    shadow_mask = np.zeros((ext_h, ext_w), dtype=np.float32)

    # Scaled corners to extended canvas (centered + shifted)
    shifted_corners = corners.copy()
    shifted_corners[:, 0] += pad + ox
    shifted_corners[:, 1] += pad + oy

    cv2.fillPoly(shadow_mask, [shifted_corners.astype(np.int32)], 255)

    # Also create the photo shape (centered, no offset)
    photo_mask = np.zeros((ext_h, ext_w), dtype=np.float32)
    photo_centered = corners.copy()
    photo_centered[:, 0] += pad
    photo_centered[:, 1] += pad
    cv2.fillPoly(photo_mask, [photo_centered.astype(np.int32)], 255)

    # Shadow region = shadow shape minus photo shape
    # This handles the overlap correctly: shadow shows where photo ISN'T
    # within the shadow shape. The shadow shape is the full area shifted by offset.
    shadow_region = np.maximum(
        shadow_mask.astype(np.uint8),
        photo_mask.astype(np.uint8)
    ).astype(np.float32)
    shadow_region = cv2.subtract(
        shadow_region.astype(np.uint8),
        photo_mask.astype(np.uint8)
    ).astype(np.float32)

    # Apply Gaussian blur for softness
    blur_size = max(7, int(blur_radius * 2) | 1)
    fuzzy_shadow = cv2.GaussianBlur(shadow_region, (blur_size, blur_size), 0)

    # Scale by opacity
    fuzzy_shadow = fuzzy_shadow / 255.0 * opacity

    # Build full RGB shadow canvas (dark gray shadow color)
    shadow_rgb = np.zeros((ext_h, ext_w, 3), dtype=np.float32)
    shadow_rgb[:, :, 0] = fuzzy_shadow
    shadow_rgb[:, :, 1] = fuzzy_shadow
    shadow_rgb[:, :, 2] = fuzzy_shadow

    # Place photo on extended canvas (no alpha needed since it's opaque)
    photo_rgba = np.zeros((ext_h, ext_w, 4), dtype=np.uint8)
    photo_rgba[pad:pad + h, pad:pad + w] = img_rgba

    # Alpha for blending
    if img_rgba.shape[2] == 4:
        a = photo_rgba[:, :, 3:4] / 255.0
    else:
        a = np.ones((ext_h, ext_w, 1), dtype=np.float32)

    # Composite: shadow BG + photo FG
    result = shadow_rgb * (1 - a) + photo_rgba[:, :, :3].astype(np.float32) * a
    return np.clip(result, 0, 255).astype(np.uint8)


def add_screen_glare(img_rgba, num_circles=None):
    """Add specular glare using screen compositing.

    Simulates light reflecting off glossy photo surface.
    Screen blend: result = 1 - (1 - src) * (1 - dst)

    Each glare blob is an elliptical gradient with Gaussian falloff,
    slightly noise-perturbed for realism.

    Args:
        img_rgba: BGRA image
        num_circles: number of glare blobs (random 1-3 if None)

    Returns:
        BGRA image with screen-composited glare
    """
    h, w = img_rgba.shape[:2]
    img_f = img_rgba[:, :, :3].astype(np.float32) / 255.0

    if num_circles is None:
        num_circles = random.randint(1, 3)

    glare = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_circles):
        cx = random.uniform(w * 0.15, w * 0.85)
        cy = random.uniform(h * 0.15, h * 0.85)
        rx = random.uniform(w * 0.08, w * 0.20)
        ry = random.uniform(h * 0.08, h * 0.20)
        angle = random.uniform(0, math.pi)

        y, x = np.ogrid[:h, :w]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        xc, yc = x - cx, y - cy

        # Rotated ellipse
        xr = xc * cos_a - yc * sin_a
        yr = xc * sin_a + yc * cos_a
        ellipse = np.maximum(0, 1 - (xr / rx)**2 - (yr / ry)**2)

        # Gaussian blur for soft falloff
        ellipse = cv2.GaussianBlur(ellipse.astype(np.float32), (31, 31), 0)

        # Add slight noise to break perfect symmetry
        noise = np.random.randn(h, w) * 0.02
        ellipse = np.clip(ellipse + noise, 0, 1)

        glare = np.maximum(glare, ellipse)

    glare_strength = random.uniform(0.15, 0.4)
    glare = glare * glare_strength

    # Screen compositing: result = 1 - (1 - glare) * (1 - img_f)
    screen_result = 1.0 - (1.0 - glare[:, :, np.newaxis]) * (1.0 - img_f)

    # Blend via alpha
    if img_rgba.shape[2] == 4:
        alpha = img_rgba[:, :, 3:4] / 255.0
        blended = img_f * (1 - alpha) + screen_result * alpha
    else:
        blended = screen_result

    result = (np.clip(blended, 0, 1) * 255).astype(np.uint8)

    if img_rgba.shape[2] == 4:
        return np.dstack([result, img_rgba[:, :, 3]])
    return result


def apply_background_gradients(bg_arr, num_gradients=None):
    """Apply 1-3 gradient overlays to background (radial + linear).

    Real scene lighting has multiple gradient sources: overhead lights,
    window light, ambient fill. This simulates that with multiply-blended
    radial, horizontal, vertical, or diagonal gradients, each either
    lightening or darkening.

    Args:
        bg_arr: BGR image (uint8)
        num_gradients: number of gradient overlays (random 1-3 if None)

    Returns:
        BGR image with gradient overlays applied
    """
    h, w = bg_arr.shape[:2]
    result = bg_arr.astype(np.float32)

    if num_gradients is None:
        num_gradients = random.randint(1, 3)

    available = ['radial', 'horizontal', 'vertical', 'diagonal']
    types = random.sample(available, k=min(num_gradients, len(available)))

    for gtype in types:
        strength = random.uniform(0.08, 0.22)
        direction = random.choice([-1, 1])  # darkening (-1) or lightening (+1)

        if gtype == 'radial':
            cx = random.uniform(w * 0.25, w * 0.75)
            cy = random.uniform(h * 0.25, h * 0.75)
            max_d = math.sqrt(w**2 + h**2) / 2
            yi, xi = np.ogrid[:h, :w]
            dist = np.sqrt((xi - cx)**2 + (yi - cy)**2)
            grad_map = 1.0 - (dist / max_d) * strength * direction

        elif gtype == 'horizontal':
            x_coords = np.linspace(0, 1, w)
            hdir = random.choice(['left', 'right', 'center'])
            if hdir == 'left':
                grad_map = 1.0 - x_coords[np.newaxis, :] * strength * direction
            elif hdir == 'right':
                grad_map = 1.0 - (1.0 - x_coords)[np.newaxis, :] * strength * direction
            else:  # center_dark or center_light
                grad_map = 1.0 - np.abs(x_coords - 0.5)[np.newaxis, :] * 2 * strength * direction

        elif gtype == 'vertical':
            y_coords = np.linspace(0, 1, h)
            vdir = random.choice(['top', 'bottom', 'center'])
            if vdir == 'top':
                grad_map = 1.0 - y_coords[:, np.newaxis] * strength * direction
            elif vdir == 'bottom':
                grad_map = 1.0 - (1.0 - y_coords)[:, np.newaxis] * strength * direction
            else:
                grad_map = 1.0 - np.abs(y_coords - 0.5)[:, np.newaxis] * 2 * strength * direction

        else:  # diagonal
            xi = np.linspace(-1, 1, w)
            yi = np.linspace(-1, 1, h)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            grad_map = 1.0 - (xi_grid + yi_grid) / 2.0 * strength * direction

        result = result * grad_map[:, :, np.newaxis]

    return np.clip(result, 0, 255).astype(np.uint8)


def add_glare(img_rgba, strength=0.3, size_variation=0.3):
    """Add specular glare to photo surface."""
    h, w = img_rgba.shape[:2]
    
    glare = np.zeros((h, w), dtype=np.float32)
    
    num_flares = random.randint(1, 3)
    
    for _ in range(num_flares):
        cx = random.uniform(w * 0.2, w * 0.8)
        cy = random.uniform(h * 0.2, h * 0.8)
        rx = random.uniform(w * 0.05, w * 0.15) * (1 + size_variation * random.uniform(-1, 1))
        ry = random.uniform(h * 0.05, h * 0.15) * (1 + size_variation * random.uniform(-1, 1))
        angle = random.uniform(0, math.pi)
        
        y, x = np.ogrid[:h, :w]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        xc = x - cx
        yc = y - cy
        
        xr = xc * cos_a - yc * sin_a
        yr = xc * sin_a + yc * cos_a
        
        ellipse_val = (xr / rx)**2 + (yr / ry)**2
        flare = np.maximum(0, 1 - ellipse_val)
        flare = cv2.GaussianBlur(flare.astype(np.float32), (31, 31), 0)
        
        glare = np.maximum(glare, flare * strength)
    
    glare_rgb = np.stack([glare * 255] * 3, axis=-1).astype(np.uint8)
    alpha = img_rgba[:, :, 3:4] / 255.0
    img_rgb = img_rgba[:, :, :3].astype(np.float32)
    img_rgb = np.minimum(255, img_rgb + glare_rgb.astype(np.float32) * alpha)
    
    return np.dstack([img_rgb.astype(np.uint8), img_rgba[:, :, 3]])


class BackgroundGenerator:
    def __init__(self):
        self.types = ['wood', 'marble', 'solid', 'fabric', 'laminate']
    
    def generate(self, w, h):
        weights = [0.20, 0.15, 0.35, 0.15, 0.15]
        bg_type = random.choices(self.types, weights=weights)[0]
        
        if bg_type == 'wood':
            return self._generate_wood(w, h)
        elif bg_type == 'marble':
            return self._generate_marble(w, h)
        elif bg_type == 'solid':
            return self._generate_solid(w, h)
        elif bg_type == 'fabric':
            return self._generate_fabric(w, h)
        else:
            return self._generate_laminate(w, h)
    
    def _generate_wood(self, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        base_colors = [
            (139, 90, 43),
            (160, 110, 65),
            (120, 75, 35),
            (180, 130, 80),
        ]
        base = random.choice(base_colors)
        img[:, :] = base
        
        num_grains = random.randint(15, 25)
        for _ in range(num_grains):
            y = random.randint(0, h - 1)
            thickness = random.randint(2, 8)
            intensity = random.randint(-30, 30)
            color_shift = random.randint(-15, 15)
            
            offset = random.uniform(-0.02, 0.02)
            for i in range(max(0, y - thickness), min(h, y + thickness)):
                wavy_x = int(math.sin(i * offset) * 20)
                for j in range(max(0, wavy_x), min(w, wavy_x + w)):
                    blend = 1 - abs(i - y) / thickness
                    img[i, j] = np.clip(img[i, j] + (intensity + color_shift) * blend, 0, 255)
        
        noise = np.random.normal(0, 8, (h, w, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return cv2.GaussianBlur(img, (3, 3), 0)
    
    def _generate_marble(self, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        base_colors = [
            [(240, 240, 245), (200, 200, 210)],
            [(245, 240, 235), (210, 180, 160)],
            [(230, 235, 240), (180, 190, 200)],
        ]
        base, vein = random.choice(base_colors)
        
        t = np.linspace(0, 4 * math.pi, w)
        offset = np.random.uniform(0, 2 * math.pi)
        
        for i in range(h):
            for j in range(w):
                wave = math.sin(t[j] + i * 0.02 + offset)
                vein_intensity = (wave + 1) / 2
                
                noise = math.sin(i * 0.1 + j * 0.15) * 0.3 + math.sin(i * 0.05 - j * 0.08) * 0.3
                vein_intensity = np.clip(vein_intensity + noise, 0, 1)
                
                color = tuple(int(b * (1 - vein_intensity) + v * vein_intensity) for b, v in zip(base, vein))
                img[i, j] = color
        
        noise = np.random.normal(0, 5, (h, w, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return cv2.GaussianBlur(img, (5, 5), 0)
    
    def _generate_solid(self, w, h):
        """Generate solid-color background with wide color palette.

        Colors are drawn from three tiers: light, dark, and mid-range,
        with noise for surface texture. This replaces the mid-gray-only
        palette with backgrounds spanning nearly black to nearly white.
        """
        # Three-tier color palette for maximum variety
        LIGHT_COLORS = [
            (240, 242, 245), (235, 240, 242), (250, 248, 245),
            (245, 245, 248), (255, 255, 252), (240, 238, 235),
            (248, 250, 252), (252, 250, 248), (238, 242, 245),
        ]
        DARK_COLORS = [
            (15, 15, 18), (20, 20, 25), (12, 15, 20),
            (25, 22, 28), (18, 18, 22), (10, 12, 15),
            (22, 20, 24), (28, 25, 30), (14, 16, 20),
        ]
        MID_COLORS = [
            (165, 160, 155), (140, 145, 138), (180, 175, 170),
            (155, 150, 148), (170, 168, 162), (148, 145, 140),
            (160, 155, 150), (175, 170, 165), (145, 142, 138),
        ]

        # Weighted selection: 30% light, 30% dark, 40% mid
        tier = random.choices(['light', 'dark', 'mid'], weights=[0.30, 0.30, 0.40])[0]
        if tier == 'light':
            color = random.choice(LIGHT_COLORS)
        elif tier == 'dark':
            color = random.choice(DARK_COLORS)
        else:
            color = random.choice(MID_COLORS)

        img = np.ones((h, w, 3), dtype=np.uint8) * np.array(color)

        # Variable noise: sigma from 3 (clean) to 15 (grainy)
        noise_sigma = random.uniform(3, 15)
        noise = np.random.normal(0, noise_sigma, (h, w, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        # Light blur for very high noise
        if noise_sigma > 12:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        return img
    
    def _generate_fabric(self, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        base_colors = [
            (120, 110, 100),
            (100, 95, 90),
            (140, 130, 120),
            (110, 105, 100),
        ]
        color = random.choice(base_colors)
        img[:, :] = color
        
        weave_scale = random.choice([3, 4, 5])
        for i in range(0, h, weave_scale):
            for j in range(0, w, weave_scale):
                shade = random.randint(-5, 5)
                img[i:min(i+weave_scale, h), j:min(j+weave_scale, w)] = np.clip(
                    np.array(color) + shade, 0, 255)
        
        noise = np.random.normal(0, 6, (h, w, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return cv2.GaussianBlur(img, (3, 3), 0)
    
    def _generate_laminate(self, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        base_colors = [
            (180, 160, 140),
            (160, 145, 125),
            (190, 175, 155),
            (170, 155, 135),
        ]
        plank_colors = random.choice([
            [(180, 165, 145), (175, 158, 138)],
            [(165, 150, 130), (170, 155, 135)],
            [(185, 170, 150), (180, 162, 142)],
        ])
        
        plank_h = random.randint(15, 25)
        plank_idx = 0
        
        for i in range(0, h, plank_h):
            color = plank_colors[plank_idx % len(plank_colors)]
            plank_idx += 1
            
            plank_width = random.randint(80, 150)
            
            for j in range(0, w, plank_width):
                shade = random.randint(-8, 8)
                this_color = tuple(max(0, min(255, c + shade)) for c in color)
                
                end_i = min(i + plank_h, h)
                end_j = min(j + plank_width, w)
                img[i:end_i, j:end_j] = this_color
            
            gap_shade = random.randint(-15, -5)
            gap_end = min(i + 2, h)
            img[i:gap_end, :] = tuple(max(0, min(255, c + gap_shade)) for c in color)
        
        noise = np.random.normal(0, 4, (h, w, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return cv2.GaussianBlur(img, (3, 3), 0)


def apply_luma_gradient(bg_arr):
    """Apply multi-gradient overlays to background.

    Delegates to apply_background_gradients() which applies 1-3 gradient
    overlays (radial, horizontal, vertical, diagonal), each either
    lightening or darkening. This replaces the single radial gradient.
    """
    return apply_background_gradients(bg_arr)



def process_photo(img_path, min_size, max_size):
    """Load and process a photo (no perspective warp yet).
    
    Perspective warp will be applied during compositing.
    
    Returns:
        photo: BGRA image (rectangular, no warp)
        corners: Corners of the rectangular photo (for packing)
        shadow_mask: Shadow mask (or None)
        original_img: Original BGR/BGRA image (for compositing)
        img_size: (width, height) of the resized image
    """
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None, None, None, None
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        if img.shape[0] < 100 or img.shape[1] < 100:
            return None, None, None, None, None
        
        # Keep original for later compositing
        original_img = img.copy()
        
        target_size = random.uniform(min_size, max_size)
        scale = target_size / max(img.shape[:2])
        
        new_w = int(img.shape[1] * scale)
        new_h = int(img.shape[0] * scale)
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        original_img = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Rectangular corners (for packing)
        corners = np.array([
            [0.0, 0.0],
            [new_w - 1.0, 0.0],
            [new_w - 1.0, new_h - 1.0],
            [0.0, new_h - 1.0]
        ], dtype=np.float32)
        
        # Apply color manipulation (brightness, contrast, saturation, gamma)
        # This runs on the ORIGINAL photo before any effects are added.
        # 80% probability to vary photo appearance.
        if random.random() < 0.80:
            original_img = manipulate_photo(original_img)
            img = manipulate_photo(img)

        # Add fuzzy dispersed drop shadow
        # Parameters: offset (2-5px), blur_radius (12-25), opacity (0.3-0.5)
        shadow_mask = None
        if random.random() < 0.85:
            blur_radius = random.randint(12, 25)
            offset_x = random.randint(2, 5)
            offset_y = random.randint(2, 5)
            opacity = random.uniform(0.3, 0.5)
            img = add_fuzzy_drop_shadow(
                img, corners,
                offset=(offset_x, offset_y),
                blur_radius=blur_radius,
                opacity=opacity
            )

        # Add variable blur
        if random.random() < 0.3:
            blur_min = random.uniform(0, 2)
            blur_max = blur_min + random.uniform(2, 6)
            edge_focus = random.uniform(0.3, 0.7)
            img = add_variable_blur(img, blur_min, blur_max, edge_focus)

        # Add screen-composited glare (replaces additive glare)
        if random.random() < 0.60:
            img = add_screen_glare(img)

        return img, corners, shadow_mask, original_img, (new_w, new_h)

    except Exception as e:
        return None, None, None, None, None


def convert_bgra_to_rgba(img):
    """Convert BGRA to RGBA for PIL compatibility."""
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def polygons_overlap(poly1, poly2, margin=0):
    """Check if two polygons overlap using SAT."""
    gap_half = margin / 2
    
    min1_x, max1_x = poly1[:, 0].min(), poly1[:, 0].max()
    min1_y, max1_y = poly1[:, 1].min(), poly1[:, 1].max()
    min2_x, max2_x = poly2[:, 0].min(), poly2[:, 0].max()
    min2_y, max2_y = poly2[:, 1].min(), poly2[:, 1].max()
    
    if max1_x + gap_half < min2_x or min1_x - gap_half > max2_x:
        return False
    if max1_y + gap_half < min2_y or min1_y - gap_half > max2_y:
        return False
    
    for poly in [poly1, poly2]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            if np.abs(edge).sum() < 0.001:
                continue
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            proj1 = poly1 @ normal
            proj2 = poly2 @ normal
            if proj1.max() + gap_half < proj2.min() or proj2.max() + gap_half < proj1.min():
                return False
    
    return True


def composite_overlay(bg, overlay, pos):
    """Composite overlay onto background using alpha channel.
    
    bg: BGR image (OpenCV default)
    overlay: BGRA image
    """
    x, y = pos
    
    h, w = overlay.shape[:2]
    
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(bg.shape[1], int(x) + w), min(bg.shape[0], int(y) + h)
    
    ov_x1 = x1 - int(x)
    ov_y1 = y1 - int(y)
    ov_x2 = ov_x1 + (x2 - x1)
    ov_y2 = ov_y1 + (y2 - y1)
    
    if ov_x2 <= ov_x1 or ov_y2 <= ov_y1:
        return bg
    
    overlay_slice = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    
    if overlay_slice.shape[2] == 4:
        alpha = overlay_slice[:, :, 3:4] / 255.0
        bg_slice = bg[y1:y2, x1:x2].astype(np.float32)
        # overlay is BGRA, so [:3] gives B, G, R - same as bg
        overlay_rgb = overlay_slice[:, :, :3].astype(np.float32)
        result = bg_slice * (1 - alpha) + overlay_rgb * alpha
        bg[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)
    else:
        # No alpha, treat as BGR
        bg[y1:y2, x1:x2] = overlay_slice
    
    return bg


    return warped_corners


def generate_camera_perspective_corners(placed_corners):
    """
    Generate perspective-warped destination corners from placed rectangular corners.
    
    Simulates camera viewing the photo from above at an angle.
    Applies subtle rotation and tiny scale variation to create the illusion
    of perspective without causing quadrilaterals to intersect.
    
    Args:
        placed_corners: 4x2 array of rectangular corners [TL, TR, BR, BL]
    
    Returns:
        warped_corners: 4x2 array of warped quadrilateral corners
    """
    # Calculate bounds and center
    orig_min_x = placed_corners[:, 0].min()
    orig_max_x = placed_corners[:, 0].max()
    orig_min_y = placed_corners[:, 1].min()
    orig_max_y = placed_corners[:, 1].max()
    
    cx = (orig_min_x + orig_max_x) / 2
    cy = (orig_min_y + orig_max_y) / 2
    width = orig_max_x - orig_min_x
    height = orig_max_y - orig_min_y
    
    # Subtle rotation (-2 to +2 degrees)
    angle = random.uniform(-0.035, 0.035)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    # Apply rotation around center
    warped = placed_corners.copy()
    for i in range(4):
        dx = warped[i, 0] - cx
        dy = warped[i, 1] - cy
        warped[i, 0] = cx + dx * cos_a - dy * sin_a
        warped[i, 1] = cy + dx * sin_a + dy * cos_a
    
    # Tiny scale variation (0.98-1.02) - simulates slight distance change
    scale = random.uniform(0.985, 1.015)
    warped[:, 0] = cx + (warped[:, 0] - cx) * scale
    warped[:, 1] = cy + (warped[:, 1] - cy) * scale
    
    # Very subtle corner nudge (1-3 pixels) to simulate keystone
    # This is so subtle it won't cause intersections
    nudge = random.uniform(1, 3)
    direction = random.choice(['tl', 'tr', 'br', 'bl'])
    corner_idx = {'tl': 0, 'tr': 1, 'br': 2, 'bl': 3}[direction]
    axis = random.choice(['x', 'y'])
    sign = random.choice([-1, 1])
    
    if axis == 'x':
        warped[corner_idx, 0] += sign * nudge
    else:
        warped[corner_idx, 1] += sign * nudge
    
    return warped


def apply_perspective_to_placements(placed, infos):
    """
    Apply camera perspective warping to all placed photos.
    
    Updates placed['corners'] and info['keypoints'] with warped corners.
    
    Args:
        placed: List of placed photo dicts with 'corners' key
        infos: List of info dicts with 'keypoints' key
    
    Returns:
        placed, infos (modified in place)
    """
    for i, photo_info in enumerate(placed):
        placed_corners = photo_info['corners']
        
        # Generate perspective-warped corners
        warped_corners = generate_camera_perspective_corners(placed_corners)
        
        # Store warped corners
        placed[i]['corners'] = warped_corners
        
        # Update keypoints in infos
        if i < len(infos):
            infos[i]['keypoints'] = [(warped_corners[j][0], warped_corners[j][1]) for j in range(4)]
            # Recalculate center
            kps = infos[i]['keypoints']
            infos[i]['x_center'] = sum(k[0] for k in kps) / 4
            infos[i]['y_center'] = sum(k[1] for k in kps) / 4
    
    return placed, infos


def apply_global_perspective_transform(canvas, placed_info_list, output_w, output_h, canvas_scale):
    """
    Apply global perspective transform to simulate camera viewing scene at an angle.
    
    Uses proper perspective projection (homography) to create a keystone distortion
    that simulates viewing the scene from an elevated camera position.
    
    Args:
        canvas: BGR image (high-res flat canvas)
        placed_info_list: List of dicts with 'corners' for each photo
        output_w, output_h: Final output dimensions
        canvas_scale: Scale factor used (e.g., 3 for 3x final size)
    
    Returns:
        transformed: Transformed BGR image (output size)
        transformed_corners: List of transformed corner arrays (in output coords)
    """
    h, w = canvas.shape[:2]
    
    # Define source rectangle (full canvas)
    src_corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)
    
    # Define camera position relative to scene
    # Camera positioned to look at the scene from above
    # Wider range for stronger perspective distortion
    camera_x = random.uniform(0.35, 0.65)  # Camera X position (farther from center = more tilt)
    camera_y = random.uniform(0.30, 0.70)  # Camera Y position (farther from center = more tilt)

    # Calculate keystone distortion based on camera position
    # Range: 4-10% of dimension — clearly trapezoidal, not just a slight tilt
    perspective_strength = random.uniform(0.04, 0.10)
    keystone_horizontal = w * perspective_strength * (camera_x - 0.5)
    keystone_vertical = h * perspective_strength * (camera_y - 0.5)

    # Destination trapezoid corners with stronger distortion
    # vert_margin gives breathing room so content doesn't clip at edges
    vert_margin = h * 0.03
    dst_corners = np.array([
        [keystone_horizontal, vert_margin],                          # TL - only 3% from top
        [w - 1 + keystone_horizontal, vert_margin],                 # TR - only 3% from top
        [w - 1 - keystone_horizontal * 0.5, h - 1 - vert_margin],  # BR - only 3% from bottom
        [keystone_horizontal * 0.5, h - 1 - vert_margin]           # BL - only 3% from bottom
    ], dtype=np.float32)
    
    # Add wider rotation for more natural camera tilt variation
    angle = random.uniform(-8, 8)
    center = np.array([w / 2, h / 2])
    rot_rad = np.radians(angle)
    cos_a, sin_a = np.cos(rot_rad), np.sin(rot_rad)
    
    # Rotate destination corners around center
    for i in range(4):
        dx = dst_corners[i, 0] - center[0]
        dy = dst_corners[i, 1] - center[1]
        dst_corners[i, 0] = center[0] + dx * cos_a - dy * sin_a
        dst_corners[i, 1] = center[1] + dx * sin_a + dy * cos_a
    
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    
    # Apply perspective warp to canvas
    warped = cv2.warpPerspective(canvas, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))

    # Center crop: take the middle portion and resize to output
    # This guarantees no black edges in the final output
    center_y = h // 2
    center_x = w // 2

    crop_margin = max(w, h) // 10  # 10% margin
    crop_h = h - 2 * crop_margin
    crop_w = w - 2 * crop_margin

    y_min = center_y - crop_h // 2
    y_max = y_min + crop_h
    x_min = center_x - crop_w // 2
    x_max = x_min + crop_w

    if y_min < 0: y_min = 0
    if y_max > h: y_max = h
    if x_min < 0: x_min = 0
    if x_max > w: x_max = w

    cropped = warped[y_min:y_max, x_min:x_max]
    crop_h, crop_w = cropped.shape[:2]

    # Black border check: flag if >0.1% pixels are fully black
    # Caller retries with larger canvas if this is too high
    mask_black = (cropped[:, :, 0] == 0) & (cropped[:, :, 1] == 0) & (cropped[:, :, 2] == 0)
    black_pct = mask_black.sum() / (crop_h * crop_w)

    # Verify corners stayed inside the crop after perspective transform.
    # If corners are outside the crop, the perspective warp moved them there,
    # and scaling them back makes them look nearly rectangular (no perspective).
    # Flag for retry with larger canvas if any corner is clipped.
    corners_clipped = 0
    for info in placed_info_list:
        corners = info['corners'].copy()
        ones = np.ones((4, 1), dtype=np.float64)
        corners_h = np.hstack([corners.astype(np.float64), ones])
        warped_pts = M @ corners_h.T
        warped_pts = warped_pts[:2, :] / warped_pts[2, :]
        warped_pts = warped_pts.T

        # Check if ANY corner is outside the crop region
        outside = (
            (warped_pts[:, 0] < x_min).any() or
            (warped_pts[:, 0] > x_max).any() or
            (warped_pts[:, 1] < y_min).any() or
            (warped_pts[:, 1] > y_max).any()
        )
        if outside:
            corners_clipped += 1

    # If any photo had clipped corners, boost black_pct to trigger retry
    if corners_clipped > 0:
        black_pct = max(black_pct, 0.05)

    # Resize to output dimensions
    transformed = cv2.resize(cropped, (output_w, output_h), interpolation=cv2.INTER_AREA)
    
    # Calculate scale factors
    scale_x = output_w / crop_w
    scale_y = output_h / crop_h
    
    # Transform all corner keypoints using the perspective matrix
    transformed_corners = []
    for info in placed_info_list:
        corners = info['corners'].copy()
        
        # Apply perspective matrix to corners
        ones = np.ones((4, 1), dtype=np.float64)
        corners_h = np.hstack([corners.astype(np.float64), ones])
        warped_pts = M @ corners_h.T
        
        # Convert from homogeneous coordinates
        warped_pts = warped_pts[:2, :] / warped_pts[2, :]
        warped_pts = warped_pts.T
        
        # Apply crop offset and scale to output
        warped_pts[:, 0] = (warped_pts[:, 0] - x_min) * scale_x
        warped_pts[:, 1] = (warped_pts[:, 1] - y_min) * scale_y

        # NOTE: We intentionally do NOT clamp corners to safety margins.
        # Clamping corners that fall near the crop boundary makes photos look
        # nearly rectangular (wrong). If perspective places corners outside
        # the crop, the entire image is flagged for retry with a larger canvas.
        # This preserves the true trapezoidal shape of each photo.
        transformed_corners.append(warped_pts.astype(np.float32))
    
    return transformed, transformed_corners, black_pct


def rotate_corners(corners, angle_deg, cx, cy):
    """Rotate corners around origin, then translate to (cx, cy).
    
    This is the correct approach: first rotate around (0,0), then translate.
    The corners should be centered at origin before calling this function.
    
    Args:
        corners: 4x2 array of corners (centered at origin)
        angle_deg: Rotation angle in degrees
        cx, cy: Final center position to translate to
    
    Returns:
        Rotated and translated corners
    """
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    rotated = corners.copy().astype(np.float64)
    for i in range(4):
        # Rotate around origin (0,0)
        x, y = rotated[i]
        rotated[i, 0] = x * cos_a - y * sin_a
        rotated[i, 1] = x * sin_a + y * cos_a
    
    # Translate to final position
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    
    return rotated.astype(np.float32)


def rotate_and_place_photo(bbox_corners, center_x, center_y, angle_deg):
    """
    Rotate rectangular corners around their center, then translate to position.
    
    Args:
        bbox_corners: Original rectangular corners (4x2 array, centered at origin-ish)
        center_x, center_y: Final center position
        angle_deg: Rotation angle in degrees
    
    Returns:
        placed_corners: Rotated and translated corners
    """
    # Get center of original bbox
    orig_cx = bbox_corners[:, 0].mean()
    orig_cy = bbox_corners[:, 1].mean()
    
    # Translate to origin
    centered = bbox_corners.copy() - np.array([orig_cx, orig_cy])
    
    # Rotate around origin
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    rotated = centered @ rot_mat.T
    
    # Translate to final position
    placed = rotated + np.array([center_x, center_y])
    
    return placed.astype(np.float32)


def get_rotated_bbox_corners(w, h, angle_deg, center_x, center_y):
    """Create rotated rectangular corners for a photo."""
    # Original corners centered at (0,0)
    orig_corners = np.array([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2]
    ], dtype=np.float32)
    
    return rotate_and_place_photo(orig_corners, center_x, center_y, angle_deg)


def get_bounding_box_from_corners(corners):
    """Get bounding box from corners (handles any quadrilateral)."""
    min_x = corners[:, 0].min()
    max_x = corners[:, 0].max()
    min_y = corners[:, 1].min()
    max_y = corners[:, 1].max()
    return {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'w': max_x - min_x,
        'h': max_y - min_y,
        'corners': corners
    }


def composite_rotated_photo(bg, original_photo, placed_corners, with_effects=True):
    """Composite a photo onto background, properly handling rotation.
    
    This handles photos that are rotated arbitrarily by applying a rotation
    transformation around the photo's center point.
    
    Args:
        bg: BGR image (canvas)
        original_photo: BGRA or BGR image (rectangular source)
        placed_corners: 4x2 array of rotated corner coordinates on the canvas
        with_effects: Whether to apply shadow, blur, glare effects
    
    Returns:
        bg: Composited background image
    """
    orig_h, orig_w = original_photo.shape[:2]
    
    # Calculate the center of the placed corners
    center_x = placed_corners[:, 0].mean()
    center_y = placed_corners[:, 1].mean()
    
    # Calculate the rotation angle from the placed corners
    # Use the top edge vector (from TL to TR) to determine rotation
    tl, tr, br, bl = placed_corners
    
    # For a rectangular photo, the top edge vector should be (width, 0) - pointing right
    src_top_vec = np.array([orig_w, 0.0])
    
    # For the placed photo, the top edge is from TL to TR
    dst_top_vec = np.array([tr[0] - tl[0], tr[1] - tl[1]])
    
    # Calculate rotation angle from vectors
    src_angle = np.arctan2(src_top_vec[1], src_top_vec[0])  # 0 radians
    dst_angle = np.arctan2(dst_top_vec[1], dst_top_vec[0])
    rotation_angle = np.degrees(dst_angle - src_angle)
    
    # Create rotation matrix
    M_rot = cv2.getRotationMatrix2D((orig_w/2, orig_h/2), rotation_angle, 1.0)
    
    # Add translation to center at placed position
    M_rot[0, 2] += center_x - orig_w/2
    M_rot[1, 2] += center_y - orig_h/2
    
    # Get output size to cover the rotated photo
    # Calculate bounding box of rotated image
    corners_rotated = np.array([
        [0, 0],
        [orig_w - 1, 0],
        [orig_w - 1, orig_h - 1],
        [0, orig_h - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    transformed_corners = cv2.transform(corners_rotated, M_rot).reshape(-1, 2)
    
    min_x = int(np.floor(transformed_corners[:, 0].min()))
    min_y = int(np.floor(transformed_corners[:, 1].min()))
    max_x = int(np.ceil(transformed_corners[:, 0].max()))
    max_y = int(np.ceil(transformed_corners[:, 1].max()))
    
    out_w = max(1, max_x - min_x + 1)
    out_h = max(1, max_y - min_y + 1)
    
    # Adjust matrix to shift output to origin
    M_out = M_rot.copy()
    M_out[0, 2] -= min_x
    M_out[1, 2] -= min_y
    
    # Warp the photo
    warped = cv2.warpAffine(original_photo, M_out, (out_w, out_h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    
    # Create polygon mask for proper compositing (ignores warped image alpha)
    # This ensures no edge artifacts from the affine transform
    has_alpha = original_photo.shape[2] == 4
    
    # Get the actual rotated corners in output coordinates
    actual_corners = transformed_corners.copy()
    actual_corners[:, 0] -= min_x
    actual_corners[:, 1] -= min_y
    
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    quad_pts = actual_corners.astype(np.int32)
    cv2.fillPoly(mask, [quad_pts], 255)
    
    # Use polygon mask for compositing (ignores any alpha in original photo)
    # This ensures no edge artifacts from affine transform
    mask = mask.astype(np.float32) / 255.0
    
    # Apply effects to the warped photo if requested
    if with_effects:
        # Add shadow
        if random.random() < 0.85:
            spread = random.randint(20, 35)
            offset = random.randint(1, 4)
            opacity = random.uniform(0.4, 0.6)
            warped, _, _ = add_drop_shadow(
                warped, 
                np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32),
                offset=offset, spread=spread, opacity=opacity
            )
            # Recalculate mask after shadow (which may have added alpha)
            if warped.shape[2] == 4:
                mask = (warped[:, :, 3] > 0).astype(np.float32) / 255.0
        
        # Add blur
        if random.random() < 0.3:
            blur_min = random.uniform(0, 2)
            blur_max = blur_min + random.uniform(2, 6)
            edge_focus = random.uniform(0.3, 0.7)
            warped = add_variable_blur(warped, blur_min, blur_max, edge_focus)
        
        # Add glare
        if random.random() < 0.4:
            warped = add_glare(warped)
    
    # Composite onto background
    y_off = min_y
    x_off = min_x
    
    # Ensure we're within bounds
    if y_off < 0:
        mask = mask[-y_off:, :]
        warped = warped[-y_off:, :] if warped.shape[0] > -y_off else warped
        y_off = 0
    if x_off < 0:
        mask = mask[:, -x_off:]
        warped = warped[:, -x_off:] if warped.shape[1] > -x_off else warped
        x_off = 0
    
    # Get intersection region
    bg_h, bg_w = bg.shape[:2]
    y_end = min(y_off + warped.shape[0], bg_h)
    x_end = min(x_off + warped.shape[1], bg_w)
    mask_h = y_end - y_off
    mask_w = x_end - x_off
    
    if mask_h <= 0 or mask_w <= 0:
        return bg  # Nothing to composite
    
    mask_region = mask[:mask_h, :mask_w]
    warped_region = warped[:mask_h, :mask_w]
    
    if warped_region.shape[2] == 4:
        alpha = mask_region[:, :, np.newaxis]
    else:
        alpha = mask_region[:, :, np.newaxis]
    
    # Blend
    for c in range(3):
        bg[y_off:y_end, x_off:x_end, c] = (
            bg[y_off:y_end, x_off:x_end, c] * (1 - alpha[:, :, 0]) +
            warped_region[:, :, c].astype(np.float32) * alpha[:, :, 0]
        ).astype(np.uint8)
    
    return bg


def composite_scaled_photo(bg, original_photo, placed_corners, with_effects=True, apply_perspective=True):
    """Composite a photo onto background using perspective warp to placed corners.
    
    This warps the original rectangular photo to the placed position, ensuring
    corners match exactly.
    
    bg: BGR image (canvas)
    original_photo: BGRA or BGR image (rectangular)
    placed_corners: 4x2 array of corner coordinates on the canvas (TL, TR, BR, BL)
    with_effects: Whether to apply shadow, blur, glare effects
    apply_perspective: Whether to apply per-photo perspective warp (for local angle effect)
    """
    orig_h, orig_w = original_photo.shape[:2]
    has_alpha = original_photo.shape[2] == 4
    
    # Source corners (rectangular image)
    src_pts = np.array([
        [0.0, 0.0],
        [orig_w - 1.0, 0.0],
        [orig_w - 1.0, orig_h - 1.0],
        [0.0, orig_h - 1.0]
    ], dtype=np.float32)
    
    # Destination corners - either warped or rectangular
    if apply_perspective and with_effects:
        # Generate perspective-warped destination corners (subtle effect)
        dst_pts = generate_camera_perspective_corners(placed_corners)
    else:
        # Use placed corners as-is (rectangular, rotation handled by placement)
        dst_pts = placed_corners.copy()
    
    # Compute forward perspective transform from source to destination
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Get output bounding box (canvas area to cover destination)
    min_x = int(np.floor(dst_pts[:, 0].min()))
    min_y = int(np.floor(dst_pts[:, 1].min()))
    max_x = int(np.ceil(dst_pts[:, 0].max()))
    max_y = int(np.ceil(dst_pts[:, 1].max()))
    
    # Output size should cover the destination region
    out_w = max(1, max_x + 1)
    out_h = max(1, max_y + 1)
    
    # Warp photo to the placed position
    warped = cv2.warpPerspective(original_photo, M, (out_w, out_h), 
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=0)
    
    # Apply effects to the warped photo if requested
    if with_effects:
        # Add shadow
        if random.random() < 0.85:
            spread = random.randint(20, 35)
            offset = random.randint(1, 4)
            opacity = random.uniform(0.4, 0.6)
            corners = np.array([
                [0.0, 0.0],
                [out_w - 1.0, 0.0],
                [out_w - 1.0, out_h - 1.0],
                [0.0, out_h - 1.0]
            ], dtype=np.float32)
            warped, _, shadow_mask = add_drop_shadow(
                warped, corners, offset=offset, spread=spread, opacity=opacity
            )
        
        # Add blur
        if random.random() < 0.3:
            blur_min = random.uniform(0, 2)
            blur_max = blur_min + random.uniform(2, 6)
            edge_focus = random.uniform(0.3, 0.7)
            warped = add_variable_blur(warped, blur_min, blur_max, edge_focus)
        
        # Add glare
        if random.random() < 0.4:
            warped = add_glare(warped)
    
    # Composite onto canvas
    canvas_h, canvas_w = bg.shape[:2]
    
    # Clamp destination bounds to canvas
    # min_x, min_y are from the original warped_corners (before shifting)
    # but warped image starts at (0,0) due to the shift in transform
    x1 = max(0, min_x)
    y1 = max(0, min_y)
    x2 = min(canvas_w, max_x + 1)
    y2 = min(canvas_h, max_y + 1)
    
    # The warped image starts at (0,0), and its content covers the area
    # from (0,0) to (out_w-1, out_h-1)
    # We need to copy the portion that falls within [y1:y2, x1:x2] on the canvas
    warp_y1 = 0
    warp_x1 = 0
    warp_y2 = min(out_h, y2 - y1)  # How much of warped to use
    warp_x2 = min(out_w, x2 - x1)
    
    # Only composite if there's valid content
    if warp_x2 > 0 and warp_y2 > 0 and y2 > y1 and x2 > x1:
        warp_slice = warped[warp_y1:warp_y2, warp_x1:warp_x2]
        canvas_slice = bg[y1:y1 + warp_slice.shape[0], x1:x1 + warp_slice.shape[1]]
        
        if has_alpha:
            alpha = warp_slice[:, :, 3:4] / 255.0
            canvas_slice_float = canvas_slice.astype(np.float32)
            overlay_rgb = warp_slice[:, :, :3].astype(np.float32)
            result = canvas_slice_float * (1 - alpha) + overlay_rgb * alpha
            bg[y1:y1 + warp_slice.shape[0], x1:x1 + warp_slice.shape[1]] = np.clip(result, 0, 255).astype(np.uint8)
        else:
            bg[y1:y1 + warp_slice.shape[0], x1:x1 + warp_slice.shape[1]] = warp_slice
    
    return bg


    return bg


def composite_warped_photo(bg, original_photo, warped_corners, with_effects=True):
    """Composite a photo warped to any 4-point quadrilateral onto the canvas.

    This is the key function for per-photo keystone warping. It takes a
    rectangular source image and applies a perspective warp so that its
    corners land exactly at the 4 arbitrary positions specified by warped_corners.
    The result is a visible trapezoid in the scene.

    Unlike composite_rotated_photo() which requires the corners to be
    derived from rotation, this function works with any quadrilateral.

    Args:
        bg: BGR image (canvas)
        original_photo: BGRA or BGR image (rectangular source)
        warped_corners: 4x2 array of destination corner coordinates (TL, TR, BR, BL)
        with_effects: Whether to apply shadow/blur/glare effects

    Returns:
        bg: Composited background image
    """
    orig_h, orig_w = original_photo.shape[:2]
    has_alpha = original_photo.shape[2] == 4

    # Source corners (rectangular image corners)
    src_pts = np.array([
        [0.0, 0.0],
        [orig_w - 1.0, 0.0],
        [orig_w - 1.0, orig_h - 1.0],
        [0.0, orig_h - 1.0]
    ], dtype=np.float32)

    # Destination corners (the trapezoid positions)
    dst_pts = warped_corners.astype(np.float32)

    # Compute perspective transform: src -> dst
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Get bounding box of destination quadrilateral to determine output size
    min_x = int(np.floor(dst_pts[:, 0].min()))
    min_y = int(np.floor(dst_pts[:, 1].min()))
    max_x = int(np.ceil(dst_pts[:, 0].max()))
    max_y = int(np.ceil(dst_pts[:, 1].max()))

    out_w = max(1, max_x - min_x + 1)
    out_h = max(1, max_y - min_y + 1)

    # Warp the source image to fill the output area
    warped = cv2.warpPerspective(original_photo, M, (out_w, out_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)

    # Fix any black pixels from perspective transform artifacts using inpainting
    # This handles cases where the warp creates holes/voids in the image
    if warped.shape[2] >= 3:
        black_mask = (warped[:, :, 0] == 0) & (warped[:, :, 1] == 0) & (warped[:, :, 2] == 0)
        if black_mask.any():
            inpaint_mask = black_mask.astype(np.uint8) * 255
            # Dilate mask slightly to include edge artifacts
            inpaint_mask = cv2.dilate(inpaint_mask, np.ones((3, 3), np.uint8), iterations=1)
            for c in range(min(3, warped.shape[2])):
                warped[:, :, c] = cv2.inpaint(warped[:, :, c], inpaint_mask, 3, cv2.INPAINT_TELEA)
            # Also fix alpha channel if present
            if warped.shape[2] == 4:
                alpha_black = warped[:, :, 3] == 0
                if alpha_black.any():
                    warped[:, :, 3] = np.where(alpha_black, 255, warped[:, :, 3])

    # Create polygon mask for the warped region (always opaque within the quad)
    # This ensures no edge artifacts from perspective warp
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    quad_pts = dst_pts.copy()
    quad_pts[:, 0] -= min_x  # Offset to output coordinates
    quad_pts[:, 1] -= min_y
    quad_pts = quad_pts.astype(np.int32)
    cv2.fillPoly(mask, [quad_pts], 255)

    # Apply effects if requested
    if with_effects:
        # Shadow uses the full warped image dimensions
        if random.random() < 0.85:
            spread = random.randint(12, 25)
            offset_x = random.randint(2, 5)
            offset_y = random.randint(2, 5)
            opacity = random.uniform(0.3, 0.5)
            corners = np.array([
                [0.0, 0.0],
                [out_w - 1.0, 0.0],
                [out_w - 1.0, out_h - 1.0],
                [0.0, out_h - 1.0]
            ], dtype=np.float32)
            warped = add_fuzzy_drop_shadow(
                warped, corners,
                offset=(offset_x, offset_y),
                blur_radius=spread,
                opacity=opacity
            )
        if random.random() < 0.3:
            blur_min = random.uniform(0, 2)
            blur_max = blur_min + random.uniform(2, 6)
            edge_focus = random.uniform(0.3, 0.7)
            warped = add_variable_blur(warped, blur_min, blur_max, edge_focus)
        if random.random() < 0.6:
            warped = add_screen_glare(warped)

    # Composite warped image onto canvas using polygon mask (not alpha)
    canvas_h, canvas_w = bg.shape[:2]

    x1 = max(0, min_x)
    y1 = max(0, min_y)
    x2 = min(canvas_w, max_x + 1)
    y2 = min(canvas_h, max_y + 1)

    if x2 > x1 and y2 > y1:
        src_y1 = max(0, -min_y)
        src_y2 = min(out_h, canvas_h - min_y)
        src_x1 = max(0, -min_x)
        src_x2 = min(out_w, canvas_w - min_x)
        dst_h = y2 - y1
        dst_w = x2 - x1
        warp_slice = warped[src_y1:src_y2, src_x1:src_x2]
        mask_slice = mask[src_y1:src_y2, src_x1:src_x2]
        canvas_slice = bg[y1:y2, x1:x2]

        if warp_slice.shape[:2] == canvas_slice.shape[:2]:
            # Use polygon mask for alpha (not warped image alpha)
            alpha = mask_slice[:, :, np.newaxis].astype(np.float32) / 255.0
            
            result = (canvas_slice.astype(np.float32) * (1 - alpha) +
                      warp_slice[:, :, :3].astype(np.float32) * alpha)
            bg[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)
        elif warp_slice.shape[0] > 0 and warp_slice.shape[1] > 0:
            resized = cv2.resize(warp_slice[:, :, :3], (dst_w, dst_h))
            resized_mask = cv2.resize(mask_slice, (dst_w, dst_h))
            alpha = resized_mask[:, :, np.newaxis].astype(np.float32) / 255.0
            result = (canvas_slice.astype(np.float32) * (1 - alpha) +
                      resized.astype(np.float32) * alpha)
            bg[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)

    return bg


def get_photo_bbox(corners, shadow_extent=0):
    """Get bounding box for photo including shadow."""
    min_x = corners[:, 0].min() - shadow_extent
    max_x = corners[:, 0].max() + shadow_extent
    min_y = corners[:, 1].min() - shadow_extent
    max_y = corners[:, 1].max() + shadow_extent
    
    w = max_x - min_x
    h = max_y - min_y
    
    return {
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'w': w, 'h': h,
        'corners': corners.copy()
    }


def check_bounds(bbox, canvas_w, canvas_h, edge_margin, shadow_extent=0):
    """Check if bounding box is within canvas bounds."""
    return (bbox['min_x'] >= edge_margin + shadow_extent and 
            bbox['max_x'] <= canvas_w - edge_margin - shadow_extent and
            bbox['min_y'] >= edge_margin + shadow_extent and 
            bbox['max_y'] <= canvas_h - edge_margin - shadow_extent)


def check_no_overlap_corners(corners1, corners2, gap):
    """Check if two photo corners don't overlap (no overlap = True)."""
    gap_half = gap / 2
    
    min1_x, max1_x = corners1[:, 0].min(), corners1[:, 0].max()
    min1_y, max1_y = corners1[:, 1].min(), corners1[:, 1].max()
    min2_x, max2_x = corners2[:, 0].min(), corners2[:, 0].max()
    min2_y, max2_y = corners2[:, 1].min(), corners2[:, 1].max()
    
    if max1_x + gap_half < min2_x or min1_x - gap_half > max2_x:
        return True
    if max1_y + gap_half < min2_y or min1_y - gap_half > max2_y:
        return True
    
    for poly in [corners1, corners2]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            if np.abs(edge).sum() < 0.001:
                continue
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            proj1 = corners1 @ normal
            proj2 = corners2 @ normal
            if proj1.max() + gap_half < proj2.min() or proj2.max() + gap_half < proj1.min():
                return True
    
    return False


def polygon_area(corners):
    """Calculate polygon area using shoelace formula."""
    x = corners[:, 0]
    y = corners[:, 1]
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= y[i] * x[j]
    return abs(area) / 2.0


def place_photo(photo_data, bbox, center_x, center_y, placed_corners, canvas_w, canvas_h):
    """Create placed photo info."""
    corners = photo_data['corners']
    
    info = {
        'min_x': center_x - bbox['w']/2,
        'max_x': center_x + bbox['w']/2,
        'min_y': center_y - bbox['h']/2,
        'max_y': center_y + bbox['h']/2,
        'center_x': center_x,
        'center_y': center_y,
        'corners': placed_corners.copy()
    }
    
    kps = [(placed_corners[i][0], placed_corners[i][1]) for i in range(4)]
    
    infos = {
        'x_center': sum(k[0] for k in kps) / 4 / canvas_w,
        'y_center': sum(k[1] for k in kps) / 4 / canvas_h,
        'keypoints': kps,
        'fill_area': polygon_area(placed_corners)
    }
    
    return info, infos


def pack_random_greedy(photos, photo_bboxes, canvas_w, canvas_h, edge_margin, photo_gap, max_shadow_extent):
    """Pack photos using random greedy placement with rotation support."""
    num_photos = len(photos)
    if num_photos == 0:
        return None
    
    placed = []
    infos = []
    
    # Generate random positions - use conservative bounds to ensure photos stay well within canvas
    safe_margin = max(edge_margin, 200)  # At least 200px from edge
    positions = []
    for _ in range(1000):
        x = random.uniform(safe_margin, canvas_w - safe_margin)
        y = random.uniform(safe_margin, canvas_h - safe_margin)
        positions.append((x, y))
    
    random.shuffle(positions)
    
    # Sort photos by size (smaller first)
    photo_order = sorted(range(num_photos), key=lambda i: photo_bboxes[i]['w'] * photo_bboxes[i]['h'])
    
    for photo_idx in photo_order:
        bbox = photo_bboxes[photo_idx]
        w = bbox['w']
        h = bbox['h']
        
        # Random rotation angle (-20 to +20 degrees)
        angle = random.uniform(-20, 20)
        
        best_pos = None
        best_corners = None
        
        for x, y in positions:
            # Create rotated corners for this position
            test_corners = get_rotated_bbox_corners(w, h, angle, x, y)
            
            # Check bounds using actual rotated corners
            bbox_check = get_bounding_box_from_corners(test_corners)
            if bbox_check['min_x'] < edge_margin or bbox_check['max_x'] > canvas_w - edge_margin or \
               bbox_check['min_y'] < edge_margin or bbox_check['max_y'] > canvas_h - edge_margin:
                continue
            
            # Check overlap with all placed photos
            overlap = False
            for p in placed:
                if not check_no_overlap_corners(test_corners, p['corners'], photo_gap):
                    overlap = True
                    break
            
            if not overlap:
                best_pos = (x, y)
                best_corners = test_corners
                break
        
        if best_pos:
            x, y = best_pos
            
            placed.append({
                'min_x': best_corners[:, 0].min(),
                'max_x': best_corners[:, 0].max(),
                'min_y': best_corners[:, 1].min(),
                'max_y': best_corners[:, 1].max(),
                'center_x': x,
                'center_y': y,
                'corners': best_corners,
                'rotation': angle
            })
            
            kps = [(best_corners[i][0], best_corners[i][1]) for i in range(4)]
            infos.append({
                'x_center': sum(k[0] for k in kps) / 4 / canvas_w,
                'y_center': sum(k[1] for k in kps) / 4 / canvas_h,
                'keypoints': kps,
                'fill_area': polygon_area(best_corners)
            })
    
    if len(placed) == 0:
        return None
    
    return placed, infos


def apply_per_photo_keystone_to_corners(rect_corners, strength=None, margin=30):
    """Warp rectangular corners into a trapezoid to simulate camera tilt on each photo.

    Creates an asymmetric quadrilateral by:
    1. Compressing one edge horizontally by 40% of half-width
    2. Stretching one diagonal vertically by 50% of height
    
    This guarantees both diagonal_ratio < 0.80 AND edge_ratio < 0.80
    regardless of photo aspect ratio.

    Args:
        rect_corners: 4x2 array of rectangular corners [TL, TR, BR, BL]
        strength: ignored (kept for API compatibility)
        margin: safety margin from output bounds (default 30px)

    Returns:
        warped_corners: 4x2 array of warped quadrilateral corners (clamped to bounds)
        transform_matrix: 3x3 perspective transform matrix (for image warping)
        src_pts: source corner points (rectangle)
        dst_pts: destination corner points (trapezoid, before clamping)
    """
    src_pts = rect_corners.copy().astype(np.float32)
    
    w = rect_corners[1,0] - rect_corners[0,0]
    h = rect_corners[2,1] - rect_corners[1,1]
    
    # Scale offsets by photo dimensions
    edge_comp = 0.40 * w / 2  # 40% of half-width each side
    vert_offset = 0.50 * h     # 50% of height
    
    # Randomize which edge is compressed and which diagonal is stretched
    compress_top = random.choice([True, False])
    stretch_diag1 = random.choice([True, False])
    
    # Start with original corners
    tl_x, tl_y = rect_corners[0,0], rect_corners[0,1]
    tr_x, tr_y = rect_corners[1,0], rect_corners[1,1]
    br_x, br_y = rect_corners[2,0], rect_corners[2,1]
    bl_x, bl_y = rect_corners[3,0], rect_corners[3,1]
    
    # Apply compression and stretch
    if compress_top:
        tl_x += edge_comp
        tr_x -= edge_comp
        if stretch_diag1:
            # Stretch TL-BR diagonal: TL moves up, BR moves down
            tl_y -= vert_offset
            br_y += vert_offset
        else:
            # Stretch TR-BL diagonal: TR moves up, BL moves down
            tr_y -= vert_offset
            bl_y += vert_offset
    else:
        bl_x += edge_comp
        br_x -= edge_comp
        if stretch_diag1:
            tl_y -= vert_offset
            br_y += vert_offset
        else:
            tr_y -= vert_offset
            bl_y += vert_offset
    
    dst_pts = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)
    
    # Clamp to bounds
    dst_pts[:, 0] = np.clip(dst_pts[:, 0], margin, 4000 - margin)
    dst_pts[:, 1] = np.clip(dst_pts[:, 1], margin, 3000 - margin)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return dst_pts, M, src_pts, dst_pts

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return dst_pts, M, src_pts, dst_pts


def pack_grid(photos, photo_bboxes, canvas_w, canvas_h, edge_margin, photo_gap, max_shadow_extent):
    """Pack photos in a grid pattern with rotation support.
    
    Uses actual rotated corners for all validation to ensure photos
    stay well within bounds and don't appear to intersect.
    """
    num_photos = len(photos)
    if num_photos == 0:
        return None
    
    # Determine grid dimensions
    cols = int(math.ceil(math.sqrt(num_photos)))
    rows = int(math.ceil(num_photos / cols))
    
    avail_w = canvas_w - 2 * edge_margin
    avail_h = canvas_h - 2 * edge_margin
    
    # Conservative rotation buffer - photos extend ~6% at 20 degrees
    rotation_buffer = 1.06
    
    # Cell size that accounts for gaps - give substantial space between photos
    # Key: cell_size + photo_gap should fit within available space
    cell_w = (avail_w - (cols - 1) * photo_gap) / cols
    cell_h = (avail_h - (rows - 1) * photo_gap) / rows
    
    # Apply rotation buffer conservatively
    effective_cell_w = cell_w / rotation_buffer
    effective_cell_h = cell_h / rotation_buffer
    
    placed = []
    infos = []
    
    # Sort photos by size (larger first)
    photo_order = sorted(range(num_photos), key=lambda i: -(photo_bboxes[i]['w'] * photo_bboxes[i]['h']))
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= num_photos:
                break
            
            photo_idx = photo_order[idx]
            bbox = photo_bboxes[photo_idx]
            w, h = bbox['w'], bbox['h']
            
            # Calculate center position
            x = edge_margin + col * (cell_w + photo_gap) + cell_w / 2
            y = edge_margin + row * (cell_h + photo_gap) + cell_h / 2
            
            # Random rotation (-20 to +20 degrees)
            angle = random.uniform(-20, 20)
            
            # Scale photo to fit within effective cell
            scale = min(effective_cell_w / w, effective_cell_h / h, 1.0)
            scaled_w = w * scale
            scaled_h = h * scale
            
            # Get corners for this scaled photo with rotation
            corners = np.array([
                [-scaled_w/2, -scaled_h/2],
                [scaled_w/2, -scaled_h/2],
                [scaled_w/2, scaled_h/2],
                [-scaled_w/2, scaled_h/2]
            ], dtype=np.float32)
            
            # Rotate and place
            placed_corners = rotate_corners(corners, angle, x, y)
            
            # Validate placement - check bounds and overlap
            valid = True
            bbox_check = get_bounding_box_from_corners(placed_corners)
            
            # Check bounds with safety margin
            safety = photo_gap * 0.5
            if bbox_check['min_x'] < edge_margin - safety or bbox_check['max_x'] > canvas_w - edge_margin + safety or \
               bbox_check['min_y'] < edge_margin - safety or bbox_check['max_y'] > canvas_h - edge_margin + safety:
                valid = False
            
            if valid:
                for p in placed:
                    if not check_no_overlap_corners(placed_corners, p['corners'], photo_gap):
                        valid = False
                        break
            
            if valid:
                placed.append({
                    'min_x': placed_corners[:, 0].min(),
                    'max_x': placed_corners[:, 0].max(),
                    'min_y': placed_corners[:, 1].min(),
                    'max_y': placed_corners[:, 1].max(),
                    'center_x': x,
                    'center_y': y,
                    'corners': placed_corners,
                    'rotation': angle
                })
                
                kps = [(placed_corners[i][0], placed_corners[i,1]) for i in range(4)]
                infos.append({
                    'x_center': sum(k[0] for k in kps) / 4 / canvas_w,
                    'y_center': sum(k[1] for k in kps) / 4 / canvas_h,
                    'keypoints': kps,
                    'fill_area': polygon_area(placed_corners)
                })
            
            idx += 1
    
    # Final validation - ensure all corners are well within bounds
    validated = []
    validated_infos = []
    
    safe_margin = edge_margin * 0.5  # Photos must stay this far from edge
    for p, info in zip(placed, infos):
        corners = p['corners']
        corners_min_x = corners[:, 0].min()
        corners_max_x = corners[:, 0].max()
        corners_min_y = corners[:, 1].min()
        corners_max_y = corners[:, 1].max()
        
        # Check all four corners are within safe bounds
        if (corners_min_x >= safe_margin and corners_max_x <= canvas_w - safe_margin and
            corners_min_y >= safe_margin and corners_max_y <= canvas_h - safe_margin):
            validated.append(p)
            validated_infos.append(info)
    
    return (validated, validated_infos) if validated else None


def pack_bottom_up(photos, photo_bboxes, canvas_w, canvas_h, edge_margin, photo_gap, max_shadow_extent):
    """Pack photos from bottom up, filling rows with maximum fill."""
    num_photos = len(photos)
    if num_photos == 0:
        return None
    
    avail_w = canvas_w - 2 * edge_margin
    avail_h = canvas_h - 2 * edge_margin
    
    # Sort photos by aspect ratio (portrait vs landscape)
    photo_order = sorted(range(num_photos), key=lambda i: photo_bboxes[i]['h'] / max(1, photo_bboxes[i]['w']))
    
    # Bin packing into rows
    rows = []
    current_row = []
    current_width = 0
    
    for idx in photo_order:
        bbox = photo_bboxes[idx]
        w, h = bbox['w'], bbox['h']
        
        if not current_row or current_width + photo_gap + w <= avail_w:
            current_row.append(idx)
            current_width += w + (photo_gap if len(current_row) > 1 else 0)
        else:
            rows.append(current_row)
            current_row = [idx]
            current_width = w
    
    if current_row:
        rows.append(current_row)
    
    # Calculate scale for each row
    total_scaled_h = 0
    row_scales = []
    row_max_heights = []
    
    for row in rows:
        max_h_in_row = max(photo_bboxes[i]['h'] for i in row)
        total_w_in_row = sum(photo_bboxes[i]['w'] for i in row) + (len(row) - 1) * photo_gap
        
        # Scale to fit width first
        scale_w = avail_w / total_w_in_row if total_w_in_row > 0 else 1.0
        
        # Scale to fit height
        target_h = avail_h * 0.98 / len(rows)
        scale_h = target_h / max_h_in_row if max_h_in_row > 0 else 1.0
        
        # Use smaller scale to fit both dimensions
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale beyond original size
        
        row_scales.append(scale)
        row_max_heights.append(max_h_in_row)
        total_scaled_h += max_h_in_row * scale + photo_gap
    
    total_scaled_h -= photo_gap  # Remove last gap
    
    # If too tall, scale down proportionally
    if total_scaled_h > avail_h:
        height_scale = avail_h / total_scaled_h
        row_scales = [s * height_scale for s in row_scales]
    
    # Place photos
    y = canvas_h - edge_margin
    placed = []
    infos = []
    
    for row_idx, row in enumerate(rows):
        scale = row_scales[row_idx]
        row_max_h = row_max_heights[row_idx]
        
        x = edge_margin
        scaled_row_max_h = row_max_h * scale
        
        for idx in row:
            bbox = photo_bboxes[idx]
            corners = bbox['corners'].copy()
            
            scaled_w = bbox['w'] * scale
            scaled_h = bbox['h'] * scale
            
            px = x + scaled_w / 2
            py = y - scaled_row_max_h / 2
            
            bbox_min_x = corners[:, 0].min()
            bbox_min_y = corners[:, 1].min()
            
            placed_corners = np.zeros_like(corners)
            placed_corners[:, 0] = px - scaled_w/2 + (corners[:, 0] - bbox_min_x) * scale
            placed_corners[:, 1] = py - scaled_h/2 + (corners[:, 1] - bbox_min_y) * scale
            
            placed.append({
                'min_x': placed_corners[:, 0].min(),
                'max_x': placed_corners[:, 0].max(),
                'min_y': placed_corners[:, 1].min(),
                'max_y': placed_corners[:, 1].max(),
                'center_x': px,
                'center_y': py,
                'corners': placed_corners
            })
            
            kps = [(placed_corners[i][0], placed_corners[i][1]) for i in range(4)]
            infos.append({
                'x_center': sum(k[0] for k in kps) / 4 / canvas_w,
                'y_center': sum(k[1] for k in kps) / 4 / canvas_h,
                'keypoints': kps,
                'fill_area': polygon_area(placed_corners)
            })
            
            x += scaled_w + photo_gap
        
        y -= scaled_row_max_h + photo_gap
    
    # Validate placement
    for p in placed:
        c = p['corners']
        if c[:, 0].min() < edge_margin or c[:, 0].max() > canvas_w - edge_margin or \
           c[:, 1].min() < edge_margin or c[:, 1].max() > canvas_h - edge_margin:
            return None  # Invalid placement
    
    return placed, infos if placed else None


def render_scene(config, sources, requested_photos=None, canvas_scale_override=None):
    """Render scene with photos placed flat, then apply global perspective transform.
    
    Steps:
    1. Create high-res flat canvas (2x final size for quality)
    2. Place photos flat (no perspective warping per photo)
    3. Apply global perspective transform to simulate camera angle
    4. Downsample to final size
    5. Calculate keypoints from final transformed positions
    
    Args:
        config: Config object with generation parameters
        sources: list of source image paths
        requested_photos: override number of photos
        canvas_scale_override: override canvas scale factor (1=fast test, 3=production)
    """
    # Final output size: 4:3 ratio, at least 2000x3000
    final_h = 3000
    final_w = int(final_h * 4 / 3)  # 4000
    
    # High-res canvas for better quality and to ensure perspective warp
    # doesn't crop into valid content. Use 3x resolution to ensure the
    # warped image can be cropped to final size with no black edges.
    scale_factor = canvas_scale_override if canvas_scale_override else 3
    canvas_w = final_w * scale_factor
    canvas_h = final_h * scale_factor
    
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
    
    # Larger margin to prevent photos going off edge after perspective transform
    # The 3x scale_factor gives us room, but we need safety margin for global warp
    edge_margin = 120 * scale_factor  # Safe margin from edge
    photo_gap = 45 * scale_factor  # Larger gap to prevent apparent intersections
    
    avail_w = canvas_w - 2 * edge_margin
    avail_h = canvas_h - 2 * edge_margin
    
    # Single photo: maximize fill
    if num_photos == 1:
        ideal_size = min(avail_w, avail_h) * 0.98
        target_size = int(ideal_size)
        base_min = max(1600, int(target_size * 0.95))
        base_max = max(base_min + 400, int(target_size * 1.05))
        
        photo, corners, shadow_mask, original_photo, img_size = process_photo(str(sampled[0]), base_min, base_max)
        if photo is None:
            return bg, [], (final_w, final_h)
        
        photo_h, photo_w = photo.shape[:2]
        
        # Use much more conservative fill to account for rotation and perspective
        # At 20 degrees rotation, photos extend ~6% beyond their bounding box
        # We need room for both rotation AND perspective distortion
        fill_factor = 0.75  # Conservative to ensure corners stay within bounds
        scale_x = avail_w * fill_factor / photo_w
        scale_y = avail_h * fill_factor / photo_h
        scale = min(scale_x, scale_y)
        
        # Random rotation (-20 to +20 degrees)
        angle = random.uniform(-20, 20)
        
        cx, cy = canvas_w / 2, canvas_h / 2
        
        # Create scaled corners centered at origin
        scaled_corners = np.array([
            [-photo_w * scale / 2, -photo_h * scale / 2],
            [photo_w * scale / 2, -photo_h * scale / 2],
            [photo_w * scale / 2, photo_h * scale / 2],
            [-photo_w * scale / 2, photo_h * scale / 2]
        ], dtype=np.float32)
        
        # Rotate then place
        new_corners = rotate_corners(scaled_corners, angle, cx, cy)
        
        # Final bounds check - ensure rotated corners stay within canvas
        # Allow substantial margin for perspective transform headroom
        safety_margin = edge_margin * 1.5
        
        if (new_corners[:, 0].min() < safety_margin or new_corners[:, 0].max() > canvas_w - safety_margin or
            new_corners[:, 1].min() < safety_margin or new_corners[:, 1].max() > canvas_h - safety_margin):
            # Photo would go off edge after rotation, scale it down
            # Scale down photo so rotated corners fit safely within canvas
            # Account for rotation (6% at 20 degrees) with safety margin
            rotation_extension = 1.08
            max_allowed_size = (canvas_w - 2 * safety_margin) / rotation_extension
            
            # Calculate how much we need to scale
            current_max_dim = max(new_corners[:, 0].max() - new_corners[:, 0].min(),
                                  new_corners[:, 1].max() - new_corners[:, 1].min())
            if current_max_dim > max_allowed_size:
                scale *= max_allowed_size / current_max_dim
            
            # Recalculate corners with adjusted scale
            scaled_corners = np.array([
                [-photo_w * scale / 2, -photo_h * scale / 2],
                [photo_w * scale / 2, -photo_h * scale / 2],
                [photo_w * scale / 2, photo_h * scale / 2],
                [-photo_w * scale / 2, photo_h * scale / 2]
            ], dtype=np.float32)
            new_corners = rotate_corners(scaled_corners, angle, cx, cy)
        
        # Recomposite on fresh background
        bg = bg_gen.generate(canvas_w, canvas_h)
        bg = apply_luma_gradient(bg)
        bg = composite_rotated_photo(bg, original_photo, new_corners, with_effects=False)
        
        # Apply global perspective transform
        # Pass scale_factor so the transform can crop appropriately
        placed_info_list = [{'corners': new_corners.copy()}]
        bg, transformed_corners_list, black_pct = apply_global_perspective_transform(
            bg, placed_info_list, final_w, final_h, scale_factor
        )

        # Retry with larger canvas if black edges detected
        if black_pct > 0.001:
            scale_factor = 4
            canvas_w = final_w * scale_factor
            canvas_h = final_h * scale_factor
            bg = bg_gen.generate(canvas_w, canvas_h)
            bg = apply_luma_gradient(bg)
            bg = composite_rotated_photo(bg, original_photo, new_corners, with_effects=False)
            placed_info_list = [{'corners': new_corners.copy()}]
            bg, transformed_corners_list, _ = apply_global_perspective_transform(
                bg, placed_info_list, final_w, final_h, scale_factor
            )
        
        # Keypoints are already in output coordinate space from transform
        info = {
            'keypoints': [(transformed_corners_list[0][i, 0], transformed_corners_list[0][i, 1]) for i in range(4)],
            'fill_area': polygon_area(transformed_corners_list[0])
        }
        
        return bg, [info], (final_w, final_h)
    
    # Multiple photos: simple grid placement without binary search
    cols = int(math.ceil(math.sqrt(num_photos)))
    rows = int(math.ceil(num_photos / cols))
    
    # Optimize grid for different photo counts
    if num_photos == 2:
        cols, rows = 2, 1
    elif num_photos == 3:
        cols, rows = 3, 1
    elif num_photos == 5:
        cols, rows = 3, 2
    elif num_photos == 6:
        cols, rows = 3, 2
    elif num_photos == 7:
        cols, rows = 4, 2
    elif num_photos == 8:
        cols, rows = 4, 2
    
    # Simple fixed size based on available space - conservatively sized
    # Account for rotation (6% at 20 degrees) and gap between photos
    rotation_safety = 1.10
    gap_safety = photo_gap * 2  # Extra safety for gaps
    
    cell_w = (avail_w - gap_safety - (cols - 1) * photo_gap) / cols / rotation_safety
    cell_h = (avail_h - gap_safety - (rows - 1) * photo_gap) / rows / rotation_safety
    target_size = int(min(cell_w, cell_h))
    base_min = max(1200, int(target_size * 0.85))
    base_max = max(base_min + 400, int(target_size * 1.15))
    
    photos = []
    for path in sampled:
        photo, corners, shadow_mask, original_photo, img_size = process_photo(str(path), base_min, base_max)
        if photo is not None and photo.shape[0] > 50 and photo.shape[1] > 50:
            photos.append({
                'photo': photo,
                'corners': corners,
                'shadow_mask': shadow_mask,
                'original_photo': original_photo,
                'img_size': img_size
            })
    
    if len(photos) < num_photos:
        if len(photos) == 0:
            return np.zeros((final_h, final_w, 3), dtype=np.uint8), [], (final_w, final_h)
        photos = (photos * (num_photos // len(photos) + 1))[:num_photos]
    
    photo_bboxes = []
    for photo_data in photos:
        corners = photo_data['corners']
        bbox = get_photo_bbox(corners, 0)
        photo_bboxes.append(bbox)
    
    result = pack_grid(photos, photo_bboxes, canvas_w, canvas_h, edge_margin, photo_gap, 0)
    
    if result is None:
        return bg, [], (final_w, final_h)
    
    placed, infos = result

    # Recomposite photos with per-photo keystone warping.
    # PRIMARY approach: per-photo keystone creates visible trapezoids.
    # We work in output space directly with NO global perspective transform
    # (which would crop and resize, interfering with keystone corners).
    #
    # Pipeline:
    # 1. Scale placed corners from canvas space to output space (by scale_factor)
    # 2. Apply per-photo keystone to create trapezoidal corners (clamped to bounds)
    # 3. Composite each photo warped to its trapezoid corners (in output space)
    # 4. Update keypoints with the trapezoidal (keystone) corners
    #
    bg = bg_gen.generate(canvas_w, canvas_h)
    bg = apply_luma_gradient(bg)

    # Generate output-space canvas by downsampling the canvas
    # (simulates the "camera looking down" effect without clipping corners)
    output_canvas = cv2.resize(bg, (final_w, final_h), interpolation=cv2.INTER_AREA)

    # Scale factor from canvas space to output space
    canvas_to_output_x = final_w / canvas_w
    canvas_to_output_y = final_h / canvas_h

    placed_warped = []  # output-space keystone corners for each photo
    output_infos = []   # updated keypoints in output space

    for i, photo_data in enumerate(photos[:len(placed)]):
        placed_corners = placed[i]['corners']

        # Scale canvas-space corners to output space (no crop, just resize)
        output_corners = placed_corners.copy()
        output_corners[:, 0] = placed_corners[:, 0] * canvas_to_output_x
        output_corners[:, 1] = placed_corners[:, 1] * canvas_to_output_y

        # Apply per-photo keystone warp to create trapezoidal corners
        # Keystone corners are clamped to [30, W-30] x [30, H-30] to stay in bounds
        keystone_corners, M, src_pts, dst_pts = apply_per_photo_keystone_to_corners(
            output_corners
        )
        placed_warped.append(keystone_corners)

        # Composite the photo warped to its trapezoid corners (in output space)
        output_canvas = composite_warped_photo(
            output_canvas, photo_data['original_photo'],
            keystone_corners, with_effects=False
        )

        # Update keypoints with trapezoidal corners
        kps = [(keystone_corners[j][0], keystone_corners[j][1]) for j in range(4)]
        x_center = sum(k[0] for k in kps) / 4
        y_center = sum(k[1] for k in kps) / 4
        output_infos.append({
            'x_center': x_center,
            'y_center': y_center,
            'keypoints': kps,
            'fill_area': polygon_area(keystone_corners)
        })

    # Update infos with output-space keypoints
    infos = output_infos

    bg = output_canvas

    fill_pct = sum(info['fill_area'] for info in infos) / (final_w * final_h) * 100
    print(f"  Final: fill={fill_pct:.1f}%, photos={len(placed)}")

    return bg, infos, (final_w, final_h)


# =============================================================================
# AUTOMATED VERIFICATION TESTS
# =============================================================================
# These tests verify each feature without human visual inspection.
# Run via: python -c "from generate_dataset import *; run_batch_tests(...)"

def test_no_black_border(img, threshold=0.001):
    """FAIL if >threshold fraction of pixels are fully black (R=G=B=0).

    Args:
        img: BGR image (numpy array)
        threshold: maximum allowed fraction of black pixels

    Returns:
        (passed: bool, detail: str)
    """
    mask = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    pct = mask.sum() / (img.shape[0] * img.shape[1])
    if pct > threshold:
        return False, f"Black border: {pct*100:.3f}% of image is black (max {threshold*100:.3f}%)"
    return True, ""


def test_perspective_distortion(corners, min_ratio=0.80):
    """FAIL if quadrilateral appears too rectangular.

    Computes the ratio of the two diagonals. A perfect rectangle = 1.0.
    A strong keystone should be visibly trapezoidal.

    Args:
        corners: 4x2 array of corner coordinates (in output image space)
        min_ratio: diagonals_ratio must be < this value to be considered trapezoidal

    Returns:
        (passed: bool, detail: str)
    """
    d1 = np.linalg.norm(corners[0] - corners[2])  # TL-BR diagonal
    d2 = np.linalg.norm(corners[1] - corners[3])  # TR-BL diagonal
    diag_ratio = min(d1, d2) / max(d1, d2)

    # Also compute top-edge vs bottom-edge width ratio
    top_w = np.linalg.norm(corners[1] - corners[0])
    bot_w = np.linalg.norm(corners[2] - corners[3])
    edge_ratio = min(top_w, bot_w) / max(top_w, bot_w)

    if diag_ratio > min_ratio and edge_ratio > min_ratio:
        return False, (
            f"Perspective too weak: diag_ratio={diag_ratio:.3f}, edge_ratio={edge_ratio:.3f} "
            f"(both should be < {min_ratio} for visible trapezoid)"
        )
    return True, ""


def test_keystone_displacement(corners, img_w, img_h, min_pct=0.03):
    """FAIL if no corner is displaced by at least min_pct of max image dimension.

    Args:
        corners: 4x2 array of corner coordinates (in output image space)
        img_w, img_h: output image dimensions
        min_pct: minimum displacement as fraction of max dimension

    Returns:
        (passed: bool, detail: str)
    """
    flat_corners = np.array([
        [0.0, 0.0],
        [img_w - 1.0, 0.0],
        [img_w - 1.0, img_h - 1.0],
        [0.0, img_h - 1.0]
    ], dtype=np.float32)

    threshold_px = min_pct * max(img_w, img_h)
    for tc in corners:
        min_dist = min(np.linalg.norm(tc - fc) for fc in flat_corners)
        if min_dist >= threshold_px:
            return True, ""

    return False, (
        f"All corners too close to flat positions (max displacement {threshold_px:.1f}px). "
        f"Perspective may be invisible."
    )


def test_fuzzy_shadow(img, photo_mask, max_grad=15.0):
    """FAIL if shadow edge has hard falloff (mean Sobel gradient > max_grad).

    Samples the gradient magnitude at the shadow boundary. A fuzzy shadow
    has gradual falloff (mean gradient < max_grad at the edge).
    Requires photo_mask (alpha channel or binary mask of photo pixels).

    Args:
        img: BGR image (for color context, not strictly needed)
        photo_mask: binary mask of photo pixels (0=bg, 255=photo)
        max_grad: maximum allowed mean Sobel gradient at shadow boundary

    Returns:
        (passed: bool, detail: str)
    """
    photo_mask = photo_mask.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)

    # Dilate photo mask to find boundary region
    dilated = cv2.dilate((photo_mask).astype(np.uint8), kernel, iterations=3)
    boundary = cv2.subtract(dilated, (photo_mask).astype(np.uint8)).astype(np.float32) / 255.0

    if boundary.sum() == 0:
        return True, "No boundary detected (shadow may be absent)"

    gx = cv2.Sobel(boundary, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(boundary, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    mean_grad = grad_mag[boundary > 0].mean()
    if mean_grad >= max_grad:
        return False, f"Shadow too hard: mean gradient={mean_grad:.2f} (should be <{max_grad})"
    return True, ""


def test_glare_brightening(img_before, img_after, min_pct=0.01, min_delta=5):
    """FAIL if glare didn't brighten at least min_pct of pixels by >min_delta.

    Args:
        img_before: BGR image before glare
        img_after: BGR image after glare
        min_pct: minimum fraction of pixels that must brighten
        min_delta: minimum brightness increase

    Returns:
        (passed: bool, detail: str)
    """
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = gray_after - gray_before
    diff_pct = (diff > min_delta).sum() / diff.size
    if diff_pct < min_pct:
        return False, (
            f"Glare effect too weak: only {diff_pct*100:.2f}% of pixels brightened "
            f"by >{min_delta} (min {min_pct*100:.2f}%)"
        )
    return True, ""


def test_background_color_distribution(img, dark_thresh=60, light_thresh=200):
    """Categorize background as dark/light/mid based on border pixel brightness.

    Samples a 100px border strip to avoid photos in the center.
    Dark: mean < dark_thresh, Light: mean > light_thresh, Mid: between.

    Args:
        img: BGR image
        dark_thresh: maximum brightness for "dark" categorization
        light_thresh: minimum brightness for "light" categorization

    Returns:
        (category: str, mean_brightness: float)
    """
    h, w = img.shape[:2]
    top_band = img[:100, :].reshape(-1, 3)
    bot_band = img[-100:, :].reshape(-1, 3)
    left_band = img[:, :100].reshape(-1, 3)
    right_band = img[:, -100:].reshape(-1, 3)
    border_pixels = np.concatenate([top_band, bot_band, left_band, right_band])
    mean_brightness = border_pixels.mean()
    if mean_brightness < dark_thresh:
        return "dark", mean_brightness
    elif mean_brightness > light_thresh:
        return "light", mean_brightness
    return "mid", mean_brightness


def test_background_has_extremes(categories):
    """FAIL if batch doesn't contain both dark and light backgrounds.

    Args:
        categories: list of category strings ('dark', 'light', 'mid')

    Returns:
        (passed: bool, detail: str)
    """
    if 'dark' not in categories:
        return False, f"No dark backgrounds in batch: {set(categories)}"
    if 'light' not in categories:
        return False, f"No light backgrounds in batch: {set(categories)}"
    return True, ""


def test_photo_color_changed(img_before, img_after, max_intersection=0.95):
    """FAIL if manipulated photo is identical to original (histogram intersection).

    Args:
        img_before: BGR image before manipulation
        img_after: BGR image after manipulation
        max_intersection: histograms more similar than this = no effect

    Returns:
        (passed: bool, detail: str)
    """
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    hist_before = cv2.calcHist([gray_before], [0], None, [256], [0, 256])
    hist_after = cv2.calcHist([gray_after], [0], None, [256], [0, 256])
    hist_before = hist_before / hist_before.sum()
    hist_after = hist_after / hist_after.sum()
    intersection = np.sum(np.minimum(hist_before, hist_after))
    if intersection > max_intersection:
        return False, (
            f"Photo color manipulation had no effect: "
            f"histogram intersection={intersection:.3f} (should be <{max_intersection})"
        )
    return True, ""


def test_min_gap_between_photos(all_corners, min_gap_px=20):
    """FAIL if any two photo corners are closer than min_gap_px.

    Args:
        all_corners: list of 4x2 arrays, one per photo
        min_gap_px: minimum Euclidean distance between any two corners

    Returns:
        (passed: bool, detail: str)
    """
    for i in range(len(all_corners)):
        for j in range(i + 1, len(all_corners)):
            for ci in all_corners[i]:
                for cj in all_corners[j]:
                    dist = np.linalg.norm(ci - cj)
                    if dist < min_gap_px:
                        return False, (
                            f"Gap between photos too small: {dist:.1f}px < {min_gap_px}px"
                        )
    return True, ""


def test_edge_margin(corners_list, img_w, img_h, min_margin_px=30):
    """FAIL if any corner is within min_margin_px of image edge.

    Args:
        corners_list: list of 4x2 arrays, one per photo
        img_w, img_h: output image dimensions
        min_margin_px: minimum distance from any corner to any edge

    Returns:
        (passed: bool, detail: str)
    """
    for corners in corners_list:
        for c in corners:
            x, y = c[0], c[1]
            if x < min_margin_px or x > img_w - min_margin_px:
                return False, f"Corner x={x:.1f} violates edge margin ({min_margin_px}px)"
            if y < min_margin_px or y > img_h - min_margin_px:
                return False, f"Corner y={y:.1f} violates edge margin ({min_margin_px}px)"
    return True, ""


def test_keypoints_in_bounds(corners_list, img_w, img_h, margin_px=5):
    """FAIL if any keypoint is outside image bounds (with margin_px safe margin).

    Args:
        corners_list: list of 4x2 arrays, one per photo
        img_w, img_h: output image dimensions
        margin_px: safe margin from image boundary

    Returns:
        (passed: bool, detail: str)
    """
    for i, corners in enumerate(corners_list):
        for j, c in enumerate(corners):
            x, y = c[0], c[1]
            if not (margin_px <= x <= img_w - margin_px):
                return False, f"Photo {i} corner {j} x={x:.1f} out of bounds [${margin_px}, ${img_w-margin_px}]"
            if not (margin_px <= y <= img_h - margin_px):
                return False, f"Photo {i} corner {j} y={y:.1f} out of bounds [${margin_px}, ${img_h-margin_px}]"
    return True, ""


def test_yolo_labels_valid(label_path, img_w, img_h):
    """FAIL if YOLO label file has invalid format.

    Each line: class x_center y_center width height [kp_x kp_y kp_visible]*4
    All normalized values must be in [0, 1]. Exactly 4 keypoints per entry.

    Args:
        label_path: path to .txt label file
        img_w, img_h: image dimensions (for context in error messages)

    Returns:
        (passed: bool, detail: str)
    """
    with open(label_path, 'r') as f:
        for lineno, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                return False, f"Line {lineno}: too few fields ({len(parts)} < 5)"
            try:
                values = [float(p) for p in parts[1:5]]
            except ValueError:
                return False, f"Line {lineno}: non-numeric bbox value"
            for v in values:
                if not (0 <= v <= 1):
                    return False, f"Line {lineno}: bbox value {v} out of [0,1]"
            kp_parts = parts[5:]
            if len(kp_parts) != 12:
                return False, f"Line {lineno}: expected 12 keypoint fields (4*3), got {len(kp_parts)}"
            for k in range(4):
                try:
                    kx = float(kp_parts[k * 3])
                    ky = float(kp_parts[k * 3 + 1])
                    kv = float(kp_parts[k * 3 + 2])
                except ValueError:
                    return False, f"Line {lineno}: non-numeric keypoint value for kp{k}"
                if not (0 <= kx <= 1):
                    return False, f"Line {lineno}: kp{k} x={kx} out of [0,1]"
                if not (0 <= ky <= 1):
                    return False, f"Line {lineno}: kp{k} y={ky} out of [0,1]"
                if kv not in (0, 1, 2):
                    return False, f"Line {lineno}: kp{k} visibility={kv} not in {{0,1,2}}"
    return True, ""


def run_automated_tests(img, corners_list, img_w, img_h, label_path=None):
    """Run all per-image automated tests.

    Args:
        img: BGR image (numpy array)
        corners_list: list of 4x2 arrays, one per photo
        img_w, img_h: output image dimensions
        label_path: optional path to YOLO label file

    Returns:
        dict of {test_name: (passed: bool, detail: str)}
    """
    results = {}

    passed, detail = test_no_black_border(img)
    results['test_no_black_border'] = (passed, detail)

    for i, corners in enumerate(corners_list):
        passed, detail = test_perspective_distortion(corners)
        results[f'test_perspective_distortion_photo{i}'] = (passed, detail)

        passed, detail = test_keystone_displacement(corners, img_w, img_h)
        results[f'test_keystone_displacement_photo{i}'] = (passed, detail)

    passed, detail = test_min_gap_between_photos(corners_list)
    results['test_min_gap'] = (passed, detail)

    passed, detail = test_edge_margin(corners_list, img_w, img_h)
    results['test_edge_margin'] = (passed, detail)

    passed, detail = test_keypoints_in_bounds(corners_list, img_w, img_h)
    results['test_keypoints_in_bounds'] = (passed, detail)

    category, brightness = test_background_color_distribution(img)
    results['test_background_color'] = (category, brightness)

    if label_path:
        passed, detail = test_yolo_labels_valid(label_path, img_w, img_h)
        results['test_yolo_labels'] = (passed, detail)

    return results


def run_batch_tests(config, sources, sample_size=20, output_dir=None):
    """Generate sample_size images and run full automated test suite.

    Args:
        config: Config object with generation parameters
        sources: list of source image Path objects
        sample_size: number of images to generate and test
        output_dir: optional path to save generated images/labels

    Returns:
        dict with 'pass', 'fail', 'batch_categories', 'details'
    """
    import tempfile
    results = {'pass': 0, 'fail': 0, 'batch_categories': [], 'details': []}

    # Use temp dir if no output_dir provided
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(tempfile.mkdtemp())

    for d in ['images', 'labels']:
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    for i in range(sample_size):
        bg, infos, canvas_size = render_scene(config, sources)

        final_w, final_h = canvas_size
        corners_list = [np.array(p['keypoints'], dtype=np.float32) for p in infos]

        img_path = out_dir / 'images' / f'test_{i:05d}.jpg'
        lbl_path = out_dir / 'labels' / f'test_{i:05d}.txt'
        cv2.imwrite(str(img_path), bg)

        with open(lbl_path, 'w') as f:
            for p in infos:
                kps = p['keypoints']
                x_center = sum(k[0] for k in kps) / 4 / final_w
                y_center = sum(k[1] for k in kps) / 4 / final_h
                width = (max(k[0] for k in kps) - min(k[0] for k in kps)) / final_w
                height = (max(k[1] for k in kps) - min(k[1] for k in kps)) / final_h
                line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                for kx, ky in kps:
                    line += f" {kx / final_w:.6f} {ky / final_h:.6f} 2"
                line += "\n"
                f.write(line)

        test_results = run_automated_tests(bg, corners_list, final_w, final_h, str(lbl_path))

        # Collect background categories
        for name, val in test_results.items():
            if name == 'test_background_color':
                if isinstance(val[0], str):
                    results['batch_categories'].append(val[0])

        # Check for failures
        failures = []
        for name, val in test_results.items():
            if isinstance(val, tuple) and len(val) == 2:
                passed, detail = val
                if not passed:
                    failures.append(f"{name}: {detail}")

        if failures:
            results['fail'] += 1
            results['details'].append(f"Image {i} ({len(infos)} photos): {'; '.join(failures)}")
        else:
            results['pass'] += 1

    # Batch-level test: background extremes
    if results['batch_categories']:
        passed, detail = test_background_has_extremes(results['batch_categories'])
        if not passed:
            results['fail'] += 1
            results['details'].append(f"BATCH: {detail}")

    total = results['pass'] + results['fail']
    print(f"\nBatch test results: {results['pass']}/{total} passed")
    if results['fail'] > 0:
        for detail in results['details']:
            print(f"  FAIL: {detail}")

    return results


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
            
            pil_img = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
            
            img_path = img_dir / f"{prefix}_{i:05d}.jpg"
            pil_img.save(img_path, quality=95)
            
            lbl_path = lbl_dir / f"{prefix}_{i:05d}.txt"
            self._save_labels(lbl_path, infos, canvas_size)
            
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num}")
    
    def _save_labels(self, path, infos, canvas_size):
        canvas_w, canvas_h = canvas_size
        with open(path, 'w') as f:
            for p in infos:
                kps = p['keypoints']
                x_center = sum(k[0] for k in kps) / 4 / canvas_w
                y_center = sum(k[1] for k in kps) / 4 / canvas_h
                width = (max(k[0] for k in kps) - min(k[0] for k in kps)) / canvas_w
                height = (max(k[1] for k in kps) - min(k[1] for k in kps)) / canvas_h
                
                line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                for kx, ky in kps:
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
    parser.add_argument("--min-photos", type=int, default=DEFAULT_CONFIG["num_photos_min"])
    parser.add_argument("--max-photos", type=int, default=DEFAULT_CONFIG["num_photos_max"])
    parser.add_argument("--min-size", type=int, default=DEFAULT_CONFIG["min_photo_size"])
    parser.add_argument("--max-size", type=int, default=DEFAULT_CONFIG["max_photo_size"])
    parser.add_argument("--source-dir", type=str, default=DEFAULT_CONFIG["source_images"])
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    config = Config(
        num_train_images=args.num_train,
        num_val_images=args.num_val,
        num_photos_min=args.min_photos,
        num_photos_max=args.max_photos,
        min_photo_size=args.min_size,
        max_photo_size=args.max_size,
        source_images=args.source_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    source_dir = Path(args.source_dir)
    sources = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.jpeg')) + list(source_dir.glob('*.png')) + list(source_dir.glob('*.webp'))
    
    print(f"Found {len(sources)} source images in {source_dir}")
    
    if len(sources) < 5:
        print("ERROR: Need at least 5 source images")
        return
    
    generator = DatasetGenerator(config, sources)
    generator.generate()


if __name__ == "__main__":
    main()
