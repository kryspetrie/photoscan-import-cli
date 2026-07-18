#!/usr/bin/env python3
"""
PhotoScan Import CLI — Shared Generation Utilities
==================================================

Common functions used by both the detection and pose data generators.
Extracted from the original `generate.py` to avoid code duplication.

Provides:
  - Background generation (random colors, gradients, textures)
  - Photo effects (glare, shadow, rotation, compositing)
  - Perspective warp
  - Corner/keypoint calculation
  - Debug visualization
"""

import cv2
import numpy as np
from pathlib import Path
import random
import math
import colorsys


# =============================================================================
# Configuration constants (shared)
# =============================================================================

CANVAS_SIZE = 640
PHOTO_SIZE_MIN = 270
PHOTO_SIZE_MAX = 640
ROTATION_RANGE = 30
BOUND_MARGIN = 5


# =============================================================================
# CORNER CALCULATION
# =============================================================================

def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """Calculate corners of a rotated rectangle in LL, UL, UR, LR order.

    Verified correct: these are the 4 corner keypoints used for pose labels.
    """
    if abs(rotation) < 1:
        return np.array([
            [center_x - width / 2, center_y + height / 2],  # LL
            [center_x - width / 2, center_y - height / 2],  # UL
            [center_x + width / 2, center_y - height / 2],  # UR
            [center_x + width / 2, center_y + height / 2],  # LR
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


def compute_rotated_bbox(width, height, center_x, center_y, rotation):
    """Compute axis-aligned bounding box of a rotated rectangle."""
    corners = get_rotated_polygon(width, height, center_x, center_y, rotation)
    xs = corners[:, 0]
    ys = corners[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


# =============================================================================
# PHOTO EFFECTS
# =============================================================================

def rotate_photo(photo, angle):
    """Rotate photo preserving alpha channel."""
    h, w = photo.shape[:2]

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

    return cv2.warpAffine(
        photo, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def fast_glare(img):
    """Add glare highlights using screen blend."""
    if random.random() < 0.5:
        h, w = img.shape[:2]

        for _ in range(random.randint(2, 4)):
            img_f = img.astype(np.float32) / 255.0

            cx = random.uniform(w * 0.15, w * 0.85)
            cy = random.uniform(h * 0.1, h * 0.7)

            rx = random.uniform(w * 0.20, w * 0.40)
            ry = random.uniform(h * 0.20, h * 0.40)

            y, x = np.ogrid[:h, :w]

            flare = np.maximum(0, 1 - (x - cx) ** 2 / (rx ** 2) - (y - cy) ** 2 / (ry ** 2))
            flare = cv2.GaussianBlur(flare.astype(np.float32), (15, 15), 0)

            opacity = random.uniform(0.60, 1.00)
            flare_f = flare[:, :, np.newaxis]
            img_f = 1 - (1 - img_f) * (1 - flare_f * opacity)

            img = np.clip(img_f * 255, 0, 255).astype(np.uint8)

    return img


def apply_photo_shadow(canvas, photo, cx, cy, offset_x, offset_y,
                       blur_sigma, opacity, orig_w, orig_h, rotation):
    """Render a drop shadow beneath the photo onto the canvas."""
    ch, cw = canvas.shape[:2]
    num_channels = canvas.shape[2]

    # Rotate the offset direction to match scene lighting
    rot_rad = math.radians(rotation)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    rotated_offset_x = offset_x * cos_r - offset_y * sin_r
    rotated_offset_y = offset_x * sin_r + offset_y * cos_r

    shadow_cx = cx + rotated_offset_x * 0.5
    shadow_cy = cy + rotated_offset_y * 0.5

    blur_pad = int(3 * blur_sigma) + 1
    mask_w = orig_w + blur_pad * 2
    mask_h = orig_h + blur_pad * 2
    shadow_mask = np.zeros((mask_h, mask_w), dtype=np.float32)
    shadow_mask[blur_pad:blur_pad + orig_h, blur_pad:blur_pad + orig_w] = 1.0
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=blur_sigma)

    if abs(rotation) > 0.5:
        center_rot = (mask_w / 2, mask_h / 2)
        rot_matrix = cv2.getRotationMatrix2D(center_rot, rotation, 1.0)
        cos_a = abs(rot_matrix[0, 0])
        sin_a = abs(rot_matrix[0, 1])
        new_w = int(mask_h * sin_a + mask_w * cos_a)
        new_h = int(mask_w * sin_a + mask_h * cos_a)
        rot_matrix[0, 2] += (new_w - mask_w) / 2
        rot_matrix[1, 2] += (new_h - mask_h) / 2
        shadow_mask = cv2.warpAffine(shadow_mask, rot_matrix, (new_w, new_h),
                                     borderValue=0, flags=cv2.INTER_LINEAR)
        mask_w = new_w
        mask_h = new_h

    if shadow_mask.max() > 0:
        shadow_mask = shadow_mask / shadow_mask.max()

    shadow_top_left_x = int(shadow_cx - mask_w / 2)
    shadow_top_left_y = int(shadow_cy - mask_h / 2)

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
    """Composite BGRA photo onto canvas with alpha blending."""
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]

    if canvas.shape[2] == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
        canvas[:, :, 3] = 255

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

    canvas_region = canvas[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32) / 255.0
    photo_region = photo[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w].astype(np.float32) / 255.0

    alpha = photo_region[:, :, 3:4]
    canvas_alpha = canvas_region[:, :, 3:4]

    result_rgb = photo_region[:, :, :3] * alpha + canvas_region[:, :, :3] * (1 - alpha)
    result_alpha = np.maximum(canvas_alpha, alpha)

    result = np.concatenate([result_rgb, result_alpha], axis=2)
    result = (np.clip(result, 0, 1) * 255).astype(np.uint8)

    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = result
    return canvas


# =============================================================================
# BACKGROUND GENERATION
# =============================================================================

def random_base_background(w, h):
    """Generate a random background with controlled brightness and saturation."""
    rand_val = random.random()

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
    textures_dir = Path(__file__).resolve().parent / "textures"

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
# PERSPECTIVE WARP
# =============================================================================

def apply_perspective_safe(canvas, corners_list):
    """Apply perspective transform that keeps all corners in bounds.

    Returns (warped_bgr, perspective_matrix, had_perspective).
    """
    h, w = canvas.shape[:2]
    max_strength = 0.05
    safety_margin = 15

    if canvas.shape[2] == 4:
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    else:
        canvas_bgr = canvas.copy()

    for strength in np.linspace(max_strength, 0.0, 25):
        max_disp = int(min(w, h) * strength)

        tl = (random.randint(-max_disp, 0), random.randint(-max_disp, 0))
        tr = (random.randint(0, max_disp), random.randint(-max_disp, 0))
        bl = (random.randint(-max_disp, 0), random.randint(0, max_disp))
        br = (random.randint(0, max_disp), random.randint(0, max_disp))

        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_pts = np.array([
            [tl[0], tl[1]],
            [w + tr[0], tr[1]],
            [w + br[0], h + br[1]],
            [bl[0], h + bl[1]],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        all_in_bounds = True
        for corners in corners_list:
            for i in range(4):
                pt = np.array([corners[i, 0], corners[i, 1], 1])
                result = M @ pt
                wx = result[0] / result[2]
                wy = result[1] / result[2]
                if (wx < safety_margin or wx > w - safety_margin or
                        wy < safety_margin or wy > h - safety_margin):
                    all_in_bounds = False

        if all_in_bounds:
            warped = cv2.warpPerspective(
                canvas_bgr, M, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(128, 128, 128),
            )
            return warped, M, True

    return canvas_bgr, np.eye(3), False


def transform_corners(corners, M):
    """Apply a 3×3 perspective matrix to an array of (x, y) corners."""
    warped = np.zeros_like(corners)
    for i in range(corners.shape[0]):
        pt = np.array([corners[i, 0], corners[i, 1], 1])
        result = M @ pt
        warped[i, 0] = result[0] / result[2]
        warped[i, 1] = result[1] / result[2]
    return warped


# =============================================================================
# PHOTO PREPARATION (shared between generators)
# =============================================================================

def load_and_prepare_photo(source_dir, target_width, target_height):
    """Load a random source photo, scale to target dimensions, add glare.

    Returns:
        photo (BGRA): the prepared photo
        orig_w, orig_h: dimensions before scaling (for shadow)
    """
    sources = list(Path(source_dir).glob('*.jpg')) + list(Path(source_dir).glob('*.jpeg'))
    if not sources:
        raise ValueError(f"No source images found in {source_dir}")

    photo = cv2.imread(str(random.choice(sources)))
    if photo is None:
        raise ValueError("Failed to load source photo")

    h_orig, w_orig = photo.shape[:2]
    scale = min(target_width / w_orig, target_height / h_orig)
    new_w = int(w_orig * scale)
    new_h = int(h_orig * scale)
    photo = cv2.resize(photo, (new_w, new_h))

    photo = fast_glare(photo)

    if photo.shape[2] == 3:
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
    photo[:, :, 3] = 255

    return photo, w_orig, h_orig, new_w, new_h


# =============================================================================
# DEBUG VISUALIZATION
# =============================================================================

def create_debug_image(img, corners_list):
    """Create debug image with corner overlays.

    corners_list: list of (4, 2) arrays in LL, UL, UR, LR order.
    """
    debug = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    names = ['LL', 'UL', 'UR', 'LR']

    for corners in corners_list:
        corners = np.array(corners, dtype=np.int32)
        cv2.polylines(debug, [corners], True, (255, 255, 255), 2)

        for i in range(4):
            pt = (int(corners[i, 0]), int(corners[i, 1]))
            cv2.circle(debug, pt, 10, colors[i], -1)
            cv2.putText(debug, names[i], (pt[0] + 12, pt[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

    return debug


# =============================================================================
# OVERLAP VALIDATION (used by detection generator only)
# =============================================================================

OVERLAP_THRESHOLD = 0.05


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
    """Compute intersection area of two convex polygons via rasterization."""
    try:
        canvas_size = 640
        mask1 = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        mask2 = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        pts1 = np.array([[int(round(c[0])), int(round(c[1]))] for c in poly1], dtype=np.int32)
        pts2 = np.array([[int(round(c[0])), int(round(c[1]))] for c in poly2], dtype=np.int32)

        cv2.fillPoly(mask1, [pts1], 1)
        cv2.fillPoly(mask2, [pts2], 1)

        overlap = np.count_nonzero(mask1 & mask2)
        return float(overlap)
    except Exception:
        return 0.0


def check_overlaps(placements, threshold=OVERLAP_THRESHOLD):
    """Check that no pair of photos overlaps more than threshold."""
    n = len(placements)
    if n <= 1:
        return True

    polys = []
    areas = []
    for p in placements:
        corners = get_rotated_polygon(p['width'], p['height'],
                                       p['center_x'], p['center_y'],
                                       p['rotation'])
        polys.append(corners)
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