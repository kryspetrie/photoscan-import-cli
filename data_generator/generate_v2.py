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

# Configuration
CANVAS_SIZE = 640
PHOTO_SIZE_MIN = 180
PHOTO_SIZE_MAX = 480
ROTATION_RANGE = 30
NUM_PHOTOS_MIN = 1
NUM_PHOTOS_MAX = 4
EDGE_MARGIN = 50


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


def pack_photos_simple(canvas_size):
    """Simple photo packing - ensures photos reach canvas edges."""
    num_photos = random.randint(NUM_PHOTOS_MIN, NUM_PHOTOS_MAX)
    placements = []
    
    if num_photos == 1:
        size = random.randint(PHOTO_SIZE_MIN, PHOTO_SIZE_MAX)
        aspect = random.uniform(0.6, 1.5)
        height = int(size * aspect)
        width = int(size)
        
        cx = canvas_size / 2 + random.uniform(-80, 80)
        cy = canvas_size / 2 + random.uniform(-80, 80)
        rotation = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        
        placements.append({
            'width': width, 'height': height,
            'center_x': cx, 'center_y': cy,
            'rotation': rotation
        })
    
    elif num_photos == 2:
        size = random.randint(PHOTO_SIZE_MIN, min(400, PHOTO_SIZE_MAX))
        
        if random.random() < 0.5:  # Horizontal
            w1 = int(size * random.uniform(0.8, 1.0))
            h1 = int(size * random.uniform(0.8, 1.2))
            w2 = int(size * random.uniform(0.8, 1.0))
            h2 = int(size * random.uniform(0.8, 1.2))
            
            cx1 = canvas_size * 0.3 + random.uniform(-40, 40)
            cy1 = canvas_size * 0.5 + random.uniform(-60, 60)
            cx2 = canvas_size * 0.7 + random.uniform(-40, 40)
            cy2 = canvas_size * 0.5 + random.uniform(-60, 60)
        else:  # Vertical
            w1 = int(size * random.uniform(0.8, 1.2))
            h1 = int(size * random.uniform(0.8, 1.0))
            w2 = int(size * random.uniform(0.8, 1.2))
            h2 = int(size * random.uniform(0.8, 1.0))
            
            cx1 = canvas_size * 0.5 + random.uniform(-60, 60)
            cy1 = canvas_size * 0.3 + random.uniform(-40, 40)
            cx2 = canvas_size * 0.5 + random.uniform(-60, 60)
            cy2 = canvas_size * 0.7 + random.uniform(-40, 40)
        
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
    
    else:  # 3 or 4 photos
        cols = 2
        rows = 2 if num_photos == 4 else (2 if num_photos == 3 and random.random() < 0.5 else 3)
        
        cell_w = (canvas_size - 2 * EDGE_MARGIN) / cols
        cell_h = (canvas_size - 2 * EDGE_MARGIN) / rows
        
        positions = []
        for r in range(rows):
            for c in range(cols):
                positions.append((r, c))
        
        random.shuffle(positions)
        positions = positions[:num_photos]
        
        for row, col in positions:
            width = int(cell_w * random.uniform(0.6, 0.85))
            height = int(cell_h * random.uniform(0.6, 0.85))
            
            cx = EDGE_MARGIN + (col + 0.5) * cell_w + random.uniform(-cell_w*0.15, cell_w*0.15)
            cy = EDGE_MARGIN + (row + 0.5) * cell_h + random.uniform(-cell_h*0.15, cell_h*0.15)
            
            placements.append({
                'width': width, 'height': height,
                'center_x': cx, 'center_y': cy,
                'rotation': random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
            })
    
    return placements


def generate_image(source_dir):
    """Generate a single image."""
    sources = list(Path(source_dir).glob('*.jpg')) + list(Path(source_dir).glob('*.jpeg'))
    if not sources:
        raise ValueError(f"No source images found")
    
    # Create gradient background
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    for y in range(CANVAS_SIZE):
        val = int(140 + 60 * y / CANVAS_SIZE)
        canvas[y, :] = [val, val, val]
    
    # Pack photos
    placements = pack_photos_simple(CANVAS_SIZE)
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
        photo = rotate_photo(photo, placement['rotation'])
        canvas = composite_photo_at_center(canvas, photo, placement['center_x'], placement['center_y'])
        
        # Calculate corners
        corners = get_rotated_polygon(new_w, new_h, placement['center_x'], placement['center_y'], placement['rotation'])
        
        photos_data.append({
            'corners': corners,
            'rotation': placement['rotation']
        })
    
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
    parser.add_argument('--output', default='./data/examples_v2', help='Output dir')
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
