#!/usr/bin/env python3
"""
Simple Data Generator - Fixed Version
=====================================

Issues fixed:
1. No artificial border - photos placed properly within canvas
2. Bounding box matches actual corners
3. Photos actually visible in output

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
EDGE_MARGIN = 40


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
    """Rotate photo."""
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
    
    return cv2.warpAffine(photo, M, (new_w, new_h), 
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, 
                          borderValue=(128, 128, 128))


def composite_photo_at_center(canvas, photo, cx, cy):
    """Composite photo onto canvas."""
    ph, pw = photo.shape[:2]
    top_left_x = int(cx - pw / 2)
    top_left_y = int(cy - ph / 2)
    
    # Calculate overlap regions
    dst_x1 = max(0, top_left_x)
    dst_y1 = max(0, top_left_y)
    dst_x2 = min(canvas.shape[1], top_left_x + pw)
    dst_y2 = min(canvas.shape[0], top_left_y + ph)
    
    src_x1 = dst_x1 - top_left_x
    src_y1 = dst_y1 - top_left_y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    
    copy_w = dst_x2 - dst_x1
    copy_h = dst_y2 - dst_y1
    
    if copy_w > 0 and copy_h > 0:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = photo[src_y1:src_y2, src_x1:src_x2]
    
    return canvas


def apply_perspective_simple(canvas, strength=0.1):
    """Apply subtle perspective warp - SIMPLE version."""
    h, w = canvas.shape[:2]
    
    # Small random offsets (5-10% of canvas size)
    max_disp = int(min(w, h) * strength)
    
    # Random direction
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
    warped = cv2.warpPerspective(canvas, M, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(128, 128, 128))
    
    return warped, M


def pack_photos_simple(canvas_size):
    """Simple photo packing - returns list of placements."""
    num_photos = random.randint(NUM_PHOTOS_MIN, NUM_PHOTOS_MAX)
    placements = []
    
    if num_photos == 1:
        # Single photo in center
        size = random.randint(PHOTO_SIZE_MIN, PHOTO_SIZE_MAX)
        aspect = random.uniform(0.6, 1.5)
        height = int(size * aspect)
        width = int(size)
        
        cx = canvas_size / 2 + random.uniform(-50, 50)
        cy = canvas_size / 2 + random.uniform(-50, 50)
        rotation = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        
        placements.append({
            'width': width,
            'height': height,
            'center_x': cx,
            'center_y': cy,
            'rotation': rotation
        })
    
    elif num_photos == 2:
        # Two photos side by side or stacked
        size = random.randint(PHOTO_SIZE_MIN, min(350, PHOTO_SIZE_MAX))
        
        # Horizontal or vertical
        if random.random() < 0.5:  # Horizontal
            width = int(size * 0.9)
            height = int(size * random.uniform(0.8, 1.2))
            
            cx1 = canvas_size / 3 + random.uniform(-30, 30)
            cy1 = canvas_size / 2 + random.uniform(-50, 50)
            cx2 = 2 * canvas_size / 3 + random.uniform(-30, 30)
            cy2 = canvas_size / 2 + random.uniform(-50, 50)
        else:  # Vertical
            width = int(size * random.uniform(0.8, 1.2))
            height = int(size * 0.9)
            
            cx1 = canvas_size / 2 + random.uniform(-50, 50)
            cy1 = canvas_size / 3 + random.uniform(-30, 30)
            cx2 = canvas_size / 2 + random.uniform(-50, 50)
            cy2 = 2 * canvas_size / 3 + random.uniform(-30, 30)
        
        placements.append({
            'width': width, 'height': height,
            'center_x': cx1, 'center_y': cy1,
            'rotation': random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        })
        placements.append({
            'width': width, 'height': height,
            'center_x': cx2, 'center_y': cy2,
            'rotation': random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        })
    
    else:  # 3 or 4 photos
        # Grid layout
        cols = 2
        rows = 2 if num_photos == 4 else random.choice([2, 3])
        
        cell_w = (canvas_size - 2 * EDGE_MARGIN) / cols
        cell_h = (canvas_size - 2 * EDGE_MARGIN) / rows
        
        for i in range(num_photos):
            row = i // cols
            col = i % cols
            
            # Random size within cell
            width = int(cell_w * random.uniform(0.5, 0.8))
            height = int(cell_h * random.uniform(0.5, 0.8))
            
            # Center within cell with some randomness
            cx = EDGE_MARGIN + (col + 0.5) * cell_w + random.uniform(-cell_w*0.2, cell_w*0.2)
            cy = EDGE_MARGIN + (row + 0.5) * cell_h + random.uniform(-cell_h*0.2, cell_h*0.2)
            
            placements.append({
                'width': width, 'height': height,
                'center_x': cx, 'center_y': cy,
                'rotation': random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
            })
    
    return placements


def generate_image(source_dir, output_dir, num=1):
    """Generate a single image with photos and labels."""
    sources = list(Path(source_dir).glob('*.jpg')) + list(Path(source_dir).glob('*.jpeg'))
    if not sources:
        raise ValueError(f"No source images found in {source_dir}")
    
    # Create canvas with gradient background
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    for y in range(CANVAS_SIZE):
        val = int(160 + 40 * y / CANVAS_SIZE)
        canvas[y, :] = [val, val, val]
    
    # Pack photos
    placements = pack_photos_simple(CANVAS_SIZE)
    
    photos_data = []
    
    for placement in placements:
        # Load and resize photo
        photo_path = random.choice(sources)
        photo = cv2.imread(str(photo_path))
        if photo is None:
            continue
        
        h_orig, w_orig = photo.shape[:2]
        scale = min(placement['width'] / w_orig, placement['height'] / h_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        photo = cv2.resize(photo, (new_w, new_h))
        
        # Rotate
        photo = rotate_photo(photo, placement['rotation'])
        
        # Composite
        canvas = composite_photo_at_center(canvas, photo, 
                                          placement['center_x'], 
                                          placement['center_y'])
        
        # Calculate corners
        corners = get_rotated_polygon(new_w, new_h, 
                                     placement['center_x'], 
                                     placement['center_y'], 
                                     placement['rotation'])
        
        photos_data.append({
            'corners': corners,
            'rotation': placement['rotation']
        })
    
    # Apply subtle perspective
    perspective_strength = random.uniform(0.02, 0.08)  # 2-8%
    canvas, persp_M = apply_perspective_simple(canvas, perspective_strength)
    
    # Transform corners through perspective
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
        
        # Bounding box from corners
        min_x = np.min(corners[:, 0])
        max_x = np.max(corners[:, 0])
        min_y = np.min(corners[:, 1])
        max_y = np.max(corners[:, 1])
        
        # YOLO format: class x_center y_center width height (normalized)
        x_center = ((min_x + max_x) / 2) / CANVAS_SIZE
        y_center = ((min_y + max_y) / 2) / CANVAS_SIZE
        width = (max_x - min_x) / CANVAS_SIZE
        height = (max_y - min_y) / CANVAS_SIZE
        
        # Clip to valid range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0.01, min(1, width))
        height = max(0.01, min(1, height))
        
        det_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Pose label: bbox + corners
        corners_yolo = " ".join([
            f"{corners[i, 0]/CANVAS_SIZE:.6f} {corners[i, 1]/CANVAS_SIZE:.6f} 2"
            for i in range(4)
        ])
        pose_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {corners_yolo}")
    
    return canvas, final_photos, det_labels, pose_labels


def create_debug_image(img, photos):
    """Create debug image with corner overlays."""
    debug = img.copy()
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    names = ['LL', 'UL', 'UR', 'LR']
    
    for photo in photos:
        corners = photo['corners']
        pts = corners.astype(np.int32)
        
        # Draw polygon
        cv2.polylines(debug, [pts], True, (255, 255, 255), 2)
        
        # Draw corners
        for i in range(4):
            pt = (int(corners[i, 0]), int(corners[i, 1]))
            cv2.circle(debug, pt, 8, colors[i], -1)
            cv2.putText(debug, names[i], (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
    
    return debug


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate training data')
    parser.add_argument('--source', default='./images', help='Source images directory')
    parser.add_argument('--output', default='./data/examples_fixed', help='Output directory')
    parser.add_argument('--count', type=int, default=10, help='Number of images to generate')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.count} images...")
    print(f"Source: {args.source}")
    print(f"Output: {output_dir}")
    print()
    
    start_time = time.time()
    
    for i in range(args.count):
        img_start = time.time()
        
        img, photos, det_labels, pose_labels = generate_image(args.source, output_dir)
        
        # Save image
        img_path = output_dir / f"example_{i+1:02d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # Save debug image
        debug = create_debug_image(img, photos)
        debug_path = output_dir / f"example_{i+1:02d}_debug.jpg"
        cv2.imwrite(str(debug_path), debug)
        
        # Save labels
        det_path = output_dir / f"example_{i+1:02d}_det.txt"
        with open(det_path, 'w') as f:
            f.write('\n'.join(det_labels))
        
        pose_path = output_dir / f"example_{i+1:02d}_pose.txt"
        with open(pose_path, 'w') as f:
            f.write('\n'.join(pose_labels))
        
        img_time = time.time() - img_start
        print(f"  {i+1:2d}/{args.count}: {img_time:.1f}s ({len(photos)} photos)")
    
    total_time = time.time() - start_time
    print(f"\n✅ Done! {args.count} images in {total_time:.1f}s")
    print(f"📂 {output_dir.absolute()}")


if __name__ == "__main__":
    main()
