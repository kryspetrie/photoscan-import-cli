#!/usr/bin/env python3
"""
Batch Dataset Generator v32 - Two-Model Architecture
====================================================

Generates 5000 training images that train BOTH YOLO models:
1. Detection Model: Axis-aligned bounding boxes
2. Pose Model: 4 corner keypoints (LL, UL, UR, LR)

SAME IMAGES → DIFFERENT LABELS

Run in background:
    python generate_batch.py > generation_log.txt 2>&1 &
    
Check progress:
    tail -f generation_log.txt

OUTPUT STRUCTURE
---------------
The generator creates BOTH label formats from the same images:

    data/
    ├── images/
    │   ├── train/          # 4000 JPEG images
    │   └── val/            # 1000 JPEG images
    │
    ├── detection/          # DETECTION MODEL DATA
    │   └── labels/
    │       ├── train/      # 4000 labels (5 columns: class x y w h)
    │       └── val/        # 1000 labels
    │
    └── pose/               # POSE MODEL DATA
        └── labels/
            ├── train/      # 4000 labels (13 columns: class x y w h + kps)
            └── val/        # 1000 labels

USAGE WITH YOLO TRAINING
------------------------
Detection Model:
    from dataset_detection.yaml
    
Pose Model:
    from dataset_pose.yaml

Both models use the SAME images but with DIFFERENT label files.

Author: Photo Pose Detector Project
Version: 32 - Two-Model Architecture
"""

import os
import sys
import time
import signal
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

# Import from generate_dataset
import random
from generate_dataset import (
    CONFIG,
    random_base_background,
    apply_texture_overlay,
    spiral_pack_photos,
    rotate_photo,
    composite_photo_at_center,
    apply_photo_shadow,
    add_rgba_alpha,
    blur_alpha_edges,
    fast_photo_manipulation,
    fast_glare,
    apply_global_perspective,
    generate_detection_label,
    generate_pose_label,
)


NUM_TRAIN = 4000
NUM_VAL = 1000
TOTAL = NUM_TRAIN + NUM_VAL
LOG_INTERVAL = 20
CHECKPOINT_INTERVAL = 100


def generate_single_image(source_images, image_index):
    """Generate a single composite image with multiple photos."""
    random.seed(image_index + int(time.time() * 1000) % 10000)
    np.random.seed(image_index + int(time.time() * 1000) % 10000)
    
    canvas_size = CONFIG['CANVAS_SIZE']  # 640
    
    # Generate background
    bg = random_base_background(canvas_size, canvas_size)
    bg = apply_texture_overlay(bg)
    canvas = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
    
    # Pack photos (1-4 per image)
    placements = spiral_pack_photos(canvas_size, canvas_size)
    placed_photos = []
    
    for placement in placements:
        photo = cv2.imread(str(random.choice(source_images)))
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
        
        center_x = placement['center_x']
        center_y = placement['center_y']
        
        # Shadow
        shadow_params = placement.get('shadow_params', {})
        if shadow_params:
            canvas = apply_photo_shadow(
                canvas, photo, center_x, center_y,
                shadow_params['offset_x'], shadow_params['offset_y'],
                shadow_params['blur_sigma'], shadow_params['opacity'],
                rotation=placement['rotation']
            )
        
        canvas = composite_photo_at_center(canvas, photo, center_x, center_y)
        
        placed_photos.append({
            'polygon': placement['polygon'].copy(),
            'rotation': placement['rotation'],
        })
    
    # Apply global perspective warp
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    photo_corners_list = [p['polygon'] for p in placed_photos]
    
    warped_canvas, global_corners, transform_matrix, content_bounds, warped_photo_corners = apply_global_perspective(
        canvas_bgr, canvas_size, canvas_size,
        photo_corners=photo_corners_list,
        crop_margin=CONFIG['CROP_MARGIN']
    )
    
    final_photos = []
    for idx, photo_data in enumerate(placed_photos):
        warped_corners = warped_photo_corners[idx]
        final_photos.append({
            'corners': warped_corners,
            'rotation': photo_data['rotation']
        })
    
    return warped_canvas, final_photos


def save_outputs(img, photos, img_path, det_lbl_path, pose_lbl_path):
    """Save image and BOTH label formats."""
    out_w, out_h = img.shape[1], img.shape[0]
    
    # Save image
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img.save(img_path, quality=90)
    
    # Generate detection label (5 columns)
    det_label = generate_detection_label(photos, out_w, out_h)
    with open(det_lbl_path, 'w') as f:
        f.write(det_label)
    
    # Generate pose label (13 columns)
    pose_label = generate_pose_label(photos, out_w, out_h)
    with open(pose_lbl_path, 'w') as f:
        f.write(pose_label)


def main():
    print("=" * 70)
    print("BATCH DATASET GENERATOR v32 - TWO-MODEL ARCHITECTURE")
    print("=" * 70)
    print(f"Total: {TOTAL} ({NUM_TRAIN} train, {NUM_VAL} val)")
    print(f"Canvas: {CONFIG['CANVAS_SIZE']}x{CONFIG['CANVAS_SIZE']} | Photos: {CONFIG['NUM_PHOTOS_MIN']}-{CONFIG['NUM_PHOTOS_MAX']}")
    print(f"Perspective: {CONFIG['PERSPECTIVE_STRENGTH_MIN']*100:.0f}%-{CONFIG['PERSPECTIVE_STRENGTH_MAX']*100:.0f}%")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Directory structure for two models
    base_dir = Path("../data")
    
    # Shared images
    img_train_dir = base_dir / "images" / "train"
    img_val_dir = base_dir / "images" / "val"
    
    # Detection model labels
    det_train_dir = base_dir / "detection" / "labels" / "train"
    det_val_dir = base_dir / "detection" / "labels" / "val"
    
    # Pose model labels
    pose_train_dir = base_dir / "pose" / "labels" / "train"
    pose_val_dir = base_dir / "pose" / "labels" / "val"
    
    # Create directories
    for d in [img_train_dir, img_val_dir, 
              det_train_dir, det_val_dir,
              pose_train_dir, pose_val_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Source images
    source_dir = Path("./images")
    source_images = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.jpeg')) + \
                   list(source_dir.glob('*.png')) + list(source_dir.glob('*.webp'))
    
    print(f"\n📁 Source images: {len(source_images)}")
    print(f"\n📂 Output structure:")
    print(f"   Images (shared):")
    print(f"      {img_train_dir}/")
    print(f"      {img_val_dir}/")
    print(f"   Detection labels: {det_train_dir.parent.parent}/")
    print(f"   Pose labels: {pose_train_dir.parent.parent}/")
    
    stats = {'total_photos': 0, 'total_time': 0}
    start_time = time.time()
    
    try:
        for i in range(TOTAL):
            img_start = time.time()
            is_train = i < NUM_TRAIN
            
            img, photos = generate_single_image(source_images, i)
            
            if is_train:
                img_path = img_train_dir / f"train_{i+1:06d}.jpg"
                det_lbl_path = det_train_dir / f"train_{i+1:06d}.txt"
                pose_lbl_path = pose_train_dir / f"train_{i+1:06d}.txt"
            else:
                idx = i - NUM_TRAIN + 1
                img_path = img_val_dir / f"val_{idx:06d}.jpg"
                det_lbl_path = det_val_dir / f"val_{idx:06d}.txt"
                pose_lbl_path = pose_val_dir / f"val_{idx:06d}.txt"
            
            save_outputs(img, photos, img_path, det_lbl_path, pose_lbl_path)
            
            img_time = time.time() - img_start
            stats['total_time'] += img_time
            stats['total_photos'] += len(photos)
            
            if (i + 1) % LOG_INTERVAL == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (TOTAL - i - 1) / rate / 3600 if rate > 0 else 0
                avg_time = stats['total_time'] / (i + 1)
                print(f"  [{i+1:5d}/{TOTAL}] {avg_time:.2f}s/img | ETA: {eta:.1f}h | Photos: {stats['total_photos']}")
            
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                elapsed = time.time() - start_time
                print(f"  ✓ Checkpoint at {i+1} ({elapsed/60:.1f} min elapsed)")
        
        total_time = time.time() - start_time
        avg_time = stats['total_time'] / TOTAL
        
        print(f"\n{'='*70}")
        print("✅ COMPLETE")
        print(f"{'='*70}")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Average: {avg_time:.2f}s per image")
        print(f"\n   Train images: {NUM_TRAIN}")
        print(f"   Val images: {NUM_VAL}")
        print(f"   Total photos: {stats['total_photos']}")
        
        print(f"\n📂 Generated files:")
        print(f"   Images: {img_train_dir.parent}/*/images/*/")
        print(f"   Detection labels: {det_train_dir.parent}/")
        print(f"   Pose labels: {pose_train_dir.parent}/")
        
        print(f"\n⚠️  NEXT STEP: Create dataset YAML files and train models!")
        print(f"   See ../training/create_yaml_files.py")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user at image {i+1}")
        print(f"   Generated: {i} images before interruption")
        print(f"   Time elapsed: {(time.time()-start_time)/60:.1f} minutes")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
