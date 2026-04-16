#!/usr/bin/env python3
"""
Batch Dataset Generator - Generate 5000 training images

Run in background:
    python generate_batch.py > generation_log.txt 2>&1 &
    
Check progress:
    tail -f generation_log.txt
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
    verify_corners_in_bounds,
    verify_trapezoid_shape,
    verify_perspective_subtle,
)

NUM_TRAIN = 4000
NUM_VAL = 1000
TOTAL = NUM_TRAIN + NUM_VAL
CANVAS_W = 1920
CANVAS_H = 1080
LOG_INTERVAL = 10
CHECKPOINT_INTERVAL = 100

def generate_single_image(source_images, image_index):
    random.seed(image_index + int(time.time() * 1000) % 10000)
    np.random.seed(image_index + int(time.time() * 1000) % 10000)
    
    bg = random_base_background(CANVAS_W, CANVAS_H)
    bg = apply_texture_overlay(bg)
    canvas = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
    
    placements = spiral_pack_photos(CANVAS_W, CANVAS_H)
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
        
        photo = fast_photo_manipulation(photo)
        photo = fast_glare(photo)
        photo = add_rgba_alpha(photo)
        photo = blur_alpha_edges(photo)
        photo = rotate_photo(photo, placement['rotation'])
        
        center_x = placement['center_x']
        center_y = placement['center_y']
        
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
    
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    photo_corners_list = [p['polygon'] for p in placed_photos]
    
    warped_canvas, global_corners, transform_matrix, content_bounds, warped_photo_corners = apply_global_perspective(
        canvas_bgr, CANVAS_W, CANVAS_H,
        photo_corners=photo_corners_list,
        crop_margin=250
    )
    
    final_photos = []
    for idx, photo_data in enumerate(placed_photos):
        warped_corners = warped_photo_corners[idx]
        final_photos.append({
            'corners': warped_corners,
            'rotation': photo_data['rotation']
        })
    
    return warped_canvas, final_photos

def save_image_and_label(img, photos, img_path, lbl_path):
    out_w, out_h = img.shape[1], img.shape[0]
    
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img.save(img_path, quality=90)
    
    with open(lbl_path, 'w') as f:
        for p in photos:
            kps = p['corners']
            x_center = sum(k[0] for k in kps) / 4 / out_w
            y_center = sum(k[1] for k in kps) / 4 / out_h
            width = (max(k[0] for k in kps) - min(k[0] for k in kps)) / out_w
            height = (max(k[1] for k in kps) - min(k[1] for k in kps)) / out_h
            
            line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            for kx, ky in kps:
                kx_clamped = max(0, min(1, kx / out_w))
                ky_clamped = max(0, min(1, ky / out_h))
                line += f" {kx_clamped:.6f} {ky_clamped:.6f} 2"
            line += "\n"
            f.write(line)

def main():
    print("=" * 60)
    print("BATCH DATASET GENERATOR (v31)")
    print("=" * 60)
    print(f"Total: {TOTAL} ({NUM_TRAIN} train, {NUM_VAL} val)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    base_dir = Path("../data")
    train_img_dir = base_dir / "images" / "train"
    train_lbl_dir = base_dir / "labels" / "train"
    val_img_dir = base_dir / "images" / "val"
    val_lbl_dir = base_dir / "labels" / "val"
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    source_dir = Path("./images")
    source_images = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.jpeg')) + \
                   list(source_dir.glob('*.png')) + list(source_dir.glob('*.webp'))
    
    print(f"Source images: {len(source_images)}")
    
    stats = {'total_photos': 0, 'total_time': 0}
    start_time = time.time()
    
    try:
        for i in range(TOTAL):
            img_start = time.time()
            is_train = i < NUM_TRAIN
            
            img, photos = generate_single_image(source_images, i)
            
            if is_train:
                img_path = train_img_dir / f"train_{i+1:06d}.jpg"
                lbl_path = train_lbl_dir / f"train_{i+1:06d}.txt"
            else:
                idx = i - NUM_TRAIN + 1
                img_path = val_img_dir / f"val_{idx:06d}.jpg"
                lbl_path = val_lbl_dir / f"val_{idx:06d}.txt"
            
            save_image_and_label(img, photos, img_path, lbl_path)
            
            img_time = time.time() - img_start
            stats['total_time'] += img_time
            stats['total_photos'] += len(photos)
            
            if (i + 1) % LOG_INTERVAL == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (TOTAL - i - 1) / rate / 3600 if rate > 0 else 0
                print(f"  [{i+1:5d}/{TOTAL}] {img_time:.1f}s/img | ETA: {eta:.1f}h | Photos: {stats['total_photos']}")
            
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                print(f"  ✓ Checkpoint at {i+1}")
        
        total_time = time.time() - start_time
        print(f"\n✅ COMPLETE in {total_time/3600:.2f}h")
        print(f"   Train: {NUM_TRAIN}, Val: {NUM_VAL}, Photos: {stats['total_photos']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
