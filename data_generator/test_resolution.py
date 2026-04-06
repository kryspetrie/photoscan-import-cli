#!/usr/bin/env python3
"""Test script to verify photo placement, resolution, and quality."""
import sys
sys.path.insert(0, '/Users/krys.petrie/dev/photo-pose-detector/data_generator')

from pathlib import Path
from generate_dataset import render_scene, Config
from PIL import Image
import numpy as np

# Find source images
source_dir = Path('/Users/krys.petrie/dev/photo-pose-detector/data_generator/images')
sources = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
print(f"Found {len(sources)} source images")

if len(sources) < 5:
    print("ERROR: Need at least 5 source images")
    sys.exit(1)

# Test configurations
test_cases = [
    (1, "single"),
    (3, "three"),
    (6, "six"),
    (9, "nine"),
]

config = Config(
    min_photo_size=200,
    max_photo_size=550,
    num_photos_min=1,
    num_photos_max=9
)

for num_photos, name in test_cases:
    print(f"\n{'='*60}")
    print(f"Testing with {num_photos} photos...")
    
    image, infos, canvas_size = render_scene(config, sources, requested_photos=num_photos)
    
    canvas_w, canvas_h = canvas_size
    print(f"  Canvas size: {canvas_w}x{canvas_h}")
    print(f"  Output shape: {image.shape}")
    print(f"  Photos placed: {len(infos)}")
    
    # Check aspect ratio
    aspect_ratio = canvas_w / canvas_h
    print(f"  Aspect ratio: {aspect_ratio:.4f} (should be 1.3333 for 4:3)")
    
    # Check minimum resolution
    min_w, min_h = 2000, 3000
    if canvas_w >= min_w and canvas_h >= min_h:
        print(f"  ✓ Resolution OK (>= {min_w}x{min_h})")
    else:
        print(f"  ✗ Resolution too small: {canvas_w}x{canvas_h}")
    
    if abs(aspect_ratio - 4/3) < 0.01:
        print(f"  ✓ Aspect ratio OK (4:3)")
    else:
        print(f"  ✗ Aspect ratio wrong")
    
    if len(infos) >= num_photos:
        print(f"  ✓ All {num_photos} photos placed")
    else:
        print(f"  ✗ Only placed {len(infos)}/{num_photos} photos")
    
    # Calculate content fill
    if infos:
        # Get bounding box of all photos
        all_x = []
        all_y = []
        for info in infos:
            for kx, ky in info['keypoints']:
                all_x.append(kx)
                all_y.append(ky)
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        content_area = (max_x - min_x) * (max_y - min_y)
        canvas_area = canvas_w * canvas_h
        fill_percent = content_area / canvas_area * 100
        
        print(f"  Content fill: {fill_percent:.1f}%")
        if fill_percent > 30:
            print(f"  ✓ Good fill")
        else:
            print(f"  ✗ Poor fill")
    
    # Check colors
    img_mean = image.mean(axis=(0, 1))
    img_std = image.std(axis=(0, 1))
    print(f"  Colors: R={img_mean[0]:.1f}, G={img_mean[1]:.1f}, B={img_mean[2]:.1f}")
    print(f"  Contrast: {img_std.mean():.1f}")
    
    # Save test image
    out_path = Path('test_output') / f'test_{name}.png'
    out_path.parent.mkdir(exist_ok=True)
    img_pil = Image.fromarray(image.astype('uint8'))
    img_pil.save(out_path)
    print(f"  Saved: {out_path}")
    
    # File size
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  File size: {size_mb:.1f} MB")

print(f"\n{'='*60}")
print("Done!")
