#!/usr/bin/env python3
"""
Visual Corner Verification - Shows corner labels on generated images
"""
import cv2
import numpy as np
from pathlib import Path

# Load example and its pose label
example_dir = Path('/Users/krys.petrie/dev/photo-pose-detector/data/examples_v34')

for i in range(1, 6):
    img = cv2.imread(str(example_dir / f'example_{i:02d}.jpg'))
    if img is None:
        continue
    
    label_path = example_dir / f'example_{i:02d}_pose.txt'
    if not label_path.exists():
        continue
    
    output = img.copy()
    
    with open(label_path) as f:
        lines = f.readlines()
    
    print(f"\nexample_{i:02d}: {len(lines)} photos")
    
    for photo_idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 17:  # 5 + 12 = 17 minimum
            continue
        
        # Parse corners: 8 values per keypoint (x, y, visible)
        corner_names = ['LL', 'UL', 'UR', 'LR']
        corners = []
        
        for kp_idx in range(4):
            kp_x = float(parts[5 + kp_idx*3]) * 640
            kp_y = float(parts[5 + kp_idx*3 + 1]) * 640
            corners.append((kp_x, kp_y))
        
        print(f"  Photo {photo_idx}:")
        for name, (x, y) in zip(corner_names, corners):
            print(f"    {name}: ({x:.1f}, {y:.1f})")
        
        # Draw polygon connecting corners
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][photo_idx % 4]
        pts = np.array([[int(x), int(y)] for x, y in corners], dtype=np.int32)
        
        # Draw the bounding box
        x_coords = [c[0] for c in corners]
        y_coords = [c[1] for c in corners]
        x1 = int(min(x_coords))
        y1 = int(min(y_coords))
        x2 = int(max(x_coords))
        y2 = int(max(y_coords))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw lines connecting corners (polygon)
        cv2.polylines(output, [pts], True, color, 2)
        
        # Draw filled circles at each corner
        for name, (x, y) in zip(corner_names, corners):
            cv2.circle(output, (int(x), int(y)), 8, color, -1)
            cv2.putText(output, name, (int(x)+12, int(y)-12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add photo index in center
        cx = sum(x_coords) / 4
        cy = sum(y_coords) / 4
        cv2.putText(output, f"P{photo_idx}", (int(cx)-15, int(cy)-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imwrite(str(example_dir / f'example_{i:02d}_corners_check.jpg'), output)

print("\n✓ Saved corner check images")
