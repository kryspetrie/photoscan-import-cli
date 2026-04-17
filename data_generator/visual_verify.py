#!/usr/bin/env python3
"""
Visual Verification Script
========================

Creates annotated images with corner markers overlaid on the actual photos
to verify alignment between computed corners and visual corners.
"""

import sys
from pathlib import Path
import numpy as np
import cv2


def verify_image_visual(img_path, pose_path, out_path):
    """Create annotated image showing bbox and corners on actual photo."""
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: {img_path}")
        return False
    
    h, w = img.shape[:2]
    debug = img.copy()
    
    # Load pose label
    if not Path(pose_path).exists():
        print(f"Error: {pose_path}")
        return False
    
    with open(pose_path) as f:
        for obj_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            
            # Parse bbox
            cx = float(parts[1]) * w
            cy = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h
            
            # Draw bbox
            x1 = int(cx - bw/2)
            y1 = int(cy - bh/2)
            x2 = int(cx + bw/2)
            y2 = int(cy + bh/2)
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Parse keypoints
            kps = []
            for i in range(4):
                kx = float(parts[5 + i*3]) * w
                ky = float(parts[6 + i*3]) * h
                vis = int(parts[7 + i*3])
                kps.append((kx, ky, vis))
            
            # Colors for LL, UL, UR, LR
            colors = [(255, 0, 0), (0, 255, 255), (0, 200, 0), (0, 100, 255)]
            labels = ['LL', 'UL', 'UR', 'LR']
            
            # Draw corner dots
            for i, ((kx, ky, vis), color, label) in enumerate(zip(kps, colors, labels)):
                x, y = int(kx), int(ky)
                if 0 <= x < w and 0 <= y < h:
                    # Draw filled circle
                    cv2.circle(debug, (x, y), 10, color, -1)
                    # Draw white border
                    cv2.circle(debug, (x, y), 10, (255, 255, 255), 2)
                    # Draw label
                    cv2.putText(debug, label, (x + 12, y - 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw quadrilateral connecting corners in order
            pts = np.array([[int(kx), int(ky)] for kx, ky, vis in kps], dtype=np.int32)
            cv2.polylines(debug, [pts], isClosed=True, color=(255, 255, 0), thickness=1)
            
            # Label number
            cv2.putText(debug, f"#{obj_num}", (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add border
    cv2.rectangle(debug, (0, 0), (w-1, h-1), (255, 255, 255), 4)
    
    # Add legend
    legend_y = h - 30
    legend_x = 10
    cv2.putText(debug, "Green=Bbox Blue=LL Yellow=UL Green=UR Orange=LR", 
               (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite(str(out_path), debug)
    return True


def main():
    data_dir = Path("../data/examples_v33")
    
    print("Creating visual verification images...\n")
    
    created = 0
    for i in range(1, 11):
        img_path = data_dir / f"example_{i:02d}.jpg"
        pose_path = data_dir / f"example_{i:02d}_pose.txt"
        out_path = data_dir / f"visual_verify_{i:02d}.png"
        
        if img_path.exists() and pose_path.exists():
            if verify_image_visual(img_path, pose_path, out_path):
                # Count objects
                with open(pose_path) as f:
                    num_objects = len([l for l in f if l.strip()])
                print(f"  ✓ visual_verify_{i:02d}.png ({num_objects} objects)")
                created += 1
    
    print(f"\nCreated {created} verification images")
    print("\nCheck the images to verify:")
    print("  1. Green bbox surrounds each photo tightly")
    print("  2. Corner dots (LL, UL, UR, LR) are on photo corners")
    print("  3. Yellow quadrilateral connects corners in order")


if __name__ == "__main__":
    main()
