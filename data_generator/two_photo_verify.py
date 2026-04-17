#!/usr/bin/env python3
"""
Two-Photo Corner Verification
============================

This creates images WITH visible corner markers to definitively verify
that our coordinate transformation is correct.

Method:
1. Load an example image
2. Draw colored corner markers at the computed positions
3. If markers align with photo corners, math is correct
"""

import sys
from pathlib import Path
import numpy as np
import cv2


def create_two_photo_verification(img_path, pose_path, out_path):
    """Draw corner markers on actual image positions."""
    
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    debug = img.copy()
    
    if not Path(pose_path).exists():
        return False
    
    with open(pose_path) as f:
        for obj_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            
            # Draw corner markers
            colors = [(255, 0, 0), (0, 255, 255), (0, 200, 0), (0, 100, 255)]
            labels = ['LL', 'UL', 'UR', 'LR']
            markers = ['X', 'X', 'X', 'X']
            
            for i in range(4):
                kx = float(parts[5 + i*3]) * w
                ky = float(parts[6 + i*3]) * h
                
                x, y = int(kx), int(ky)
                
                # Draw X marker
                size = 15
                color = colors[i]
                cv2.line(debug, (x-size, y-size), (x+size, y+size), color, 3)
                cv2.line(debug, (x+size, y-size), (x-size, y+size), color, 3)
                
                # Draw label
                cv2.putText(debug, labels[i], (x + size + 5, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add border
    cv2.rectangle(debug, (0, 0), (w-1, h-1), (255, 255, 255), 4)
    
    # Add title
    cv2.putText(debug, f"Corner Verification - {Path(img_path).name}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(str(out_path), debug)
    return True


def main():
    data_dir = Path("../data/examples_v33")
    
    print("Creating two-photo verification images...\n")
    
    for i in range(1, 4):  # First 3 images
        img_path = data_dir / f"example_{i:02d}.jpg"
        pose_path = data_dir / f"example_{i:02d}_pose.txt"
        out_path = data_dir / f"corner_verify_{i:02d}.png"
        
        if create_two_photo_verification(img_path, pose_path, out_path):
            print(f"  ✓ {out_path.name}")
    
    print("\nIf X markers are on photo corners, math is correct!")
    print("If X markers are off, there's still a bug.")


if __name__ == "__main__":
    main()
