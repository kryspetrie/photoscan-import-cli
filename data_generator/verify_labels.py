#!/usr/bin/env python3
"""
Verify Generated Images
======================

Creates annotated versions of generated images showing:
- Green bbox: Detection bounding box
- Corner dots: LL, UL, UR, LR keypoints
"""

import sys
from pathlib import Path
import numpy as np
import cv2

def verify_image(img_path, det_path, pose_path, out_path):
    """Create annotated image with bbox and corners."""
    
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error loading {img_path}")
        return False
    
    h, w = img.shape[:2]
    debug = img.copy()
    
    # Load detection label
    if Path(det_path).exists():
        with open(det_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, cx, cy, bw, bh = map(float, parts[:5])
                    
                    # Convert to pixels
                    cx_px = int(cx * w)
                    cy_px = int(cy * h)
                    bw_px = int(bw * w)
                    bh_px = int(bh * h)
                    
                    x1 = cx_px - bw_px // 2
                    y1 = cy_px - bh_px // 2
                    x2 = x1 + bw_px
                    y2 = y1 + bh_px
                    
                    # Draw bbox (green)
                    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Load pose label
    if Path(pose_path).exists():
        with open(pose_path) as f:
            line_num = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 17:  # class + bbox + 4 keypoints * 3
                    # Parse keypoints
                    kps = []
                    for i in range(4):
                        kx = float(parts[5 + i*3])
                        ky = float(parts[6 + i*3])
                        vis = int(parts[7 + i*3])
                        kps.append((kx * w, ky * h, vis))
                    
                    # Draw corner dots with labels
                    colors = [(255, 0, 0), (0, 255, 255), (0, 255, 0), (0, 128, 255)]  # B, Y, G, O
                    labels = ['LL', 'UL', 'UR', 'LR']
                    
                    # Sort by y then x to determine order
                    kp_sorted = sorted(enumerate(kps), key=lambda x: (x[1][1], x[1][0]))
                    
                    for rank, (idx, (kx, ky, vis)) in enumerate(kp_sorted):
                        x, y = int(kx), int(ky)
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(debug, (x, y), 6, colors[rank], -1)
                            cv2.putText(debug, labels[rank], (x + 8, y + 8),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Draw lines connecting corners in order
                    pts = np.array([[int(kx), int(ky)] for kx, ky, vis in kps], dtype=np.int32)
                    cv2.polylines(debug, [pts], isClosed=True, color=(255, 255, 0), thickness=1)
    
    # Add border
    cv2.rectangle(debug, (0, 0), (w-1, h-1), (255, 255, 255), 3)
    
    # Save
    cv2.imwrite(str(out_path), debug)
    return True


def main():
    data_dir = Path("../data/examples_v33")
    
    print("Creating annotated images for verification...\n")
    
    for i in range(1, 11):
        img_path = data_dir / f"example_{i:02d}.jpg"
        det_path = data_dir / f"example_{i:02d}_det.txt"
        pose_path = data_dir / f"example_{i:02d}_pose.txt"
        out_path = data_dir / f"verify_{i:02d}.png"
        
        if img_path.exists():
            if verify_image(img_path, det_path, pose_path, out_path):
                print(f"  Created: {out_path.name}")
    
    print("\nAnnotation key:")
    print("  Green box: Detection bounding box")
    print("  Blue dot: LL (Lower-Left)")
    print("  Yellow dot: UL (Upper-Left)")
    print("  Green dot: UR (Upper-Right)")
    print("  Orange dot: LR (Lower-Right)")
    print("  Cyan lines: Quadrilateral from corners")


if __name__ == "__main__":
    main()
