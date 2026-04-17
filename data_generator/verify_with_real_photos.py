#!/usr/bin/env python3
"""
Verification script using real photographs.

This script:
1. Loads real photos from the images/ folder
2. Applies the exact pipeline transformations from generate_dataset.py
3. Overlays corner markers on the OUTPUT image at calculated positions
4. Saves both original photo corners AND calculated corners for comparison

The key verification: do the overlaid corners align with actual photo edges?
"""

import cv2
import numpy as np
import os
from pathlib import Path
import random

# Import the core pipeline functions from generate_dataset.py
import sys
sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import CONFIG


# =============================================================================
# CONFIGURATION (must match generate_dataset.py)
# =============================================================================

CANVAS_SIZE = CONFIG['CANVAS_SIZE']           # 640
PADDING = CONFIG['CANVAS_PADDING']            # 300
PADDED_CANVAS_SIZE = CANVAS_SIZE + 2 * PADDING  # 1240
CROP_MARGIN = CONFIG['CROP_MARGIN']           # 60
OUTPUT_SIZE = CONFIG['CANVAS_SIZE']           # 640


# =============================================================================
# COPY THE EXACT PIPELINE FUNCTIONS FROM generate_dataset.py
# =============================================================================

def rotate_photo(photo, angle):
    """Rotate a photo by specified degrees."""
    h, w = photo.shape[:2]
    
    if abs(angle) < 1:
        return photo, (w, h), None
    
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(
        photo, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128, 0)
    )
    
    return rotated, (new_w, new_h), M


def composite_photo_at_center(canvas, photo, cx, cy):
    """Composite photo onto canvas with center at (cx, cy)."""
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    
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
        src_x2 = cw - dst_x1
        dst_x2 = cw
    if dst_y2 > ch:
        src_y2 = ch - dst_y1
        dst_y2 = ch
    
    copy_w = int(dst_x2 - dst_x1)
    copy_h = int(dst_y2 - dst_y1)
    
    if copy_w <= 0 or copy_h <= 0:
        return canvas, (top_left_x, top_left_y)
    
    canvas[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
        photo[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]
    
    return canvas, (top_left_x, top_left_y)


def apply_global_perspective(canvas, canvas_w, canvas_h, crop_margin=60):
    """Apply perspective warp to canvas."""
    max_disp = int(canvas_w * CONFIG['PERSPECTIVE_STRENGTH_MAX'])
    
    tl = (random.randint(-max_disp, 0), random.randint(-max_disp, 0))
    tr = (random.randint(0, max_disp), random.randint(-max_disp, 0))
    bl = (random.randint(-max_disp, 0), random.randint(0, max_disp))
    br = (random.randint(0, max_disp), random.randint(0, max_disp))
    
    src_pts = np.array([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]], dtype=np.float32)
    dst_pts = np.array([
        [tl[0], tl[1]],
        [canvas_w + tr[0], tr[1]],
        [canvas_w + br[0], canvas_h + br[1]],
        [bl[0], canvas_h + bl[1]]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(canvas, M, (canvas_w, canvas_h), 
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(128, 128, 128, 0))
    
    # Crop to output size
    result = warped[crop_margin:crop_margin+CANVAS_SIZE, 
                   crop_margin:crop_margin+CANVAS_SIZE]
    
    return result, (tl, tr, bl, br)


def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """
    Calculate the 4 corner coordinates of a rotated rectangle.
    Uses raw rotation matrix (M_raw) without canvas expansion offset.
    """
    if abs(rotation) < 1:
        return np.array([
            [center_x - width/2, center_y - height/2],
            [center_x + width/2, center_y - height/2],
            [center_x + width/2, center_y + height/2],
            [center_x - width/2, center_y + height/2]
        ], dtype=np.float32)
    
    photo_center = (width / 2, height / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    # Canvas offset: where the photo center is in the padded canvas
    canvas_offset = (center_x - width / 2, center_y - height / 2)
    
    # Original photo corners (before rotation)
    corners_photo = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Transform each corner
    corners_final = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        corners_final[i, 0] = canvas_offset[0] + rotated[0]
        corners_final[i, 1] = canvas_offset[1] + rotated[1]
    
    return corners_final


def apply_perspective_to_point(pt, tl, tr, bl, br, w, h):
    """Apply perspective warp transformation to a single point."""
    x, y = pt
    
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array([
        [tl[0], tl[1]],
        [w + tr[0], tr[1]],
        [w + br[0], h + br[1]],
        [bl[0], h + bl[1]]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    pt_h = np.array([x, y, 1.0])
    transformed = M @ pt_h
    return np.array([transformed[0] / transformed[2], transformed[1] / transformed[2]])


# =============================================================================
# MAIN VERIFICATION FUNCTION
# =============================================================================

def verify_with_real_photo(photo_path, rotation, center_x, center_y, output_path=None):
    """
    Apply the full pipeline to a real photo and overlay corner markers.
    
    Returns the corners in output space for verification.
    """
    # Load photo
    photo = cv2.imread(photo_path)
    if photo is None:
        raise FileNotFoundError(f"Could not load: {photo_path}")
    
    h, w = photo.shape[:2]
    
    print(f"\n  Input photo: {w}x{h}")
    print(f"  Rotation: {rotation}°")
    print(f"  Center: ({center_x}, {center_y})")
    
    # Step 1: Rotate
    rotated_photo, (rot_w, rot_h), M = rotate_photo(photo, rotation)
    print(f"  Rotated size: {rot_w}x{rot_h}")
    
    # Step 2: Create padded canvas and composite
    padded_canvas = np.zeros((PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE, 3), dtype=np.uint8)
    padded_canvas[:] = (128, 128, 128)  # Gray background
    canvas, _ = composite_photo_at_center(padded_canvas, rotated_photo, center_x, center_y)
    
    # Step 3: Apply perspective warp
    warped_canvas, persp_corners = apply_global_perspective(
        canvas, PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE, crop_margin=CROP_MARGIN
    )
    tl, tr, bl, br = persp_corners
    print(f"  Perspective offsets: TL={tl}, TR={tr}, BL={bl}, BR={br}")
    
    # Step 4: Get corners in placement space (before perspective)
    corners_placement = get_rotated_polygon(w, h, center_x, center_y, rotation)
    print(f"  Corners in placement space:")
    print(f"    LL: ({corners_placement[0,0]:.1f}, {corners_placement[0,1]:.1f})")
    print(f"    UL: ({corners_placement[1,0]:.1f}, {corners_placement[1,1]:.1f})")
    print(f"    UR: ({corners_placement[2,0]:.1f}, {corners_placement[2,1]:.1f})")
    print(f"    LR: ({corners_placement[3,0]:.1f}, {corners_placement[3,1]:.1f})")
    
    # Step 5: Transform corners through perspective warp
    corners_persp = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        corners_persp[i] = apply_perspective_to_point(
            corners_placement[i], tl, tr, bl, br, PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE
        )
    
    # Step 6: Crop and transform to output space
    crop_x = CROP_MARGIN
    crop_y = CROP_MARGIN
    crop_size = PADDED_CANVAS_SIZE - 2 * CROP_MARGIN
    
    corners_cropped = corners_persp.copy()
    corners_cropped[:, 0] -= crop_x
    corners_cropped[:, 1] -= crop_y
    
    # Step 7: Scale to output size
    scale = OUTPUT_SIZE / crop_size
    corners_output = corners_cropped * scale
    
    print(f"  Corners in output space (640x640):")
    print(f"    LL: ({corners_output[0,0]:.1f}, {corners_output[0,1]:.1f})")
    print(f"    UL: ({corners_output[1,0]:.1f}, {corners_output[1,1]:.1f})")
    print(f"    UR: ({corners_output[2,0]:.1f}, {corners_output[2,1]:.1f})")
    print(f"    LR: ({corners_output[3,0]:.1f}, {corners_output[3,1]:.1f})")
    
    # Create output with overlay
    output_img = warped_canvas.copy()
    overlay = output_img.copy()
    
    # Colors: LL=green, UL=blue, UR=yellow, LR=magenta
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    labels = ['LL', 'UL', 'UR', 'LR']
    
    # Draw polygon outline
    pts = corners_output.astype(np.int32)
    cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 255), thickness=3)
    
    # Draw corner markers
    marker_size = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i, (corner, color, label) in enumerate(zip(corners_output, colors, labels)):
        x, y = int(corner[0]), int(corner[1])
        
        # Crosshair
        cv2.line(overlay, (x - marker_size, y), (x + marker_size, y), color, 3)
        cv2.line(overlay, (x, y - marker_size), (x, y + marker_size), color, 3)
        
        # Circle
        cv2.circle(overlay, (x, y), marker_size + 5, color, 3)
        
        # Label with background
        text_size = cv2.getTextSize(label, font, 0.6, 2)[0]
        lx, ly = x + marker_size + 10, y - marker_size - 10
        cv2.rectangle(overlay, (lx-3, ly-text_size[1]-3), 
                     (lx+text_size[0]+3, ly+3), (0, 0, 0), -1)
        cv2.putText(overlay, label, (lx, ly), font, 0.6, color, 2)
    
    # Info
    cv2.putText(overlay, f"Rot:{rotation}°", (10, 30), font, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"Persp: TL{tl} TR{tr}", (10, 50), font, 0.4, (200, 200, 200), 1)
    cv2.putText(overlay, f"Persp: BL{bl} BR{br}", (10, 65), font, 0.4, (200, 200, 200), 1)
    
    # Blend overlay
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0, output_img)
    
    if output_path:
        cv2.imwrite(output_path, output_img)
        print(f"  Saved: {output_path}")
    
    return output_img, corners_output


def main():
    """Generate verification examples with real photos."""
    
    # Get real photos
    images_dir = Path(__file__).parent / "images"
    output_dir = Path(__file__).parent / "verification_with_real_photos"
    output_dir.mkdir(exist_ok=True)
    
    photo_files = sorted(images_dir.glob("*.jpg"))[:5]  # Use first 5 photos
    
    if not photo_files:
        print("No photos found in images/ folder!")
        return
    
    print("=" * 70)
    print("VERIFICATION WITH REAL PHOTOGRAPHS")
    print("=" * 70)
    
    # Test configurations
    rotations = [0, 15, -15]
    centers = [
        (PADDED_CANVAS_SIZE // 2, PADDED_CANVAS_SIZE // 2),
        (PADDED_CANVAS_SIZE // 2 + 50, PADDED_CANVAS_SIZE // 2 - 30),
    ]
    
    total = 0
    for photo_path in photo_files:
        print(f"\n{'='*70}")
        print(f"PHOTO: {photo_path.name}")
        print(f"{'='*70}")
        
        for rot in rotations:
            for cx, cy in centers:
                output_path = output_dir / f"{photo_path.stem}_rot{rot}_cx{cx}_cy{cy}.jpg"
                
                try:
                    img, corners = verify_with_real_photo(
                        photo_path=str(photo_path),
                        rotation=rot,
                        center_x=cx,
                        center_y=cy,
                        output_path=str(output_path)
                    )
                    total += 1
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"COMPLETE! Generated {total} verification images")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")
    
    print("\n📋 HOW TO VERIFY:")
    print("  1. Open images in the verification_with_real_photos folder")
    print("  2. Look at the overlaid corner markers (LL, UL, UR, LR)")
    print("  3. Check if markers align with the actual photo edges/corners")
    print("  4. If aligned: the polygon calculation is correct ✓")
    print("  5. If not aligned: there's still an issue with the formula")


if __name__ == "__main__":
    main()
