#!/usr/bin/env python3
"""
Generate example images with corner overlays for visual verification.

This script creates synthetic test photos with distinct colored corner markers,
applies the same transformations as the data generator, and overlays the
calculated polygon corners so you can verify accuracy.

Run this script to see if the overlaid corner points align with the actual
colored markers visible in the output images.
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path


# =============================================================================
# CONFIGURATION - Match generate_dataset.py v34
# =============================================================================

CONFIG = {
    'CANVAS_SIZE': 640,
    'CANVAS_PADDING': 300,  # 300px on each side (total: 1240px)
    'CROP_MARGIN': 60,
    'ROTATION_RANGE': 30,
    'PERSPECTIVE_STRENGTH_MIN': 0.05,
    'PERSPECTIVE_STRENGTH_MAX': 0.20,
}

CANVAS_SIZE = CONFIG['CANVAS_SIZE']
PADDED_CANVAS_SIZE = CONFIG['CANVAS_SIZE'] + 2 * CONFIG['CANVAS_PADDING']  # 1240
CROP_MARGIN = CONFIG['CROP_MARGIN']
OUTPUT_SIZE = CONFIG['CANVAS_SIZE']  # 640


# =============================================================================
# PIPELINE FUNCTIONS (from generate_dataset.py)
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
    
    # Clip to canvas bounds
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
    """
    Apply perspective warp to canvas and crop to output size.
    """
    # Random perspective displacement (matching generator logic)
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
    result = warped[crop_margin:crop_margin+CONFIG['CANVAS_SIZE'], 
                   crop_margin:crop_margin+CONFIG['CANVAS_SIZE']]
    
    return result, (tl, tr, bl, br)


def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """
    Calculate the 4 corner coordinates of a rotated rectangle.
    
    CORRECTED: Uses raw rotation matrix (M_raw) without canvas expansion offset.
    The canvas expansion offset is already included in how the photo is placed
    via composite_photo_at_center().
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
        # Apply canvas offset (NOT M with canvas expansion baked in)
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
# CREATE TEST PHOTOS
# =============================================================================

def create_test_photos():
    """Create synthetic test photos with distinct corner markers."""
    photos = []
    photo_size = 400  # Use fixed size for cleaner testing
    
    # Photo 1: Gradient with colored corner markers
    photo1 = np.zeros((photo_size, photo_size, 3), dtype=np.uint8)
    for i in range(photo_size):
        photo1[i, :] = [i * 255 // photo_size, 128, 255 - i * 255 // photo_size]
    
    # Corner markers
    m = 25  # margin
    s = 50  # size
    # Bottom-Left (LL) - Green
    cv2.rectangle(photo1, (m, photo_size-m-s), (m+s, photo_size-m), (0, 255, 0), -1)
    # Top-Left (UL) - Blue
    cv2.rectangle(photo1, (m, m), (m+s, m+s), (255, 0, 0), -1)
    # Top-Right (UR) - Yellow
    cv2.rectangle(photo1, (photo_size-m-s, m), (photo_size-m, m+s), (0, 255, 255), -1)
    # Bottom-Right (LR) - Magenta
    cv2.rectangle(photo1, (photo_size-m-s, photo_size-m-s), (photo_size-m, photo_size-m), (255, 0, 255), -1)
    
    # Center crosshair
    c = photo_size // 2
    cv2.line(photo1, (c-60, c), (c+60, c), (255, 255, 255), 2)
    cv2.line(photo1, (c, c-60), (c, c+60), (255, 255, 255), 2)
    
    photos.append(('Gradient with corners', photo1))
    
    # Photo 2: Grid pattern with labeled markers
    photo2 = np.ones((photo_size, photo_size, 3), dtype=np.uint8) * 200
    for i in range(0, photo_size, 50):
        cv2.line(photo2, (i, 0), (i, photo_size), (150, 150, 150), 1)
        cv2.line(photo2, (0, i), (photo_size, i), (150, 150, 150), 1)
    
    r = 35
    # Top-Left (UL) - Blue
    cv2.circle(photo2, (r, r), r, (255, 0, 0), -1)
    cv2.putText(photo2, 'UL', (r-18, r+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    # Top-Right (UR) - Yellow
    cv2.circle(photo2, (photo_size-r, r), r, (0, 255, 255), -1)
    cv2.putText(photo2, 'UR', (photo_size-r-18, r+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    # Bottom-Left (LL) - Green
    cv2.circle(photo2, (r, photo_size-r), r, (0, 255, 0), -1)
    cv2.putText(photo2, 'LL', (r-18, photo_size-r+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    # Bottom-Right (LR) - Magenta
    cv2.circle(photo2, (photo_size-r, photo_size-r), r, (255, 0, 255), -1)
    cv2.putText(photo2, 'LR', (photo_size-r-18, photo_size-r+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    photos.append(('Grid with labeled corners', photo2))
    
    # Photo 3: Distinct quadrants
    photo3 = np.zeros((photo_size, photo_size, 3), dtype=np.uint8)
    q = photo_size // 2
    photo3[:q, :q] = [200, 50, 50]    # Top-left: Reddish
    photo3[:q, q:] = [50, 200, 50]     # Top-right: Greenish
    photo3[q:, :q] = [50, 50, 200]     # Bottom-left: Bluish
    photo3[q:, q:] = [200, 200, 50]   # Bottom-right: Yellowish
    
    # White border
    cv2.rectangle(photo3, (0, 0), (photo_size-1, photo_size-1), (255, 255, 255), 4)
    
    photos.append(('Distinct quadrants', photo3))
    
    return photos


# =============================================================================
# GENERATE EXAMPLE WITH OVERLAY
# =============================================================================

def generate_example_with_overlay(
    photo,
    rotation,
    center_x,
    center_y,
    output_path=None
):
    """Generate an example with corner overlays for verification."""
    h, w = photo.shape[:2]
    
    # Step 1: Rotate the photo
    rotated_photo, (rot_w, rot_h), M = rotate_photo(photo, rotation)
    
    # Create padded canvas
    padded_canvas = np.zeros((PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE, 3), dtype=np.uint8)
    padded_canvas[:] = (128, 128, 128)
    
    # Step 2: Composite rotated photo onto canvas
    padded_canvas, _ = composite_photo_at_center(padded_canvas, rotated_photo, center_x, center_y)
    
    # Get corners in placement space (before perspective)
    corners_placement = get_rotated_polygon(w, h, center_x, center_y, rotation)
    
    # Step 3: Apply perspective warp
    warped_canvas, persp_corners = apply_global_perspective(
        padded_canvas, PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE, crop_margin=CROP_MARGIN
    )
    
    tl, tr, bl, br = persp_corners
    
    # Transform corners through perspective
    corners_warped = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        corners_warped[i] = apply_perspective_to_point(
            corners_placement[i], tl, tr, bl, br, PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE
        )
    
    # Crop and transform to output space
    crop_x = CROP_MARGIN
    crop_y = CROP_MARGIN
    
    corners_cropped = corners_warped.copy()
    corners_cropped[:, 0] -= crop_x
    corners_cropped[:, 1] -= crop_y
    
    # Scale to output size
    scale = OUTPUT_SIZE / (PADDED_CANVAS_SIZE - 2 * CROP_MARGIN)
    corners_output = corners_cropped * scale
    
    # Final output image (warped_canvas is already 640x640)
    output_img = warped_canvas.copy()
    
    # Create overlay
    overlay = output_img.copy()
    
    # Colors for corners (LL, UL, UR, LR order)
    colors = [
        (0, 255, 0),      # LL - Green
        (255, 0, 0),      # UL - Blue
        (0, 255, 255),    # UR - Yellow
        (255, 0, 255),    # LR - Magenta
    ]
    labels = ['LL', 'UL', 'UR', 'LR']
    
    # Draw polygon
    pts = corners_output.astype(np.int32)
    cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
    
    # Draw corner markers
    marker_size = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i, (corner, color, label) in enumerate(zip(corners_output, colors, labels)):
        x, y = int(corner[0]), int(corner[1])
        
        # Crosshair
        cv2.line(overlay, (x - marker_size, y), (x + marker_size, y), color, 2)
        cv2.line(overlay, (x, y - marker_size), (x, y + marker_size), color, 2)
        
        # Circle
        cv2.circle(overlay, (x, y), marker_size + 4, color, 2)
        
        # Label with background
        font_scale = 0.55
        thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        label_x = x + marker_size + 8
        label_y = y - marker_size - 8
        
        # Background
        cv2.rectangle(overlay,
                     (label_x - 2, label_y - text_size[1] - 2),
                     (label_x + text_size[0] + 2, label_y + 2),
                     (0, 0, 0), -1)
        cv2.putText(overlay, label, (label_x, label_y), font, font_scale, color, thickness)
    
    # Info text
    info = f"Rot:{rotation:.0f}°"
    cv2.putText(overlay, info, (10, 25), font, 0.5, (255, 255, 255), 1)
    
    # Blend
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0, output_img)
    
    # Save
    if output_path:
        cv2.imwrite(output_path, output_img)
    
    return output_img, corners_output


# =============================================================================
# MAIN
# =============================================================================

def main():
    import random
    
    output_dir = Path(__file__).parent / "verification_examples"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("GENERATING VERIFICATION EXAMPLES WITH CORNER OVERLAYS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Canvas size: {CANVAS_SIZE}x{CANVAS_SIZE}")
    print(f"  Padded canvas: {PADDED_CANVAS_SIZE}x{PADDED_CANVAS_SIZE}")
    print(f"  Crop margin: {CROP_MARGIN}px")
    print(f"  Output size: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    print()
    
    photos = create_test_photos()
    
    # Test configurations
    rotations = [0, 10, 15, -15, 20, -20, 25]
    positions = [
        (PADDED_CANVAS_SIZE // 2, PADDED_CANVAS_SIZE // 2),
        (PADDED_CANVAS_SIZE // 2 + 100, PADDED_CANVAS_SIZE // 2 - 50),
        (PADDED_CANVAS_SIZE // 2 - 80, PADDED_CANVAS_SIZE // 2 + 60),
    ]
    
    total = 0
    
    for photo_idx, (photo_name, photo) in enumerate(photos):
        print(f"\n{'='*70}")
        print(f"PHOTO {photo_idx + 1}: {photo_name}")
        print(f"{'='*70}")
        
        for pos_idx, (cx, cy) in enumerate(positions):
            for rot_idx, rotation in enumerate(rotations):
                # Limit total examples
                if total >= 15:
                    break
                
                # Skip some combinations
                if pos_idx > 0 and rot_idx > 3:
                    continue
                
                output_path = output_dir / f"verify_p{photo_idx+1}_pos{pos_idx}_rot{rotation:.0f}.png"
                
                print(f"\n  Position {pos_idx}: center=({cx}, {cy}), rotation={rotation:.0f}°")
                
                img, corners = generate_example_with_overlay(
                    photo=photo,
                    rotation=rotation,
                    center_x=cx,
                    center_y=cy,
                    output_path=str(output_path)
                )
                
                print(f"    LL: ({corners[0,0]:.1f}, {corners[0,1]:.1f})")
                print(f"    UL: ({corners[1,0]:.1f}, {corners[1,1]:.1f})")
                print(f"    UR: ({corners[2,0]:.1f}, {corners[2,1]:.1f})")
                print(f"    LR: ({corners[3,0]:.1f}, {corners[3,1]:.1f})")
                print(f"    Saved: {output_path.name}")
                
                total += 1
            
            if total >= 15:
                break
        
        if total >= 15:
            break
    
    print(f"\n{'='*70}")
    print(f"COMPLETE! Generated {total} verification examples")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    print("\n📋 LEGEND:")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  LL (Green)   → Lower-Left corner of the photo          │")
    print("  │  UL (Blue)    → Upper-Left corner of the photo         │")
    print("  │  UR (Yellow)  → Upper-Right corner of the photo        │")
    print("  │  LR (Magenta) → Lower-Right corner of the photo        │")
    print("  │  White outline → Calculated bounding quadrilateral    │")
    print("  └─────────────────────────────────────────────────────────┘")
    
    print("\n✅ VERIFICATION STEPS:")
    print("  1. Open the generated images in the verification_examples folder")
    print("  2. Visually compare the colored corner markers (in the photo)")
    print("     with the overlaid corner points (LL, UL, UR, LR)")
    print("  3. The overlaid points should align precisely with the")
    print("     colored corners visible in the output image")
    print("  4. If aligned: <5px accuracy achieved ✓")
    print("  5. If misaligned: There's still an issue with the formula")


if __name__ == "__main__":
    main()
