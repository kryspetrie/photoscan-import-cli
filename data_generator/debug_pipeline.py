#!/usr/bin/env python3
"""
Debug script to trace through the pipeline step by step.
Creates simple test images at each stage to see what's happening.
"""

import cv2
import numpy as np
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import CONFIG

# Configuration
CANVAS_SIZE = CONFIG['CANVAS_SIZE']           # 640
PADDING = CONFIG['CANVAS_PADDING']            # 300
PADDED_CANVAS_SIZE = CANVAS_SIZE + 2 * PADDING  # 1240
CROP_MARGIN = CONFIG['CROP_MARGIN']           # 60


def create_debug_photo(width, height):
    """Create a simple debug photo with numbered corners."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Colored corner squares
    s = min(80, width // 5, height // 5)
    
    # Draw colored rectangles in each corner
    # LL (0,0) - Green
    img[0:s, 0:s] = (0, 255, 0)
    cv2.putText(img, 'LL', (5, s//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # UL (0,W) - Blue  
    img[0:s, width-s:width] = (255, 0, 0)
    cv2.putText(img, 'UL', (width-s+5, s//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # UR (H,W) - Yellow
    img[height-s:height, width-s:width] = (0, 255, 255)
    cv2.putText(img, 'UR', (width-s+5, height-s//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # LR (H,0) - Magenta
    img[height-s:height, 0:s] = (255, 0, 255)
    cv2.putText(img, 'LR', (5, height-s//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Center dot
    cv2.circle(img, (width//2, height//2), 10, (0, 0, 0), -1)
    
    return img


def rotate_photo(photo, angle):
    """Rotate a photo."""
    h, w = photo.shape[:2]
    
    if abs(angle) < 1:
        return photo.copy(), (w, h), None
    
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
    """Composite photo onto canvas."""
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    
    top_left_x = int(cx - pw / 2)
    top_left_y = int(cy - ph / 2)
    
    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = pw, ph
    dst_x1, dst_y1 = top_left_x, top_left_y
    dst_x2, dst_y2 = top_left_x + pw, top_left_y + ph
    
    # Clip
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


def apply_global_perspective(canvas, canvas_w, canvas_h, crop_margin=60, seed=None):
    """Apply perspective warp to canvas."""
    if seed is not None:
        random.seed(seed)
    
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
    
    result = warped[crop_margin:crop_margin+CANVAS_SIZE, 
                   crop_margin:crop_margin+CANVAS_SIZE]
    
    return result, (tl, tr, bl, br), M


def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """Calculate corner positions in placement space."""
    if abs(rotation) < 1:
        return np.array([
            [center_x - width/2, center_y - height/2],
            [center_x + width/2, center_y - height/2],
            [center_x + width/2, center_y + height/2],
            [center_x - width/2, center_y + height/2]
        ], dtype=np.float32)
    
    photo_center = (width / 2, height / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    canvas_offset = (center_x - width / 2, center_y - height / 2)
    
    corners_photo = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_final = np.zeros_like(corners_photo)
    
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        corners_final[i, 0] = canvas_offset[0] + rotated[0]
        corners_final[i, 1] = canvas_offset[1] + rotated[1]
    
    return corners_final


def main():
    output_dir = Path(__file__).parent / "debug_pipeline"
    output_dir.mkdir(exist_ok=True)
    
    # Simple test case
    photo_w, photo_h = 300, 200
    rotation = 0
    center_x, center_y = 620, 620
    seed = 42
    
    print("=" * 70)
    print("PIPELINE DEBUG TRACE")
    print("=" * 70)
    print(f"\nInput: Photo {photo_w}x{photo_h}, Rot={rotation}°, Center=({center_x}, {center_y})")
    print(f"Canvas: {PADDED_CANVAS_SIZE}x{PADDED_CANVAS_SIZE} (padded)")
    print(f"Crop: {CROP_MARGIN}px margin")
    
    # Step 1: Create debug photo
    photo = create_debug_photo(photo_w, photo_h)
    cv2.imwrite(str(output_dir / "01_original.jpg"), photo)
    print(f"\nStep 1: Created photo {photo_w}x{photo_h}")
    print(f"  Corner colors: LL=GREEN, UL=BLUE, UR=YELLOW, LR=MAGENTA")
    
    # Step 2: Rotate
    rotated, (rot_w, rot_h), rot_M = rotate_photo(photo, rotation)
    cv2.imwrite(str(output_dir / "02_rotated.jpg"), rotated)
    print(f"\nStep 2: Rotated to {rot_w}x{rot_h}")
    
    # Step 3: Composite onto padded canvas
    canvas = np.zeros((PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE, 3), dtype=np.uint8)
    canvas[:] = (128, 128, 128)  # Gray background
    canvas, top_left = composite_photo_at_center(canvas, rotated, center_x, center_y)
    
    # Draw placement markers on canvas
    canvas_copy = canvas.copy()
    # Mark center
    cv2.circle(canvas_copy, (center_x, center_y), 20, (0, 255, 255), 3)
    cv2.putText(canvas_copy, f"CENTER({center_x},{center_y})", 
               (center_x + 30, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    # Mark where photo corners should be
    corners = get_rotated_polygon(photo_w, photo_h, center_x, center_y, rotation)
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        x, y = int(corners[i][0]), int(corners[i][1])
        cv2.circle(canvas_copy, (x, y), 15, (0, 0, 255), 2)
        cv2.putText(canvas_copy, name, (x + 20, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(output_dir / "03_canvas_with_markers.jpg"), canvas_copy)
    print(f"\nStep 3: Composited onto {PADDED_CANVAS_SIZE}x{PADDED_CANVAS_SIZE} canvas")
    print(f"  Photo top-left in canvas: {top_left}")
    print(f"  Expected corners in canvas space:")
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        print(f"    {name}: ({corners[i][0]:.1f}, {corners[i][1]:.1f})")
    
    # Check: Are the colored corners visible in the canvas?
    # Extract region around where we expect the photo
    tl_x, tl_y = top_left
    margin = 20
    region = canvas[max(0, tl_y-margin):min(PADDED_CANVAS_SIZE, tl_y+photo_h+margin),
                   max(0, tl_x-margin):min(PADDED_CANVAS_SIZE, tl_x+photo_w+margin)]
    cv2.imwrite(str(output_dir / "03b_canvas_region.jpg"), region)
    
    # Step 4: Apply perspective
    warped, persp_corners, persp_M = apply_global_perspective(
        canvas, PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE, 
        crop_margin=CROP_MARGIN, seed=seed
    )
    tl, tr, bl, br = persp_corners
    cv2.imwrite(str(output_dir / "04_warped.jpg"), warped)
    print(f"\nStep 4: Applied perspective warp")
    print(f"  TL offset: {tl}")
    print(f"  TR offset: {tr}")
    print(f"  BL offset: {bl}")
    print(f"  BR offset: {br}")
    print(f"  Output size: {warped.shape[1]}x{warped.shape[0]}")
    
    # Draw calculated corners on warped image
    warped_copy = warped.copy()
    
    # Transform placement corners through perspective
    corners_persp = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        pt = np.array([corners[i, 0], corners[i, 1], 1.0])
        result = persp_M @ pt
        corners_persp[i, 0] = result[0] / result[2]
        corners_persp[i, 1] = result[1] / result[2]
    
    # Crop (subtract margin)
    corners_cropped = corners_persp.copy()
    corners_cropped[:, 0] -= CROP_MARGIN
    corners_cropped[:, 1] -= CROP_MARGIN
    
    print(f"\n  Corners after perspective transform (before crop):")
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        print(f"    {name}: ({corners_persp[i, 0]:.1f}, {corners_persp[i, 1]:.1f})")
    
    print(f"\n  Corners after crop (subtract {CROP_MARGIN}):")
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        print(f"    {name}: ({corners_cropped[i, 0]:.1f}, {corners_cropped[i, 1]:.1f})")
    
    # Draw on warped image
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        x, y = int(corners_cropped[i, 0]), int(corners_cropped[i, 1])
        if 0 <= x < 640 and 0 <= y < 640:
            cv2.circle(warped_copy, (x, y), 15, colors[i], 3)
            cv2.putText(warped_copy, name, (x + 20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
    cv2.imwrite(str(output_dir / "05_warped_with_corners.jpg"), warped_copy)
    
    # Step 5: Find actual colored corners in warped image
    print(f"\nStep 5: Detecting actual colored corners in warped image...")
    
    # Detect green corners
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    
    # Green mask
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Blue mask
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Yellow mask
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Magenta mask
    lower_magenta = np.array([140, 100, 100])
    upper_magenta = np.array([170, 255, 255])
    mask_magenta = cv2.inRange(hsv, lower_magenta, upper_magenta)
    
    def get_centroid(mask, name):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        return None
    
    detected = {
        'LL': get_centroid(mask_green, 'LL'),
        'UL': get_centroid(mask_blue, 'UL'),
        'UR': get_centroid(mask_yellow, 'UR'),
        'LR': get_centroid(mask_magenta, 'LR'),
    }
    
    # Save masks
    cv2.imwrite(str(output_dir / "06_mask_green.jpg"), mask_green)
    cv2.imwrite(str(output_dir / "06_mask_blue.jpg"), mask_blue)
    cv2.imwrite(str(output_dir / "06_mask_yellow.jpg"), mask_yellow)
    cv2.imwrite(str(output_dir / "06_mask_magenta.jpg"), mask_magenta)
    
    print(f"\n  Detected corners (CV):")
    for name, pos in detected.items():
        if pos:
            print(f"    {name}: {pos}")
        else:
            print(f"    {name}: NOT DETECTED")
    
    print(f"\n  Calculated corners (formula):")
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        print(f"    {name}: ({corners_cropped[i, 0]:.1f}, {corners_cropped[i, 1]:.1f})")
    
    # Final overlay
    final = warped.copy()
    for i, (name, color) in enumerate(zip(['LL', 'UL', 'UR', 'LR'], colors)):
        # Calculated - white cross
        x, y = int(corners_cropped[i, 0]), int(corners_cropped[i, 1])
        cv2.line(final, (x-15, y), (x+15, y), (255, 255, 255), 3)
        cv2.line(final, (x, y-15), (x, y+15), (255, 255, 255), 3)
        
        # Detected - colored circle
        if detected[name]:
            cv2.circle(final, detected[name], 20, color, 3)
            cv2.putText(final, f"D:{name}", 
                       (detected[name][0] + 25, detected[name][1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Label calculated
        cv2.putText(final, f"C:{name}", (x + 25, y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(str(output_dir / "07_final_comparison.jpg"), final)
    
    print(f"\n" + "=" * 70)
    print("Saved debug images to:", output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
