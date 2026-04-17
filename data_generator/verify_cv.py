#!/usr/bin/env python3
"""
Comprehensive verification using CV corner detection on synthetic images.

This script:
1. Creates synthetic photos with DISTINCT COLORED CORNERS (Green, Blue, Yellow, Magenta)
2. Runs them through the exact pipeline
3. Uses computer vision to DETECT the actual corner positions in the output
4. Compares detected positions to calculated positions
5. Reports pixel error

This verifies BOTH:
- The transformation pipeline is correct
- The corner overlay/visualization is correct
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
OUTPUT_SIZE = CONFIG['CANVAS_SIZE']           # 640


# =============================================================================
# SYNTHETIC PHOTO CREATION WITH COLORED CORNERS
# =============================================================================

def create_synthetic_photo_with_corners(width, height):
    """
    Create a synthetic photo with distinct colored corner markers.
    
    Colors:
    - LL (Lower-Left): Green (0, 255, 0)
    - UL (Upper-Left): Blue (255, 0, 0)
    - UR (Upper-Right): Yellow (0, 255, 255)
    - LR (Lower-Right): Magenta (255, 0, 255)
    """
    # Create base image with gradient (helps see rotation)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        img[y, :] = [int(y * 200 / height) + 30, 100, int((width - y) * 200 / width) + 30]
    
    # Add grid pattern for visual reference
    grid_size = 50
    for i in range(0, width, grid_size):
        cv2.line(img, (i, 0), (i, height), (60, 60, 60), 1)
    for i in range(0, height, grid_size):
        cv2.line(img, (0, i), (width, i), (60, 60, 60), 1)
    
    # Corner marker size
    marker_size = min(60, width // 8, height // 8)
    margin = 10
    
    # Lower-Left (LL) - Green
    cv2.rectangle(img, 
                  (margin, height - margin - marker_size),
                  (margin + marker_size, height - margin),
                  (0, 255, 0), -1)
    
    # Upper-Left (UL) - Blue
    cv2.rectangle(img,
                  (margin, margin),
                  (margin + marker_size, margin + marker_size),
                  (255, 0, 0), -1)
    
    # Upper-Right (UR) - Yellow
    cv2.rectangle(img,
                  (width - margin - marker_size, margin),
                  (width - margin, margin + marker_size),
                  (0, 255, 255), -1)
    
    # Lower-Right (LR) - Magenta
    cv2.rectangle(img,
                  (width - margin - marker_size, height - margin - marker_size),
                  (width - margin, height - margin),
                  (255, 0, 255), -1)
    
    # Center crosshair
    cx, cy = width // 2, height // 2
    cv2.line(img, (cx - 30, cy), (cx + 30, cy), (255, 255, 255), 2)
    cv2.line(img, (cx, cy - 30), (cx, cy + 30), (255, 255, 255), 2)
    
    return img


# =============================================================================
# COMPUTER VISION CORNER DETECTION
# =============================================================================

def detect_colored_corner(img, target_color, tolerance=50):
    """
    Detect the center of a colored region in the image.
    Uses color thresholding + contour centroid detection.
    """
    # Convert to HSV for better color matching
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color range
    if target_color == 'green':  # (0, 255, 0)
        lower = np.array([35, 100, 100])
        upper = np.array([85, 255, 255])
    elif target_color == 'blue':  # (255, 0, 0)
        lower = np.array([100, 100, 100])
        upper = np.array([130, 255, 255])
    elif target_color == 'yellow':  # (0, 255, 255)
        lower = np.array([15, 100, 100])
        upper = np.array([35, 255, 255])
    elif target_color == 'magenta':  # (255, 0, 255)
        lower = np.array([140, 100, 100])
        upper = np.array([170, 255, 255])
    else:
        raise ValueError(f"Unknown color: {target_color}")
    
    # Create mask
    mask = cv2.inRange(hsv, lower, upper)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, mask
    
    # Get largest contour (should be our corner marker)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    if area < 100:  # Too small
        return None, None, mask
    
    # Get centroid
    M = cv2.moments(largest)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy), area, mask
    
    return None, area, mask


def detect_all_corners(img):
    """
    Detect all 4 corners in the output image.
    Returns dict with corner name -> (x, y) position
    """
    corners = {}
    
    colors = ['green', 'blue', 'yellow', 'magenta']
    names = ['LL', 'UL', 'UR', 'LR']
    
    for color, name in zip(colors, names):
        pos, area, mask = detect_colored_corner(img, color)
        corners[name] = pos
        corners[f'{name}_area'] = area
        corners[f'{name}_mask'] = mask
    
    return corners


# =============================================================================
# PIPELINE FUNCTIONS (same as generate_dataset.py)
# =============================================================================

def rotate_photo(photo, angle):
    """Rotate a photo by specified degrees."""
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
    """Calculate the 4 corner coordinates of a rotated rectangle."""
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
    
    corners_photo = np.array([
        [0, 0], [width, 0], [width, height], [0, height]
    ], dtype=np.float32)
    
    corners_final = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        corners_final[i, 0] = canvas_offset[0] + rotated[0]
        corners_final[i, 1] = canvas_offset[1] + rotated[1]
    
    return corners_final


# =============================================================================
# MAIN VERIFICATION
# =============================================================================

def run_verification(photo_w, photo_h, rotation, center_x, center_y, seed, output_dir):
    """Run full verification with CV corner detection."""
    
    print(f"\n{'='*70}")
    print(f"PHOTO: {photo_w}x{photo_h}, Rot: {rotation}°, Center: ({center_x}, {center_y}), Seed: {seed}")
    print(f"{'='*70}")
    
    # Step 1: Create synthetic photo with colored corners
    photo = create_synthetic_photo_with_corners(photo_w, photo_h)
    
    print(f"\nStep 1: Created synthetic photo with colored corners")
    
    # Step 2: Rotate
    rotated, (rot_w, rot_h), rot_M = rotate_photo(photo, rotation)
    
    # Step 3: Composite onto canvas
    padded_canvas = np.zeros((PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE, 3), dtype=np.uint8)
    padded_canvas[:] = (128, 128, 128)
    canvas, _ = composite_photo_at_center(padded_canvas, rotated, center_x, center_y)
    
    # Step 4: Apply perspective warp
    warped, persp_corners, persp_M = apply_global_perspective(
        canvas, PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE,
        crop_margin=CROP_MARGIN, seed=seed
    )
    
    print(f"Step 4: Applied perspective warp, TL={persp_corners[0]}, TR={persp_corners[1]}")
    
    # Step 5: Use CV to detect actual corner positions in warped output
    detected = detect_all_corners(warped)
    
    print(f"\nStep 5: CV-detected corners in output image:")
    for name in ['LL', 'UL', 'UR', 'LR']:
        pos = detected[name]
        area = detected.get(f'{name}_area')
        if pos:
            area_str = f"{area:.0f}" if area is not None else "N/A"
            print(f"  {name}: ({pos[0]:.1f}, {pos[1]:.1f}), area={area_str}")
        else:
            area_str = f"{area:.0f}" if area is not None else "N/A"
            print(f"  {name}: NOT DETECTED, area={area_str}")
    
    # Step 6: Calculate expected corner positions
    corners_placement = get_rotated_polygon(photo_w, photo_h, center_x, center_y, rotation)
    
    # Transform through perspective
    corners_persp = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        pt = np.array([corners_placement[i, 0], corners_placement[i, 1], 1.0])
        result = persp_M @ pt
        corners_persp[i, 0] = result[0] / result[2]
        corners_persp[i, 1] = result[1] / result[2]
    
    # Crop
    corners_cropped = corners_persp.copy()
    corners_cropped[:, 0] -= CROP_MARGIN
    corners_cropped[:, 1] -= CROP_MARGIN
    
    # Scale (but output is already CANVAS_SIZE x CANVAS_SIZE, not scaled)
    # The warped result is already 640x640 after crop
    corners_output = corners_cropped  # Already in correct space
    
    print(f"\nStep 6: Calculated corners (in output space):")
    for name, c in zip(['LL', 'UL', 'UR', 'LR'], corners_output):
        print(f"  {name}: ({c[0]:.1f}, {c[1]:.1f})")
    
    # Step 7: Compare detected vs calculated
    print(f"\nStep 7: Error analysis:")
    errors = {}
    all_valid = True
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        detected_pos = detected[name]
        calculated = corners_output[i]
        
        if detected_pos is None:
            print(f"  {name}: Could not detect - cannot compute error")
            all_valid = False
            continue
        
        error_x = detected_pos[0] - calculated[0]
        error_y = detected_pos[1] - calculated[1]
        error_dist = np.sqrt(error_x**2 + error_y**2)
        
        errors[name] = {
            'detected': detected_pos,
            'calculated': (calculated[0], calculated[1]),
            'error_x': error_x,
            'error_y': error_y,
            'error_dist': error_dist
        }
        
        status = "✓" if error_dist < 5 else "✗"
        print(f"  {name}: Detected=({detected_pos[0]:.1f},{detected_pos[1]:.1f}) "
              f"Calc=({calculated[0]:.1f},{calculated[1]:.1f}) "
              f"Error=({error_x:.1f},{error_y:.1f}) Distance={error_dist:.2f}px {status}")
    
    # Step 8: Create visualization
    output_img = warped.copy()
    overlay = output_img.copy()
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]  # LL, UL, UR, LR
    
    # Draw detected corners (solid circles)
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        pos = detected[name]
        if pos:
            cv2.circle(overlay, pos, 20, colors[i], 3)
            cv2.putText(overlay, f"D:{name}", (pos[0]+25, pos[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    
    # Draw calculated corners (crosses)
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        c = corners_output[i]
        x, y = int(c[0]), int(c[1])
        if 0 <= x < OUTPUT_SIZE and 0 <= y < OUTPUT_SIZE:
            cv2.line(overlay, (x-10, y), (x+10, y), (255, 255, 255), 2)
            cv2.line(overlay, (x, y-10), (x, y+10), (255, 255, 255), 2)
            cv2.putText(overlay, f"C:{name}", (x+15, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw error lines between detected and calculated
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        if detected[name] and errors.get(name):
            d = detected[name]
            c = errors[name]['calculated']
            cv2.line(overlay, d, (int(c[0]), int(c[1])), (0, 255, 0), 1)
    
    # Info
    cv2.putText(overlay, f"Rot:{rotation}° Size:{photo_w}x{photo_h}", 
               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Blend
    cv2.addWeighted(overlay, 0.7, output_img, 0.3, 0, output_img)
    
    # Save
    output_path = output_dir / f"verify_cv_w{photo_w}_h{photo_h}_rot{rotation}_s{seed}.jpg"
    cv2.imwrite(str(output_path), output_img)
    print(f"\nSaved visualization: {output_path.name}")
    
    # Also save masks for debugging
    for name in ['LL', 'UL', 'UR', 'LR']:
        mask = detected.get(f'{name}_mask')
        if mask is not None and detected.get(name) is not None:
            mask_path = output_dir / f"mask_{name}_{output_path.stem}.png"
            cv2.imwrite(str(mask_path), mask)
    
    return errors, all_valid


def main():
    output_dir = Path(__file__).parent / "verification_cv"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("COMPUTER VISION CORNER DETECTION VERIFICATION")
    print("=" * 70)
    print("\nThis verification:")
    print("1. Creates synthetic photos with colored corner markers")
    print("2. Runs them through the transformation pipeline")
    print("3. Uses CV (color detection) to find ACTUAL corner positions")
    print("4. Compares detected vs calculated positions")
    print("5. Reports pixel error")
    print("\n" + "=" * 70)
    
    # Test configurations
    test_cases = [
        # (width, height, rotation, center_x, center_y, seed)
        (400, 300, 0, 620, 620, 42),      # No rotation
        (400, 300, 15, 620, 620, 42),     # 15 degree
        (400, 300, -15, 620, 620, 42),    # -15 degree
        (400, 300, 20, 620, 620, 42),     # 20 degree
        (300, 400, 0, 620, 620, 42),      # Portrait orientation
        (300, 400, 15, 620, 620, 42),      # Portrait with rotation
        (500, 350, 10, 650, 600, 123),     # Large photo
        (350, 500, -10, 600, 650, 123),    # Tall photo
    ]
    
    all_errors = []
    valid_count = 0
    
    for w, h, rot, cx, cy, seed in test_cases:
        errors, valid = run_verification(w, h, rot, cx, cy, seed, output_dir)
        
        if valid:
            valid_count += 1
            for name, err in errors.items():
                all_errors.append(err['error_dist'])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_errors:
        avg_error = np.mean(all_errors)
        max_error = np.max(all_errors)
        min_error = np.min(all_errors)
        within_5px = sum(1 for e in all_errors if e < 5)
        
        print(f"\nTotal error measurements: {len(all_errors)}")
        print(f"Average error: {avg_error:.2f}px")
        print(f"Maximum error: {max_error:.2f}px")
        print(f"Minimum error: {min_error:.2f}px")
        print(f"Within 5px: {within_5px}/{len(all_errors)} ({100*within_5px/len(all_errors):.1f}%)")
        
        if max_error < 5:
            print("\n✅ SUCCESS! All corners within 5px accuracy")
        else:
            print("\n❌ FAILURE! Some corners exceed 5px accuracy")
    else:
        print("\n❌ No valid measurements (corners not detected)")
    
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
