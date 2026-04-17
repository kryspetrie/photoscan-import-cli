#!/usr/bin/env python3
"""
Proper verification using real photos - scaled to match generator.
"""

import cv2
import numpy as np
from pathlib import Path
import random

import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import CONFIG

# Configuration (must match generate_dataset.py)
CANVAS_SIZE = CONFIG['CANVAS_SIZE']           # 640
PADDING = CONFIG['CANVAS_PADDING']            # 300
PADDED_CANVAS_SIZE = CANVAS_SIZE + 2 * PADDING  # 1240
CROP_MARGIN = CONFIG['CROP_MARGIN']           # 60
OUTPUT_SIZE = CONFIG['CANVAS_SIZE']           # 640


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


def scale_photo_to_generator_size(photo, rnd=None):
    """Scale photo to match generator's photo size range (180-480px)."""
    if rnd is None:
        rnd = random.Random()
    
    h, w = photo.shape[:2]
    
    # Match generator logic
    size = rnd.randint(CONFIG['PHOTO_SIZE_MIN'], int(CONFIG['PHOTO_SIZE_MAX'] * 0.8))
    height = int(size * rnd.uniform(0.75, 0.85))
    width = int(height * w / h)
    
    scaled = cv2.resize(photo, (width, height), interpolation=cv2.INTER_AREA)
    
    return scaled, (width, height)


def verify_pipeline(photo_path, rotation, center_x, center_y, seed, output_dir):
    """Run the full pipeline with proper scaling."""
    
    # Load and scale photo (like the generator does)
    photo = cv2.imread(photo_path)
    if photo is None:
        raise FileNotFoundError(f"Could not load: {photo_path}")
    
    rnd = random.Random(seed)
    photo, (orig_w, orig_h) = scale_photo_to_generator_size(photo, rnd)
    h, w = photo.shape[:2]
    
    print(f"\n  Photo: {Path(photo_path).name}")
    print(f"  Original size: {orig_w}x{orig_h}")
    print(f"  Scaled to: {w}x{h}")
    print(f"  Rotation: {rotation}°, Center: ({center_x}, {center_y}), Seed: {seed}")
    
    # Step 1: Rotate
    rotated, (rot_w, rot_h), rot_M = rotate_photo(photo, rotation)
    
    # Step 2: Composite onto canvas
    padded_canvas = np.zeros((PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE, 3), dtype=np.uint8)
    padded_canvas[:] = (128, 128, 128)
    canvas, tl = composite_photo_at_center(padded_canvas, rotated, center_x, center_y)
    
    # Step 3: Apply perspective warp
    warped, persp_corners, persp_M = apply_global_perspective(
        canvas, PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE, 
        crop_margin=CROP_MARGIN, seed=seed
    )
    tl_off, tr_off, bl_off, br_off = persp_corners
    
    # Step 4: Calculate corners in placement space
    corners_placement = get_rotated_polygon(w, h, center_x, center_y, rotation)
    
    # Step 5: Transform through perspective
    corners_persp = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        pt = np.array([corners_placement[i, 0], corners_placement[i, 1], 1.0])
        result = persp_M @ pt
        corners_persp[i, 0] = result[0] / result[2]
        corners_persp[i, 1] = result[1] / result[2]
    
    # Step 6: Crop
    corners_cropped = corners_persp.copy()
    corners_cropped[:, 0] -= CROP_MARGIN
    corners_cropped[:, 1] -= CROP_MARGIN
    
    # Step 7: Scale to output
    scale = OUTPUT_SIZE / (PADDED_CANVAS_SIZE - 2 * CROP_MARGIN)
    corners_output = corners_cropped * scale
    
    print(f"  Perspective offsets: TL={tl_off}, TR={tr_off}, BL={bl_off}, BR={br_off}")
    print(f"  Corners in output space:")
    for name, c in zip(['LL', 'UL', 'UR', 'LR'], corners_output):
        x, y = c
        status = "✓" if 0 <= x <= OUTPUT_SIZE and 0 <= y <= OUTPUT_SIZE else "✗"
        print(f"    {name}: ({x:.1f}, {y:.1f}) {status}")
    
    # Create overlay
    output_img = warped.copy()
    overlay = output_img.copy()
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]  # LL, UL, UR, LR
    labels = ['LL', 'UL', 'UR', 'LR']
    
    # Draw polygon (convert LL,UL,UR,LR to TL,TR,BR,BL order for cv2)
    # LL=BL(2), UL=TL(1), UR=TR(0), LR=BR(3)
    poly_order = [3, 1, 0, 2]  # UL, LL, LR, UR -> TL, TR, BR, BL
    pts = corners_output[poly_order].astype(np.int32)
    cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 255), thickness=3)
    
    # Draw corners
    for i, (corner, color, label) in enumerate(zip(corners_output, colors, labels)):
        x, y = int(np.clip(corner[0], 0, OUTPUT_SIZE-1)), int(np.clip(corner[1], 0, OUTPUT_SIZE-1))
        
        cv2.circle(overlay, (x, y), 15, color, 3)
        cv2.line(overlay, (x-12, y), (x+12, y), color, 3)
        cv2.line(overlay, (x, y-12), (x, y+12), color, 3)
        
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        lx, ly = x + 20, y - 20
        cv2.rectangle(overlay, (lx-3, ly-text_size[1]-3), (lx+text_size[0]+3, ly+3), (0, 0, 0), -1)
        cv2.putText(overlay, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Info
    cv2.putText(overlay, f"Rot:{rotation}° Size:{w}x{h}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Blend
    cv2.addWeighted(overlay, 0.85, output_img, 0.15, 0, output_img)
    
    # Save
    stem = Path(photo_path).stem
    output_path = output_dir / f"{stem}_rot{rotation}_seed{seed}.jpg"
    cv2.imwrite(str(output_path), output_img)
    print(f"  Saved: {output_path.name}")
    
    return corners_output


def main():
    output_dir = Path(__file__).parent / "verification_scaled"
    output_dir.mkdir(exist_ok=True)
    
    images_dir = Path(__file__).parent / "images"
    photo_files = sorted(images_dir.glob("*.jpg"))[:5]
    
    print("=" * 70)
    print("VERIFICATION WITH PROPER SCALING")
    print("=" * 70)
    print(f"Configuration (matching generator):")
    print(f"  Photo size: {CONFIG['PHOTO_SIZE_MIN']}-{CONFIG['PHOTO_SIZE_MAX']}px (scaled)")
    print(f"  Canvas: {CANVAS_SIZE}x{CANVAS_SIZE} + {PADDING}px padding = {PADDED_CANVAS_SIZE}x{PADDED_CANVAS_SIZE}")
    print(f"  Crop: {CROP_MARGIN}px margin")
    print(f"  Output: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    print()
    
    test_cases = [
        (0, 620, 620, 42),
        (15, 620, 620, 42),
        (-15, 620, 620, 42),
        (0, 670, 590, 123),
        (15, 670, 590, 123),
    ]
    
    for photo_path in photo_files:
        print(f"\n{'='*70}")
        print(f"PHOTO: {photo_path.name}")
        print(f"{'='*70}")
        
        for rot, cx, cy, seed in test_cases:
            try:
                verify_pipeline(str(photo_path), rot, cx, cy, seed, output_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"COMPLETE! Check {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
