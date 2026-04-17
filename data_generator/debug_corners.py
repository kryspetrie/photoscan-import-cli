#!/usr/bin/env python3
"""
Debug: Compare calculated vs actual corner positions
=====================================================

This script traces exactly where the discrepancy comes from.
"""

import cv2
import numpy as np
import math
from pathlib import Path


def get_rotated_polygon_current(width, height, center_x, center_y, rotation):
    """
    Current implementation - uses OpenCV rotation matrix.
    This is what generate_dataset.py uses.
    """
    if abs(rotation) < 1:
        hw, hh = width / 2, height / 2
        return np.array([
            [center_x - hw, center_y - hh],
            [center_x + hw, center_y - hh],
            [center_x + hw, center_y + hh],
            [center_x - hw, center_y + hh]
        ], dtype=np.float32)
    
    M = cv2.getRotationMatrix2D((center_x, center_y), rotation, 1.0)
    
    hw, hh = width / 2, height / 2
    corners = np.array([
        [center_x - hw, center_y - hh],
        [center_x + hw, center_y - hh],
        [center_x + hw, center_y + hh],
        [center_x - hw, center_y + hh]
    ], dtype=np.float32)
    
    rotated = np.zeros_like(corners)
    for i in range(4):
        pt = np.array([corners[i, 0], corners[i, 1], 1], dtype=np.float32)
        result = M @ pt
        rotated[i] = [result[0], result[1]]
    
    return rotated


def rotate_photo_with_details(photo, angle):
    """
    Rotate photo and return transformation details.
    Returns: rotated photo, old_size, new_size, center, M, M_adj
    """
    h, w = photo.shape[:2]
    center = (w / 2, h / 2)
    
    if abs(angle) < 1:
        return photo, (w, h), (w, h), center, np.eye(3), np.eye(3)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    
    M_adj = M.copy()
    M_adj[0, 2] += (new_w - w) / 2
    M_adj[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(photo, M_adj, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(128, 128, 128, 255))
    
    return rotated, (w, h), (new_w, new_h), center, M, M_adj


def composite_with_details(canvas, photo, cx, cy):
    """
    Composite and return top-left position.
    Returns: composited canvas, top_left_x, top_left_y
    """
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    
    if photo.shape[2] == 3:
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
    
    top_x = int(cx - pw / 2)
    top_y = int(cy - ph / 2)
    
    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = pw, ph
    dst_x1, dst_y1 = top_x, top_y
    dst_x2, dst_y2 = top_x + pw, top_y + ph
    
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
        return canvas, top_x, top_y
    
    canvas_f = canvas.astype(np.float32) / 255.0
    photo_f = photo[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w].astype(np.float32) / 255.0
    
    alpha = photo_f[:, :, 3:4]
    canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, :3] = (
        photo_f[:, :, :3] * alpha + 
        canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, :3] * (1 - alpha)
    ).astype(np.float32)
    canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, 3] = np.maximum(
        canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, 3],
        photo_f[:, :, 3]
    )
    
    return (canvas_f * 255).astype(np.uint8), top_x, top_y


def place_marker_at_photo_corner(photo, corner_idx, size=15):
    """Place a marker at a specific photo corner."""
    h, w = photo.shape[:2]
    result = photo.copy()
    
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 0, 0)]
    positions = [
        (size, size),
        (w - size - 1, size),
        (w - size - 1, h - size - 1),
        (size, h - size - 1),
    ]
    
    x, y = positions[corner_idx]
    cv2.circle(result, (x, y), size, colors[corner_idx], -1)
    cv2.line(result, (x - size, y), (x + size, y), (255, 255, 255), 2)
    cv2.line(result, (x, y - size), (x, y + size), (255, 255, 255), 2)
    
    return result


def detect_marker(photo, corner_idx):
    """Detect a specific colored marker."""
    img_check = photo[:, :, :3] if len(photo.shape) > 2 else photo
    
    ranges = {
        0: (np.array([0, 200, 0]), np.array([100, 255, 100])),    # Green
        1: (np.array([0, 0, 200]), np.array([100, 100, 255])),    # Red
        2: (np.array([200, 0, 200]), np.array([255, 100, 255])),  # Magenta
        3: (np.array([200, 0, 0]), np.array([255, 100, 100])),     # Blue
    }
    
    lower, upper = ranges[corner_idx]
    mask = cv2.inRange(img_check, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 25:
            M = cv2.moments(largest)
            if M['m00'] > 0:
                return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return None


def calculate_marker_position_in_canvas(photo_w, photo_h, rotation, corner_idx,
                                         canvas_cx, canvas_cy, new_w, new_h):
    """
    Calculate where a marker should be in canvas space.
    
    This uses the same math that rotate_photo() and composite use.
    """
    # Marker position in photo space (relative to photo center)
    marker_offset = 15
    corners_photo = [
        (marker_offset, marker_offset),                      # TL
        (photo_w - marker_offset - 1, marker_offset),         # TR
        (photo_w - marker_offset - 1, photo_h - marker_offset - 1),  # BR
        (marker_offset, photo_h - marker_offset - 1),          # BL
    ]
    
    mx, my = corners_photo[corner_idx]
    pcx, pcy = photo_w / 2, photo_h / 2  # Photo center
    
    # Rotate using OpenCV's getRotationMatrix2D formula
    # The matrix uses: new_x = cos*x + sin*y + offset
    # NOT: cos*(x-cx) - sin*(y-cy) + cx
    
    # Use the matrix directly (same as get_rotated_polygon)
    M = cv2.getRotationMatrix2D((pcx, pcy), rotation, 1.0)
    
    # The rotated photo will have new dimensions
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    rot_w = int(photo_h * sin_a + photo_w * cos_a)
    rot_h = int(photo_h * cos_a + photo_w * sin_a)
    
    # Adjust matrix for new canvas origin
    M_adj = M.copy()
    M_adj[0, 2] += (rot_w - photo_w) / 2
    M_adj[1, 2] += (rot_h - photo_h) / 2
    
    # Apply to marker
    pt = np.array([mx, my, 1])
    rot = M_adj @ pt
    
    # Photo top-left in canvas
    photo_tl_x = canvas_cx - rot_w / 2
    photo_tl_y = canvas_cy - rot_h / 2
    
    # Final position in canvas
    canvas_x = photo_tl_x + rot[0]
    canvas_y = photo_tl_y + rot[1]
    
    return (canvas_x, canvas_y)


def calculate_polygon_corner_in_canvas(photo_w, photo_h, rotation, corner_idx,
                                       canvas_cx, canvas_cy, new_w, new_h):
    """
    Calculate where get_rotated_polygon() says a corner should be.
    
    Uses the matrix directly (same as get_rotated_polygon).
    """
    pcx, pcy = photo_w / 2, photo_h / 2
    
    # Use the matrix directly
    M = cv2.getRotationMatrix2D((pcx, pcy), rotation, 1.0)
    
    # Calculate rotated dimensions
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    rot_w = int(photo_h * sin_a + photo_w * cos_a)
    rot_h = int(photo_h * cos_a + photo_w * sin_a)
    
    # Adjust matrix for new canvas origin
    M_adj = M.copy()
    M_adj[0, 2] += (rot_w - photo_w) / 2
    M_adj[1, 2] += (rot_h - photo_h) / 2
    
    # Photo corners in its own coordinate system
    hw, hh = photo_w / 2, photo_h / 2
    corners = [
        (pcx - hw, pcy - hh),  # TL
        (pcx + hw, pcy - hh),  # TR
        (pcx + hw, pcy + hh),  # BR
        (pcx - hw, pcy + hh),  # BL
    ]
    
    cx, cy = corners[corner_idx]
    pt = np.array([cx, cy, 1])
    rot = M_adj @ pt
    
    # Photo top-left in canvas
    photo_tl_x = canvas_cx - rot_w / 2
    photo_tl_y = canvas_cy - rot_h / 2
    
    # Final position in canvas
    canvas_x = photo_tl_x + rot[0]
    canvas_y = photo_tl_y + rot[1]
    
    return (canvas_x, canvas_y)


def run_debug():
    """Run debug comparison."""
    print("=" * 70)
    print("DEBUG: Corner Position Comparison")
    print("=" * 70)
    
    # Load test image
    sources = list(Path("./images").glob('*.jpg'))
    if not sources:
        print("ERROR: No images found")
        return
    
    photo = cv2.imread(str(sources[0]))
    
    # Parameters
    PHOTO_W = 250
    PHOTO_H = 200
    CENTER = (300, 300)  # In 0-640 space
    ROTATION = 20
    PADDING = 300
    CANVAS_SIZE = 640 + 2 * PADDING
    
    # Resize photo
    photo = cv2.resize(photo, (PHOTO_W, PHOTO_H))
    
    # Canvas setup
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 4), dtype=np.uint8)
    canvas[:, :, :3] = 180
    canvas[:, :, 3] = 255
    
    # Canvas center (with padding offset)
    canvas_cx = CENTER[0] + PADDING
    canvas_cy = CENTER[1] + PADDING
    
    # Rotate
    rotated, old_size, new_size, center, M, M_adj = rotate_photo_with_details(photo, ROTATION)
    print(f"\nOriginal size: {old_size}")
    print(f"Rotated size: {new_size}")
    print(f"Center: {center}")
    print(f"M matrix:\n{M}")
    print(f"M_adj matrix:\n{M_adj}")
    
    # Calculate where corners should be using correct formula
    correct_positions = []
    polygon_positions = []
    
    for corner_idx in range(4):
        # Correct calculation (matching what rotate_photo does)
        correct = calculate_marker_position_in_canvas(
            PHOTO_W, PHOTO_H, ROTATION, corner_idx,
            canvas_cx, canvas_cy, new_size[0], new_size[1]
        )
        correct_positions.append(correct)
        
        # Polygon calculation (what get_rotated_polygon does)
        polygon = calculate_polygon_corner_in_canvas(
            PHOTO_W, PHOTO_H, ROTATION, corner_idx,
            canvas_cx, canvas_cy, new_size[0], new_size[1]
        )
        polygon_positions.append(polygon)
    
    # Place markers and composite
    photo_with_markers = place_marker_at_photo_corner(photo.copy(), 0)
    photo_with_markers = place_marker_at_photo_corner(photo_with_markers, 1)
    photo_with_markers = place_marker_at_photo_corner(photo_with_markers, 2)
    photo_with_markers = place_marker_at_photo_corner(photo_with_markers, 3)
    
    # Rotate the photo WITH markers (using same function)
    rotated_with_markers, old_size, new_size, center, M, M_adj = rotate_photo_with_details(photo_with_markers, ROTATION)
    
    print(f"\nOriginal size: {old_size}")
    print(f"Rotated size: {new_size}")
    print(f"Center: {center}")
    print(f"M matrix:\n{M}")
    print(f"M_adj matrix:\n{M_adj}")
    
    # Calculate where corners should be using matrix (same as get_rotated_polygon)
    correct_positions = []
    polygon_positions = []
    
    for corner_idx in range(4):
        # Both calculations now use the matrix directly
        correct = calculate_marker_position_in_canvas(
            PHOTO_W, PHOTO_H, ROTATION, corner_idx,
            canvas_cx, canvas_cy, new_size[0], new_size[1]
        )
        correct_positions.append(correct)
        
        polygon = calculate_polygon_corner_in_canvas(
            PHOTO_W, PHOTO_H, ROTATION, corner_idx,
            canvas_cx, canvas_cy, new_size[0], new_size[1]
        )
        polygon_positions.append(polygon)
    
    # Composite using rotated photo WITH markers
    canvas_composite, tl_x, tl_y = composite_with_details(canvas.copy(), rotated_with_markers, canvas_cx, canvas_cy)
    
    print(f"\nPhoto top-left in canvas: ({tl_x}, {tl_y})")
    print(f"Canvas center: ({canvas_cx}, {canvas_cy})")
    
    # Detect markers
    detected = [detect_marker(canvas_composite, i) for i in range(4)]
    
    print(f"\n{'=' * 50}")
    print("COMPARISON: Detected vs Calculated")
    print("{'=' * 50}")
    print(f"\n{'Corner':<8} {'Detected':<20} {'Correct':<25} {'Polygon':<25} {'D-C Error':<12} {'D-P Error':<12}")
    print("-" * 100)
    
    for i in range(4):
        d = detected[i] if detected[i] else (0, 0)
        c = correct_positions[i]
        p = polygon_positions[i]
        
        d_c_error = math.sqrt((d[0] - c[0])**2 + (d[1] - c[1])**2) if detected[i] else 999
        d_p_error = math.sqrt((d[0] - p[0])**2 + (d[1] - p[1])**2) if detected[i] else 999
        
        print(f"{i:<8} ({d[0]:<5},{d[1]:<5})       ({c[0]:<7.1f},{c[1]:<7.1f})     ({p[0]:<7.1f},{p[1]:<7.1f})     {d_c_error:<12.1f} {d_p_error:<12.1f}")
    
    # Show which calculation matches
    total_d_c = sum(math.sqrt((detected[i][0] - correct_positions[i][0])**2 + 
                              (detected[i][1] - correct_positions[i][1])**2) 
                   for i in range(4) if detected[i])
    total_d_p = sum(math.sqrt((detected[i][0] - polygon_positions[i][0])**2 + 
                              (detected[i][1] - polygon_positions[i][1])**2) 
                   for i in range(4) if detected[i])
    
    print(f"\nTotal error with CORRECT formula: {total_d_c:.1f}px")
    print(f"Total error with POLYGON formula: {total_d_p:.1f}px")
    
    if total_d_c < total_d_p:
        print("\n✅ CORRECT formula matches detected positions")
        print("   The polygon calculation needs to be FIXED")
    else:
        print("\n✅ POLYGON formula matches detected positions")
    
    # Save composite
    cv2.imwrite('/tmp/debug_composite.png', canvas_composite[:, :, :3])
    print("\nSaved /tmp/debug_composite.png")


if __name__ == '__main__':
    run_debug()