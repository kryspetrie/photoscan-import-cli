#!/usr/bin/env python3
"""
Test corrected polygon calculation.
"""

import cv2
import numpy as np

def get_rotated_polygon_corrected(width, height, center_x, center_y, rotation):
    """
    Calculate rotated rectangle corners - CORRECTED VERSION.
    
    Strategy: 
    1. Get corners in a coordinate frame centered on the photo center
    2. Rotate those corners
    3. Add the canvas center offset
    """
    if abs(rotation) < 1:
        hw, hh = width / 2, height / 2
        return np.array([
            [center_x - hw, center_y - hh],
            [center_x + hw, center_y - hh],
            [center_x + hw, center_y + hh],
            [center_x - hw, center_y + hh]
        ], dtype=np.float32)
    
    # Photo corners relative to photo center
    photo_center = np.array([width / 2, height / 2])
    corners_relative = np.array([
        [-width/2, -height/2],   # TL (x-center, y-center)
        [ width/2, -height/2],   # TR
        [ width/2,  height/2],    # BR
        [-width/2,  height/2]     # BL
    ], dtype=np.float32)
    
    # Rotate around photo center
    angle_rad = np.radians(rotation)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    R = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    
    # Rotate corners
    rotated_relative = corners_relative @ R.T
    
    # Add center offset to get placement space coordinates
    # Note: width and height here are the ORIGINAL photo dimensions
    # The rotation doesn't change where the CENTER is placed, just the corners around it
    photo_center_placement = np.array([center_x, center_y])
    corners_placement = rotated_relative + photo_center_placement
    
    return corners_placement


def get_rotated_polygon_v2(width, height, center_x, center_y, rotation):
    """
    Alternative: Use M matrix but extract correct origin.
    
    M transforms (x, y) -> (M[0,0]*x + M[0,1]*y + M[0,2], M[1,0]*x + M[1,1]*y + M[1,2])
    
    The shifted origin (where (0,0) lands) is at M[:,2] = (M[0,2], M[1,2])
    The rotated center is at photo_center @ R.T + M[:,2]
    
    To map back to placement space:
    1. Transform through M to get "rotated image space" coordinates
    2. Where does the photo center land? photo_center @ R.T + M[:,2]
    3. Origin in "rotated image space" = M[:,2] (where (0,0) in photo space maps to)
    4. Offset = (center) - (rotated center in rotated image space)
    5. Final = transformed corner + offset
    
    But wait, the canvas placement (center_x, center_y) is what we're positioning TO,
    not where the photo center lands in the rotated image.
    
    The correct formula should use the rotation to compute where ORIGINAL photo corners
    map to, then offset by center.
    """
    if abs(rotation) < 1:
        hw, hh = width / 2, height / 2
        return np.array([
            [center_x - hw, center_y - hh],
            [center_x + hw, center_y - hh],
            [center_x + hw, center_y + hh],
            [center_x - hw, center_y + hh]
        ], dtype=np.float32)
    
    photo_center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(height * sin_a + width * cos_a)
    new_h = int(height * cos_a + width * sin_a)
    
    # Add translation for expanded canvas
    M[0, 2] += (new_w - width) / 2
    M[1, 2] += (new_h - height) / 2
    
    # Corners relative to photo center
    corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Transform each corner through M
    corners_rot = np.zeros_like(corners)
    for i in range(4):
        pt = np.array([corners[i, 0], corners[i, 1], 1])
        result = M @ pt
        corners_rot[i] = [result[0], result[1]]
    
    # This is where (0,0) in photo space maps to
    origin = M[:, 2]  # M[0,2], M[1,2]
    
    # This is where photo center maps to
    center_rot = M @ np.array([photo_center[0], photo_center[1], 1])
    
    # The canvas center is at (center_x, center_y)
    # In rotated image space, the photo center is at center_rot
    # So the offset is center - center_rot
    offset_x = center_x - center_rot[0]
    offset_y = center_y - center_rot[1]
    
    return corners_rot + np.array([offset_x, offset_y])


def verify_with_visual():
    """Test by creating a visual representation and checking."""
    from generate_dataset import rotate_photo, composite_photo_at_center
    
    CANVAS_SIZE = 640
    PADDING = 300
    
    width, height = 300, 300
    center_x, center_y = 320 + PADDING, 320 + PADDING
    rotation = 45
    
    print(f"\n{'='*70}")
    print("Verification")
    print(f"{'='*70}")
    print(f"  Center in padded space: ({center_x}, {center_y})")
    print(f"  Expected TL corner (before rotation): ({center_x - 150}, {center_y - 150})")
    
    # Method 1: Simple rotation around center
    result1 = get_rotated_polygon_corrected(width, height, center_x, center_y, rotation)
    print(f"\n  Method 1 (rotate relative):")
    for i, pt in enumerate(result1):
        print(f"    corner {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # Method 2: Using M matrix with offset
    result2 = get_rotated_polygon_v2(width, height, center_x, center_y, rotation)
    print(f"\n  Method 2 (M matrix offset):")
    for i, pt in enumerate(result2):
        print(f"    corner {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # Expected: After 45° rotation, the corners should form a diamond around center
    # The distance from center to each corner = half the diagonal = 150*sqrt(2) = 212px
    print(f"\n  Expected corner positions (approx):")
    dist = 150 * np.sqrt(2)  # distance from center to corner
    print(f"    TL: ({center_x - dist:.0f}, {center_y - dist:.0f})")
    print(f"    TR: ({center_x + dist:.0f}, {center_y - dist:.0f})")
    print(f"    BR: ({center_x + dist:.0f}, {center_y + dist:.0f})")
    print(f"    BL: ({center_x - dist:.0f}, {center_y + dist:.0f})")
    
    return result1, result2


if __name__ == '__main__':
    verify_with_visual()