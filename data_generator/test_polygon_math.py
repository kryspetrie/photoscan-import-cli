#!/usr/bin/env python3
"""
Test the polygon calculation directly to find the bug.
"""

import cv2
import numpy as np

def get_rotated_polygon_fixed(width, height, center_x, center_y, rotation):
    """
    Clean implementation of rotated rectangle corners.
    """
    if abs(rotation) < 1:
        hw, hh = width / 2, height / 2
        return np.array([
            [center_x - hw, center_y - hh],
            [center_x + hw, center_y - hh],
            [center_x + hw, center_y + hh],
            [center_x - hw, center_y + hh]
        ], dtype=np.float32)
    
    # Build rotation matrix around photo center (like rotate_photo)
    photo_center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    # Calculate rotated dimensions (same as rotate_photo)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(height * sin_a + width * cos_a)
    new_h = int(height * cos_a + width * sin_a)
    
    # Add the translation offset for canvas expansion
    M[0, 2] += (new_w - width) / 2
    M[1, 2] += (new_h - height) / 2
    
    print(f"    photo_center: {photo_center}")
    print(f"    new_w, new_h: {new_w}, {new_h}")
    print(f"    M matrix:\n{M}")
    
    # Corners in PHOTO SPACE (at the edges of the original photo)
    # These are at offsets from (width/2, height/2)
    corners_photo = np.array([
        [0, 0],              # TL
        [width, 0],          # TR
        [width, height],     # BR
        [0, height]          # BL
    ], dtype=np.float32)
    
    # Transform corners through M to get their positions in rotated image
    corners_rotated = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        result = M @ pt
        corners_rotated[i] = [result[0], result[1]]
        print(f"    corner {i}: photo={corners_photo[i]} -> rotated={corners_rotated[i]}")
    
    # Where does the photo center go in rotated image?
    rotated_center = M @ np.array([photo_center[0], photo_center[1], 1])
    print(f"    rotated_center: {rotated_center}")
    
    # In placement space, origin = center - rotated_center
    # corner_placement = origin + corner_rotated
    corners_final = np.zeros_like(corners_rotated)
    corners_final[:, 0] = center_x - rotated_center[0] + corners_rotated[:, 0]
    corners_final[:, 1] = center_y - rotated_center[1] + corners_rotated[:, 1]
    
    return corners_final


def test_rotation():
    width, height = 300, 300
    center_x, center_y = 620, 620  # In padded canvas space
    rotation = 45
    
    print(f"\n{'='*70}")
    print("Testing get_rotated_polygon")
    print(f"{'='*70}")
    print(f"  width={width}, height={height}")
    print(f"  center=({center_x}, {center_y})")
    print(f"  rotation={rotation}°")
    print()
    
    result = get_rotated_polygon_fixed(width, height, center_x, center_y, rotation)
    print(f"\n  Final polygon:")
    for i, pt in enumerate(result):
        print(f"    {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # For no rotation, check if it matches expected
    hw, hh = width / 2, height / 2
    expected_no_rot = np.array([
        [center_x - hw, center_y - hh],
        [center_x + hw, center_y - hh],
        [center_x + hw, center_y + hh],
        [center_x - hw, center_y + hh]
    ], dtype=np.float32)
    
    print(f"\n  Expected for 0° rotation: {expected_no_rot[0]}")
    
    # Now test if the math makes sense
    # For a 300x300 photo centered at (620, 620), corners should be:
    # TL: (470, 470), TR: (770, 470), BR: (770, 770), LL: (470, 770)
    print(f"\n  Expected physical placement:")
    print(f"    TL: ({center_x - 150}, {center_y - 150}) = ({center_x - 150}, {center_y - 150})")
    print(f"    TR: ({center_x + 150}, {center_y - 150}) = ({center_x + 150}, {center_y - 150})")


def test_rotate_photo():
    """Test the rotate_photo function to see what dimensions it produces."""
    from generate_dataset import rotate_photo
    
    # Create a test photo with a cross pattern
    photo = np.zeros((300, 300, 3), dtype=np.uint8)
    photo[:] = (200, 200, 200)
    # Draw cross at center
    cv2.line(photo, (150, 0), (150, 300), (255, 0, 0), 2)  # Vertical blue
    cv2.line(photo, (0, 150), (300, 150), (0, 255, 0), 2)  # Horizontal green
    
    print(f"\n{'='*70}")
    print("Testing rotate_photo")
    print(f"{'='*70}")
    print(f"  Original: {photo.shape}")
    
    rotated = rotate_photo(photo, 45)
    print(f"  Rotated 45°: {rotated.shape}")
    
    # Save for visual inspection
    cv2.imwrite('/tmp/test_rotate_original.jpg', photo)
    cv2.imwrite('/tmp/test_rotate_45.jpg', rotated)
    print(f"  Saved test images")


if __name__ == '__main__':
    test_rotation()
    test_rotate_photo()