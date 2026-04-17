#!/usr/bin/env python3
"""Debug why markers are lost after perspective warp."""
import numpy as np
import cv2


def rotate_photo(photo, angle):
    """Rotate a photo by specified degrees."""
    h, w = photo.shape[:2]
    
    if abs(angle) < 1:
        return photo
    
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
    
    return rotated


def debug_pipeline():
    """Debug the pipeline step by step."""
    CS, PD = 640, 300
    pw, ph = 300, 200
    cx, cy = CS//2 + PD, CS//2 + PD
    OUTPUT_SIZE = 640
    angle = 0  # Start simple
    
    print("="*70)
    print("DEBUGGING FULL PIPELINE")
    print("="*70)
    
    # Create photo with markers
    photo = np.ones((ph, pw, 4), dtype=np.uint8) * 180
    photo[:, :, 3] = 255
    
    ms = 10  # Larger markers for visibility
    photo[0:ms, 0:ms] = [0, 0, 255, 255]
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]
    photo[ph-ms:ph, pw-ms:pw] = [255, 0, 0, 255]
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]
    
    print(f"\n1. Original photo: {pw}x{ph}")
    print(f"   Marker positions in photo: TL(0-10,0-10), TR(0-10,{pw-10}-{pw}), BR({ph-10}-{ph},{pw-10}-{pw}), BL({ph-10}-{ph},0-10)")
    
    # Rotate
    rotated = rotate_photo(photo, angle)
    rot_h, rot_w = rotated.shape[:2]
    print(f"\n2. After rotation: {rot_w}x{rot_h}")
    
    # Create canvas and composite
    canvas = np.ones((CS + 2*PD, CS + 2*PD, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    
    top_left_x = int(cx - rot_w / 2)
    top_left_y = int(cy - rot_h / 2)
    print(f"\n3. Composite at center ({cx}, {cy})")
    print(f"   Canvas top-left: ({top_left_x}, {top_left_y})")
    
    result = canvas.copy()
    for y in range(rot_h):
        for x in range(rot_w):
            result[int(top_left_y) + y, int(top_left_x) + x] = rotated[y, x]
    
    # Check composite
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    print(f"\n4. After composite:")
    for lower, upper, name in [([0, 100, 100], [15, 255, 255], "RED"),
                               ([35, 100, 100], [85, 255, 255], "GREEN"),
                               ([100, 100, 100], [130, 255, 255], "BLUE"),
                               ([15, 100, 100], [45, 255, 255], "YELLOW")]:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        points = np.where(mask > 0)
        if len(points[0]) > 0:
            print(f"   {name}: found at {len(points[0])} pixels")
        else:
            print(f"   {name}: NOT FOUND")
    
    # Save composite for inspection
    cv2.imwrite('/tmp/debug_composite.png', result)
    print(f"\n   Saved composite to /tmp/debug_composite.png")
    
    # Apply perspective
    canvas_w, canvas_h = CS + 2*PD, CS + 2*PD
    src_corners = np.array([
        [0, 0], [canvas_w - 1, 0], [canvas_w - 1, canvas_h - 1], [0, canvas_h - 1]
    ], dtype=np.float32)
    
    perspective_strength = 0.12
    max_offset_x = canvas_w * perspective_strength
    max_offset_y = canvas_h * perspective_strength
    
    dst_corners = np.array([
        [max_offset_x * 0.8, max_offset_y * 0.8],
        [canvas_w - 1 - max_offset_x * 0.5, max_offset_y * 0.8],
        [canvas_w - 1, canvas_h - 1],
        [0, canvas_h - 1]
    ], dtype=np.float32)
    
    min_x = min(c[0] for c in dst_corners)
    max_x = max(c[0] for c in dst_corners)
    min_y = min(c[1] for c in dst_corners)
    max_y = max(c[1] for c in dst_corners)
    
    out_w = int(max_x - min_x) + 1
    out_h = int(max_y - min_y) + 1
    
    offset_x = -min_x
    offset_y = -min_y
    dst_offset = dst_corners.copy()
    dst_offset[:, 0] += offset_x
    dst_offset[:, 1] += offset_y
    
    M = cv2.getPerspectiveTransform(src_corners, dst_offset)
    print(f"\n5. Perspective transform:")
    print(f"   Output size: {out_w}x{out_h}")
    print(f"   Offset: ({offset_x}, {offset_y})")
    
    warped = cv2.warpPerspective(
        result, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Check warped
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    print(f"\n6. After perspective warp:")
    for lower, upper, name in [([0, 100, 100], [15, 255, 255], "RED"),
                               ([35, 100, 100], [85, 255, 255], "GREEN"),
                               ([100, 100, 100], [130, 255, 255], "BLUE"),
                               ([15, 100, 100], [45, 255, 255], "YELLOW")]:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        points = np.where(mask > 0)
        if len(points[0]) > 0:
            print(f"   {name}: found at {len(points[0])} pixels")
        else:
            print(f"   {name}: NOT FOUND")
    
    cv2.imwrite('/tmp/debug_warped.png', warped)
    print(f"   Saved warped to /tmp/debug_warped.png")
    
    # Crop
    crop_margin = 60
    crop_x1 = crop_margin
    crop_y1 = crop_margin
    crop_x2 = out_w - crop_margin
    crop_y2 = out_h - crop_margin
    
    print(f"\n7. Crop: x=({crop_x1}, {crop_x2}), y=({crop_y1}, {crop_y2})")
    
    if crop_x2 > crop_x1 + 100 and crop_y2 > crop_y1 + 100:
        cropped = warped[crop_y1:crop_y2, crop_x1:crop_x2]
        print(f"   Cropped size: {cropped.shape[1]}x{cropped.shape[0]}")
        
        # Resize
        final = cv2.resize(cropped, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        print(f"   Resized to: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
        
        # Check final
        hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
        print(f"\n8. After crop and resize:")
        for lower, upper, name in [([0, 100, 100], [15, 255, 255], "RED"),
                                   ([35, 100, 100], [85, 255, 255], "GREEN"),
                                   ([100, 100, 100], [130, 255, 255], "BLUE"),
                                   ([15, 100, 100], [45, 255, 255], "YELLOW")]:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            points = np.where(mask > 0)
            if len(points[0]) > 0:
                print(f"   {name}: found at {len(points[0])} pixels")
            else:
                print(f"   {name}: NOT FOUND")
        
        cv2.imwrite('/tmp/debug_final.png', final)
        print(f"   Saved final to /tmp/debug_final.png")
    else:
        print(f"   Crop too small, skipping")


if __name__ == '__main__':
    debug_pipeline()
