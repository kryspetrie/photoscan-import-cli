import numpy as np
import cv2
import os
import random
from pathlib import Path
from data_generator.generate_dataset import (
    cv2_perspective_warp, add_drop_shadow, add_variable_blur,
    composite_with_shadow, BackgroundGenerator, apply_luma_gradient
)

random.seed(42)
np.random.seed(42)

os.makedirs("test_output", exist_ok=True)

# Load a test photo
src_dir = Path("data_generator/images")
sources = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
if not sources:
    print("No source images!")
    exit(1)
    
test_photo = cv2.imread(str(sources[0]), cv2.IMREAD_UNCHANGED)
if test_photo is None:
    print("Failed to load image")
    exit(1)

# Convert to BGRA if needed
if test_photo.shape[2] == 3:
    test_photo = cv2.cvtColor(test_photo, cv2.COLOR_BGR2BGRA)

print(f"Loaded image shape: {test_photo.shape}")

# Create background
bg_gen = BackgroundGenerator()
bg = bg_gen.generate(1280, 720)
bg = apply_luma_gradient(bg)

# === WITH EDGE BLUR ===
print("\nProcessing WITH edge blur...")
# First warp the photo
warped_with, corners_with = cv2_perspective_warp(test_photo, strength=0.15)
print(f"Warped shape: {warped_with.shape}")

# Then add variable blur to the warped photo
warped_blurred = add_variable_blur(warped_with.copy(), min_blur=1, max_blur=7, edge_focus=0.6)
print(f"After blur: {warped_blurred.shape}")

# Add shadow - returns (canvas_with_shadow, new_corners, shadow_mask)
canvas_with_shadow, new_corners, shadow_mask = add_drop_shadow(
    warped_blurred, corners_with, offset=3, spread=20, opacity=0.6
)

# Composite onto background (position 200, 200)
result_with = composite_with_shadow(bg.copy(), canvas_with_shadow, shadow_mask, (200, 200))
cv2.imwrite("test_output/scene_WITH_edge_blur.png", result_with)
print("Saved scene_WITH_edge_blur.png")

# === WITHOUT BLUR ===
print("\nProcessing WITHOUT edge blur...")
warped_no_blur, corners_no_blur = cv2_perspective_warp(test_photo, strength=0.15)
canvas_no_shadow, new_corners2, shadow_mask2 = add_drop_shadow(
    warped_no_blur, corners_no_blur, offset=3, spread=20, opacity=0.6
)
result_no_blur = composite_with_shadow(bg.copy(), canvas_no_shadow, shadow_mask2, (200, 200))
cv2.imwrite("test_output/scene_NO_edge_blur.png", result_no_blur)
print("Saved scene_NO_edge_blur.png")

# Save warped photos on white background for clear comparison
white_bg = np.ones((800, 800, 3), dtype=np.uint8) * 255

def overlay_on_white(bg, photo):
    """Simple overlay of photo on white background"""
    h, w = photo.shape[:2]
    result = bg.copy()
    y_offset = (bg.shape[0] - h) // 2
    x_offset = (bg.shape[1] - w) // 2
    
    if photo.shape[2] == 4:
        alpha = photo[:, :, 3:4] / 255.0
        rgb = photo[:, :, :3]
        for c in range(3):
            result[y_offset:y_offset+h, x_offset:x_offset+w, c] = (
                result[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1 - alpha[:, :, 0]) +
                rgb[:, :, c] * alpha[:, :, 0]
            ).astype(np.uint8)
    else:
        result[y_offset:y_offset+h, x_offset:x_offset+w] = photo[:, :, :3]
    return result

cv2.imwrite("test_output/photo_only_WITH_blur.png", overlay_on_white(white_bg, warped_blurred))
cv2.imwrite("test_output/photo_only_NO_blur.png", overlay_on_white(white_bg, warped_no_blur))
print("Saved photo-only comparisons")

# Also save close crops of the edges
def crop_edges(photo, size=150):
    """Crop the 4 edges of the photo for detailed comparison"""
    h, w = photo.shape[:2]
    crops = {}
    crops['top'] = photo[:size, :, :]
    crops['bottom'] = photo[-size:, :, :]
    crops['left'] = photo[:, :size, :]
    crops['right'] = photo[:, -size:, :]
    return crops

crops_blur = crop_edges(warped_blurred)
crops_no_blur = crop_edges(warped_no_blur)

# Create side-by-side comparison
def make_comparison_row(photo_blur, photo_no_blur, label=""):
    """Create a row showing edge crops"""
    h = max(photo_blur.shape[0], photo_no_blur.shape[0])
    w = photo_blur.shape[1] + photo_no_blur.shape[1] + 10
    row = np.ones((h, w, 3), dtype=np.uint8) * 255
    row[:photo_blur.shape[0], :photo_blur.shape[1]] = photo_blur[:, :, :3]
    row[:photo_no_blur.shape[0], photo_blur.shape[1]+10:] = photo_no_blur[:, :, :3]
    return row

# Combine all edge comparisons
top = make_comparison_row(crops_blur['top'], crops_no_blur['top'])
bottom = make_comparison_row(crops_blur['bottom'], crops_no_blur['bottom'])
left = make_comparison_row(crops_blur['left'], crops_no_blur['left'])
right = make_comparison_row(crops_blur['right'], crops_no_blur['right'])

comparison = np.vstack([top, bottom, left, right])
cv2.imwrite("test_output/edge_comparison.png", comparison)
print("Saved edge_comparison.png (top, bottom, left, right edges)")

print("\n=== Done! ===")
print("Files created in test_output/:")
print("  1. scene_WITH_edge_blur.png - Full scene with blur")
print("  2. scene_NO_edge_blur.png - Full scene without blur")
print("  3. photo_only_WITH_blur.png - Photo on white with blur")
print("  4. photo_only_NO_blur.png - Photo on white without blur")
print("  5. edge_comparison.png - Close-up of all 4 edges")
