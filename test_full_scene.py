import numpy as np
import cv2
import os
from data_generator.generate_dataset import render_scene

# Test backgrounds
bg_path = "data_generator/backgrounds/wood_table.jpg"
if not os.path.exists(bg_path):
    bg_path = "data_generator/backgrounds/marble.jpg"
if not os.path.exists(bg_path):
    bg_path = "data_generator/backgrounds/fabric.jpg"

# Find an actual background
for root, dirs, files in os.walk("data_generator/backgrounds"):
    for f in files:
        if f.endswith(('.jpg', '.png')):
            bg_path = os.path.join(root, f)
            break

print(f"Using background: {bg_path}")

# Generate a full scene with edge blur
scene, keypoints = render_scene(
    background_path=bg_path,
    num_photos=3,
    output_size=(1920, 1080),
    apply_blur=True  # Enable blur
)

cv2.imwrite("test_scene_WITH_blur.png", scene)
print(f"Saved test_scene_WITH_blur.png")

# Also generate without blur for comparison
scene_no_blur, _ = render_scene(
    background_path=bg_path,
    num_photos=3,
    output_size=(1920, 1080),
    apply_blur=False  # No blur
)

cv2.imwrite("test_scene_NO_blur.png", scene_no_blur)
print(f"Saved test_scene_NO_blur.png")

print("Done! Compare the two images - look at the edges of the photos.")
print("With blur: edges should be soft/diffused")
print("Without blur: edges should be crisp")
