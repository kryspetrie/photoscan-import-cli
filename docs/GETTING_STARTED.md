# Getting Started with Photo Pose Detector

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU works) | NVIDIA GPU with 4GB+ VRAM |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB free |

### Software Requirements

- **Python 3.9+** (3.10 or 3.11 recommended)
- **CUDA 11.8+** (optional, for GPU training)
- **Java 17+** (for Kotlin integration)
- **Android Studio / IntelliJ IDEA** (for Kotlin development)

---

## Step 1: Project Setup

### 1.1 Clone or Create Project Directory

```bash
# Navigate to your workspace
cd /path/to/your/workspace

# Create project directory
mkdir -p photo-pose-detector
cd photo-pose-detector

# Create directory structure
mkdir -p data_generator training export onnx_inference kotlin_integration docs
mkdir -p data/images/train data/images/val data/labels/train data/labels/val
mkdir -p models/photo-corner-detector
```

### 1.2 Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install ultralytics and dependencies
pip install ultralytics torch torchvision
pip install numpy pillow opencv-python
pip install onnx onnxruntime  # For ONNX export testing
```

### 1.3 Verify Installation

```bash
python -c "
import ultralytics
import torch
print(f'Ultralytics version: {ultralytics.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

Expected output:
```
Ultralytics version: 8.x.x
PyTorch version: 2.x.x
CUDA available: True  # or False if no GPU
```

---

## Step 2: Generate Training Data

### 2.1 Navigate to Data Generator

```bash
cd data_generator
```

### 2.2 Create Data Generator Script

Create `generate_dataset.py`:

```python
#!/usr/bin/env python3
"""
Photo Pose Detector - Synthetic Training Data Generator

Generates synthetic images of photographs on tables with ground truth keypoints
for training a YOLO26-pose model to detect photo corners.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
import json

# Configuration
CONFIG = {
    "num_train_images": 800,
    "num_val_images": 200,
    "output_dir": "../data",
    "image_size": (1920, 1080),  # High-res source images
    "num_photos_per_scene": (1, 5),  # Random number of photos per scene
    "min_photo_size": (300, 300),
    "max_photo_size": (800, 800),
    "backgrounds_dir": "./backgrounds",  # Optional: custom backgrounds
}

# YOLO Pose Keypoint Indices
KEYPOINT_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
NUM_KEYPOINTS = 4


def create_synthetic_photo(content_image, target_size, rotation, perspective_factor=0.1):
    """
    Create a synthetic photo with perspective distortion.
    
    Args:
        content_image: PIL Image to use as photo content
        target_size: (width, height) tuple
        rotation: Rotation angle in degrees
        perspective_factor: How much perspective distortion (0 = none, 1 = max)
    
    Returns:
        PIL Image with perspective transform applied
    """
    width, height = target_size
    
    # Resize content to fit
    content = content_image.copy()
    content.thumbnail(target_size, Image.LANCZOS)
    
    # Create a canvas for the photo
    photo = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(photo)
    
    # Center the content
    x_offset = (width - content.width) // 2
    y_offset = (height - content.height) // 2
    photo.paste(content, (x_offset, y_offset))
    
    # Add white border (photo frame effect)
    border = 8
    draw.rectangle([0, 0, width-1, height-1], outline=(200, 200, 200), width=border)
    
    # Apply rotation
    if rotation != 0:
        photo = photo.rotate(rotation, expand=True, fillcolor=(128, 128, 128))
    
    # Apply perspective distortion (simulated with slightly off-center crop)
    # This creates a trapezoid-like shape to simulate perspective
    if perspective_factor > 0:
        w, h = photo.size
        # We simulate perspective by just rotating for now
        # True perspective requires more complex warping
        pass
    
    return photo


def create_random_background(size):
    """Create a random background texture."""
    width, height = size
    
    # Random background type
    bg_type = random.choice(["solid_light", "solid_dark", "wood", "gradient", "texture"])
    
    if bg_type == "solid_light":
        # Light table surface
        color = (
            random.randint(180, 250),
            random.randint(180, 250),
            random.randint(180, 250)
        )
        bg = Image.new('RGB', size, color)
    
    elif bg_type == "solid_dark":
        # Dark table surface
        color = (
            random.randint(20, 80),
            random.randint(20, 80),
            random.randint(20, 80)
        )
        bg = Image.new('RGB', size, color)
    
    elif bg_type == "wood":
        # Simulated wood grain
        bg = Image.new('RGB', size, (139, 90, 43))
        # Add noise for grain effect
        pixels = np.array(bg)
        noise = np.random.randint(-20, 20, pixels.shape)
        pixels = np.clip(pixels.astype(int) + noise, 0, 255).astype(np.uint8)
        bg = Image.fromarray(pixels)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=2))
    
    elif bg_type == "gradient":
        # Radial or linear gradient
        colors = [
            (random.randint(180, 220), random.randint(180, 220), random.randint(180, 220)),
            (random.randint(100, 150), random.randint(100, 150), random.randint(100, 150)),
        ]
        bg = Image.new('RGB', size, colors[0])
        draw = ImageDraw.Draw(bg)
        for i in range(height):
            ratio = i / height
            r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
            g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
            b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
            draw.line([(0, i), (width, i)], fill=(r, g, b))
    
    else:  # texture
        # Random textured background
        pixels = np.random.randint(
            100, 200, (height, width, 3), dtype=np.uint8
        )
        bg = Image.fromarray(pixels)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Add vignette effect sometimes
    if random.random() < 0.3:
        bg = add_vignette(bg)
    
    return bg


def add_vignette(image, intensity=0.3):
    """Add vignette effect to image."""
    width, height = image.size
    vignette = Image.new('RGB', image.size, (255, 255, 255))
    draw = ImageDraw.Draw(vignette)
    
    cx, cy = width // 2, height // 2
    max_dist = np.sqrt(cx**2 + cy**2)
    
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            factor = 1 - (dist / max_dist) * intensity
            factor = max(0, min(1, factor))
            
            if factor < 1:
                px = image.getpixel((x, y))
                new_px = tuple(int(p * factor) for p in px)
                vignette.putpixel((x, y), new_px)
    
    return vignette


def add_shadow(image, position='random', blur=15, opacity=80):
    """Add drop shadow to an image."""
    # Create shadow
    shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
    shadow.paste((0, 0, 0, opacity), [0, 0, image.width, image.height])
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur))
    
    # Create result with shadow offset
    result = Image.new('RGBA', (image.width + 30, image.height + 30), (0, 0, 0, 0))
    
    # Offset shadow position
    offset_x = random.randint(10, 25) if position == 'random' else 15
    offset_y = random.randint(10, 25) if position == 'random' else 15
    
    # Paste shadow, then image
    result.paste(shadow, (offset_x, offset_y))
    result.paste(image, (0, 0))
    
    return result.convert('RGB')


def place_photo_on_background(background, photo, position=None, rotation=None):
    """Place a photo on the background with optional position/rotation."""
    bg = background.copy()
    
    # Random position if not specified
    if position is None:
        max_x = background.width - photo.width - 20
        max_y = background.height - photo.height - 20
        position = (
            random.randint(20, max(20, max_x)),
            random.randint(20, max(20, max_y))
        )
    
    # Add shadow before placing
    photo_with_shadow = add_shadow(
        photo,
        blur=random.randint(10, 20),
        opacity=random.randint(60, 100)
    )
    
    # Paste photo onto background
    if photo_with_shadow.mode != 'RGB':
        photo_with_shadow = photo_with_shadow.convert('RGB')
    
    bg.paste(photo_with_shadow, position)
    
    return bg


def generate_keypoints_yaml_format(x, y, width, height, confidence=1.0):
    """
    Generate keypoints in YOLO pose format.
    
    YOLO pose format: class_id x_center y_center width height keypoint1_x keypoint1_y keypoint1_v ...
    where v is visibility (0=not visible, 1=visible, 2=hidden but labeled)
    
    For our photo corner detector:
    - 4 keypoints: top_left, top_right, bottom_right, bottom_left
    - All keypoints are visible (v=2 for labeled but possibly occluded, v=1 for visible)
    """
    # Normalized center and size
    x_center = (x + width / 2) / CONFIG["image_size"][0]
    y_center = (y + height / 2) / CONFIG["image_size"][1]
    w_norm = width / CONFIG["image_size"][0]
    h_norm = height / CONFIG["image_size"][1]
    
    # Keypoints in image coordinates (top-left corner)
    kpts = [
        x, y, 2,                      # top_left (x, y, visibility=2)
        x + width, y, 2,             # top_right
        x + width, y + height, 2,    # bottom_right
        x, y + height, 2,            # bottom_left
    ]
    
    # Normalize keypoints
    kpts_normalized = []
    for i in range(0, len(kpts), 3):
        kx = kpts[i] / CONFIG["image_size"][0]
        ky = kpts[i + 1] / CONFIG["image_size"][1]
        kv = kpts[i + 2]  # visibility
        kpts_normalized.extend([kx, ky, kv])
    
    return x_center, y_center, w_norm, h_norm, kpts_normalized


def create_sample_photo_content():
    """Create a sample photo content (placeholder for real image loading)."""
    # Create a random colorful image as placeholder
    width, height = 600, 400
    pixels = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Add some structure
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    for _ in range(5):
        x1, y1 = random.randint(0, width-100), random.randint(0, height-100)
        x2, y2 = x1 + random.randint(50, 150), y1 + random.randint(50, 150)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
    
    # Add some lines
    for _ in range(3):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        draw.line([x1, y1, x2, y2], fill=color, width=3)
    
    return img


def generate_scene():
    """Generate a single scene with photos on a background."""
    # Create background
    background = create_random_background(CONFIG["image_size"])
    
    # Determine number of photos in this scene
    num_photos = random.randint(*CONFIG["num_photos_per_scene"])
    
    photos_info = []  # Store info about each photo
    
    for _ in range(num_photos):
        # Create photo content
        content = create_sample_photo_content()
        
        # Random photo size
        width = random.randint(*CONFIG["min_photo_size"])
        height = random.randint(*CONFIG["min_photo_size"])
        
        # Random rotation (within reasonable range for photos on table)
        rotation = random.uniform(-45, 45)
        
        # Create the photo with content
        photo = create_synthetic_photo(content, (width, height), rotation)
        
        # Track the actual position after rotation
        # Rotation expands the image, so we need to calculate center
        rotated = photo.copy()
        
        # Random position on background
        max_x = CONFIG["image_size"][0] - rotated.width - 20
        max_y = CONFIG["image_size"][1] - rotated.height - 20
        
        if max_x < 20 or max_y < 20:
            continue  # Skip if photo is too large
        
        pos_x = random.randint(20, max(20, max_x))
        pos_y = random.randint(20, max(20, max_y))
        
        # Place photo on background
        background = place_photo_on_background(background, rotated, (pos_x, pos_y))
        
        # Generate keypoints (in rotated image coordinates relative to original position)
        # For a simple approach, we use the bounding box corners
        x_center, y_center, w_norm, h_norm, kpts = generate_keypoints_yaml_format(
            pos_x, pos_y, rotated.width, rotated.height
        )
        
        photos_info.append({
            "x_center": x_center,
            "y_center": y_center,
            "width": w_norm,
            "height": h_norm,
            "keypoints": kpts
        })
    
    # Add some post-processing to the entire scene
    if random.random() < 0.3:
        # Add slight blur (motion blur simulation)
        background = background.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    if random.random() < 0.2:
        # Add noise
        pixels = np.array(background)
        noise = np.random.randint(-15, 15, pixels.shape)
        pixels = np.clip(pixels.astype(int) + noise, 0, 255).astype(np.uint8)
        background = Image.fromarray(pixels)
    
    return background, photos_info


def save_yolo_label(label_path, photos_info):
    """Save annotations in YOLO format."""
    with open(label_path, 'w') as f:
        for photo in photos_info:
            # YOLO format: class_id x_center y_center width height keypoints...
            # class_id = 0 (photo_corner)
            line = f"0 {photo['x_center']:.6f} {photo['y_center']:.6f} {photo['width']:.6f} {photo['height']:.6f}"
            for kp in photo['keypoints']:
                line += f" {kp:.6f}"
            line += "\n"
            f.write(line)


def generate_dataset():
    """Generate the full dataset."""
    output_dir = Path(CONFIG["output_dir"])
    
    train_dir = output_dir / "images" / "train"
    val_dir = output_dir / "images" / "val"
    train_labels = output_dir / "labels" / "train"
    val_labels = output_dir / "labels" / "val"
    
    for d in [train_dir, val_dir, train_labels, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Generate training images
    print(f"Generating {CONFIG['num_train_images']} training images...")
    for i in range(CONFIG["num_train_images"]):
        image, photos_info = generate_scene()
        
        # Save image
        image_path = train_dir / f"train_{i:05d}.jpg"
        image.save(image_path, quality=95)
        
        # Save label
        label_path = train_labels / f"train_{i:05d}.txt"
        save_yolo_label(label_path, photos_info)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{CONFIG['num_train_images']}")
    
    # Generate validation images
    print(f"Generating {CONFIG['num_val_images']} validation images...")
    for i in range(CONFIG["num_val_images"]):
        image, photos_info = generate_scene()
        
        # Save image
        image_path = val_dir / f"val_{i:05d}.jpg"
        image.save(image_path, quality=95)
        
        # Save label
        label_path = val_labels / f"val_{i:05d}.txt"
        save_yolo_label(label_path, photos_info)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{CONFIG['num_val_images']}")
    
    # Save dataset configuration
    dataset_yaml = f"""
# YOLO Pose Dataset Configuration for Photo Corner Detection
path: {output_dir.absolute()}
train: images/train
val: images/val

# Keypoint definitions (4 corners of a photo)
kpt_shape: [{NUM_KEYPOINTS}, 2]  # {NUM_KEYPOINTS} keypoints, each with (x, y)
flip_idx: [1, 0, 3, 2]  # Left-right flip mapping for corners

# Number of classes (1 = photo, 0 = background)
names:
  0: photo_corner

# Pose configuration
pose:
  flip_idx: [1, 0, 3, 2]
  single_point: False
"""
    
    yaml_path = Path("../training/dataset.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(dataset_yaml)
    
    print(f"\nDataset generation complete!")
    print(f"  Training images: {CONFIG['num_train_images']}")
    print(f"  Validation images: {CONFIG['num_val_images']}")
    print(f"  Dataset config: {yaml_path.absolute()}")


if __name__ == "__main__":
    generate_dataset()
```

### 2.3 Run Data Generation

```bash
cd photo-pose-detector/data_generator
python generate_dataset.py
```

Expected output:
```
Generating 800 training images...
  Generated 100/800
  Generated 200/800
  ...
  Generated 800/800
Generating 200 validation images...
  Generated 50/200
  ...
  Generated 200/200

Dataset generation complete!
  Training images: 800
  Validation images: 200
  Dataset config: /path/to/photo-pose-detector/training/dataset.yaml
```

### 2.4 Verify Generated Data

```bash
cd photo-pose-detector

# Check image count
ls -la data/images/train | head -10
ls -la data/images/train | wc -l

# Check label count
ls -la data/labels/train | wc -l

# View a sample label
cat data/labels/train/train_00000.txt
```

Sample label format:
```
0 0.451234 0.512345 0.234567 0.345678 0.234123 0.156789 2 0.789234 0.178901 2 0.812345 0.891234 2 0.201234 0.845678 2
```
Where:
- `0` = class ID (photo_corner)
- `0.451234 0.512345` = center x, y (normalized)
- `0.234567 0.345678` = width, height (normalized)
- Then 4 keypoints, each with x, y, visibility (v=2 means labeled)

---

## Step 3: Train the Model

### 3.1 Create Training Script

Create `training/train.py`:

```python
#!/usr/bin/env python3
"""
Photo Pose Detector - YOLO26 Pose Training

Fine-tunes YOLO26n-pose for photo corner detection.
"""

import os
from ultralytics import YOLO

def train():
    # Load pretrained YOLO26n-pose model
    model = YOLO("yolo26n-pose.pt")
    
    # Train on custom dataset
    results = model.train(
        # Dataset configuration
        data="dataset.yaml",
        
        # Training parameters
        epochs=100,           # Number of epochs
        patience=20,          # Early stopping patience
        batch=16,             # Batch size (adjust for GPU memory)
        imgsz=640,            # Input image size
        
        # Model save settings
        project="runs/pose",
        name="photo-corner-detector",
        exist_ok=True,
        
        # Optimization (reduced for small custom dataset)
        optimizer="auto",
        lr0=0.001,            # Initial learning rate
        lrf=0.01,             # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation (reduced for corner detection)
        augmentation=True,
        mosaic=0.5,           # Reduce from default 0.9
        mixup=0.0,            # Disable for pose
        copy_paste=0.0,       # Disable for pose
        scale=0.5,            # Reduce scale variation
        degrees=10.0,          # Allow some rotation
        translate=0.1,        # Allow position variation
        flipud=0.0,           # No vertical flip (photos aren't upside down)
        fliplr=0.5,          # Allow horizontal flip
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        
        # Training settings
        pretrained=True,
        close_mosaic=10,     # Disable mosaic in last 10 epochs
        workers=8,
        device=0,             # GPU device, or 'cpu'
        
        # Logging
        verbose=True,
        val=True,
        plots=True,
    )
    
    print("\nTraining complete!")
    print(f"Best model: runs/pose/photo-corner-detector/weights/best.pt")
    print(f"Last model: runs/pose/photo-corner-detector/weights/last.pt")
    
    return results


if __name__ == "__main__":
    train()
```

### 3.2 Run Training

```bash
cd photo-pose-detector/training

# Activate virtual environment if not already
source ../venv/bin/activate

# Run training
python train.py
```

Expected output:
```
Epoch 1/100 [00:05<10:00, it/s 12.50] loss/box: 1.234 loss/cls: 0.567 loss/dfl: 0.890 loss/kpt: 1.234
Epoch 2/100 [00:10<09:55, it/s 12.45] loss/box: 1.012 loss/cls: 0.456 loss/dfl: 0.789 loss/kpt: 1.012
...
Epoch 100/100 [08:30<00:00, it/s 12.50] loss/box: 0.234 loss/cls: 0.123 loss/dfl: 0.234 loss/kpt: 0.345

Training complete!
```

### 3.3 Monitor Training

Training results are saved to `runs/pose/photo-corner-detector/`:
- `weights/best.pt` - Best model based on validation metrics
- `weights/last.pt` - Last checkpoint
- `results.csv` - Training metrics over time
- `results.png` - Training curves
- `val_batch*_pred.jpg` - Validation predictions

### 3.4 Training Tips for Small Datasets

If you have < 1000 images, consider these adjustments:

```python
# In train.py
results = model.train(
    # ... other params
    
    # Reduce augmentation more aggressively
    mosaic=0.3,
    scale=0.3,
    degrees=5.0,
    
    # Lower learning rate
    lr0=0.0005,
    
    # Freeze early layers
    freeze=10,  # Freeze first 10 layers
    
    # More epochs with early stopping
    epochs=150,
    patience=30,
)
```

---

## Step 4: Export to ONNX

### 4.1 Create Export Script

Create `export/export_onnx.py`:

```python
#!/usr/bin/env python3
"""
Photo Pose Detector - ONNX Export

Exports the trained model to ONNX format for cross-platform deployment.
"""

import os
from ultralytics import YOLO


def export_to_onnx(model_path="runs/pose/photo-corner-detector/weights/best.pt", 
                   output_dir="../models/photo-corner-detector"):
    """Export trained model to ONNX format."""
    
    # Load trained model
    model = YOLO(model_path)
    
    # Export to ONNX
    success = model.export(
        format="onnx",
        dynamic=True,         # Dynamic input sizes
        simplify=True,        # Simplify ONNX graph
        opset=12,             # ONNX opset version
        imgsz=640,            # Input size
    )
    
    print(f"\nExport complete!")
    print(f"ONNX model: {success}")
    
    # Move to desired location
    import shutil
    os.makedirs(output_dir, exist_ok=True)
    onnx_name = os.path.basename(success)
    final_path = os.path.join(output_dir, onnx_name)
    shutil.move(success, final_path)
    
    print(f"Moved to: {final_path}")
    return final_path


if __name__ == "__main__":
    export_to_onnx()
```

### 4.2 Run Export

```bash
cd photo-pose-detector/export

python export_onnx.py
```

Expected output:
```
Exporting YOLO26n-pose model to ONNX format...
ONNX export success ✅, saved as runs/pose/photo-corner-detector/weights/best.onnx

Export complete!
ONNX model: runs/pose/photo-corner-detector/weights/best.onnx
Moved to: ../models/photo-corner-detector/best.onnx
```

### 4.3 Verify ONNX Model

```bash
# Check file size
ls -lh models/photo-corner-detector/best.onnx

# Verify ONNX model (requires onnxruntime)
python -c "
import onnx
model = onnx.load('../models/photo-corner-detector/best.onnx')
print(f'ONNX model loaded successfully')
print(f'IR version: {model.ir_version}')
print(f'Opset version: {model.opset_import[0].version}')
print(f'Inputs: {[i.name for i in model.graph.input]}')
print(f'Outputs: {[o.name for o in model.graph.output]}')
"
```

---

## Step 5: Test ONNX Inference (Python)

### 5.1 Create Inference Script

Create `onnx_inference/infer.py`:

```python
#!/usr/bin/env python3
"""
Photo Pose Detector - ONNX Inference Testing

Tests ONNX model inference and keypoint extraction.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import cv2


class PhotoCornerDetector:
    """ONNX-based photo corner detector."""
    
    def __init__(self, model_path, confidence_threshold=0.5):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold
        self.input_size = 640  # YOLO default
        
    def preprocess(self, image):
        """Preprocess image for inference."""
        # Resize to input size
        img = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Transpose to CHW format
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def postprocess(self, output, orig_width, orig_height):
        """Extract keypoints from model output."""
        # Output shape depends on model configuration
        # For pose model: (batch, 4 + num_classes + keypoint_dims, num_predictions)
        # Simplified extraction
        
        output = output[0]  # Remove batch dimension
        
        # This is a simplified extraction - actual implementation
        # needs to match the exact output format of the pose model
        results = []
        
        # For YOLO26-pose, output contains:
        # - Box predictions (4 values)
        # - Class predictions
        # - Keypoint predictions (for 4 keypoints, each with x, y, visibility)
        
        # Find detections above threshold
        # This is a placeholder - actual implementation depends on model output
        # format and whether NMS is included in the model
        
        return results
    
    def detect(self, image_path):
        """Run detection on an image."""
        # Load and preprocess
        image = Image.open(image_path)
        orig_width, orig_height = image.size
        
        input_data = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data})
        
        # Post-process
        detections = self.postprocess(outputs[0], orig_width, orig_height)
        
        return detections, image
    
    def draw_keypoints(self, image, keypoints, save_path=None):
        """Draw keypoints on image."""
        draw = ImageDraw.Draw(image)
        
        colors = ['red', 'green', 'blue', 'yellow']
        keypoint_names = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > self.confidence_threshold:
                color = colors[i % len(colors)]
                # Draw circle
                r = 5
                draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline='black')
                # Draw label
                draw.text((x+5, y-10), f"{keypoint_names[i]}: {conf:.2f}", fill=color)
        
        if save_path:
            image.save(save_path)
        
        return image


def test_on_validation_images():
    """Test detector on validation images."""
    import os
    from pathlib import Path
    
    model_path = "../models/photo-corner-detector/best.onnx"
    val_images = Path("../data/images/val")
    
    detector = PhotoCornerDetector(model_path)
    
    # Test on first 5 images
    image_files = sorted(list(val_images.glob("*.jpg")))[:5]
    
    for img_path in image_files:
        print(f"Processing: {img_path}")
        
        detections, image = detector.detect(str(img_path))
        
        # Draw and save result
        output_path = f"output_{img_path.name}"
        detector.draw_keypoints(image, detections, output_path)
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    test_on_validation_images()
```

---

## Step 6: Kotlin Integration

### 6.1 Project Setup

Create `kotlin_integration/build.gradle.kts`:

```kotlin
plugins {
    kotlin("jvm") version "1.9.22"
    application
}

repositories {
    mavenCentral()
}

dependencies {
    // ONNX Runtime for inference
    implementation("org.onnxruntime:onnxruntime:1.17.0")
    
    // BoofCV for image processing
    implementation("boofcv:boofcv-core:0.40")
    implementation("boofcv:boofcv-android:0.40")
    
    // JavaCV for image loading
    implementation("org.bytedeco:javacv-platform:1.5.9")
    
    // Testing
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
}

application {
    mainClass.set("MainKt")
}

kotlin {
    jvmToolchain(17)
}
```

### 6.2 Create Kotlin Detection Class

Create `kotlin_integration/src/main/kotlin/PhotoCornerDetector.kt`:

```kotlin
package photopose

import boofcv.alg.distort.DistortImageOps
import boofcv.alg.interpolate.InterpolateType
import boofcv.io.image.ConvertBufferedImage
import boofcv.struct.border.BorderType
import boofcv.struct.image.GrayF32
import boofcv.struct.image.InterleavedF32
import georegression.struct.homography.ModelHomography2D_F32
import georegression.struct.point.Point2D_F32
import org.onnxruntime.OnnxRuntime
import java.awt.image.BufferedImage
import java.io.File
import java.nio.FloatBuffer
import javax.imageio.ImageIO
import kotlin.math.sqrt

/**
 * Detected photo with corner keypoints.
 */
data class DetectedPhoto(
    val confidence: Float,
    val corners: List<Point2D_F32>,
    val boundingBox: BoundingBox
)

data class BoundingBox(
    val x: Float, val y: Float,
    val width: Float, val height: Float
)

/**
 * Photo corner detector using ONNX model.
 */
class PhotoCornerDetector(
    private val modelPath: String,
    private val confidenceThreshold: Float = 0.5f
) {
    private val env = OnnxRuntime.getEnv()
    private val session: org.onnxruntime.Session
    
    private val inputSize = 640
    private val numKeypoints = 4
    
    init {
        val options = org.onnxruntime.SessionOptions()
        session = env.createSession(modelPath, options)
    }
    
    /**
     * Preprocess image for ONNX inference.
     */
    private fun preprocess(image: BufferedImage): FloatArray {
        // Resize to input size
        val resized = BufferedImage(inputSize, inputSize, BufferedImage.TYPE_3BYTE_BGR)
        val g = resized.createGraphics()
        g.drawImage(image.getScaledInstance(inputSize, inputSize, java.awt.Image.SCALE_SMOOTH), 0, 0, null)
        g.dispose()
        
        // Convert to float array (CHW format)
        val pixels = FloatArray(3 * inputSize * inputSize)
        val raster = resized.getRaster()
        
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val pixel = raster.getPixel(x, y, FloatArray(3))
                val idx = y * inputSize + x
                pixels[idx] = pixel[0] / 255.0f              // B
                pixels[idx + inputSize * inputSize] = pixel[1] / 255.0f  // G
                pixels[idx + 2 * inputSize * inputSize] = pixel[2] / 255.0f  // R
            }
        }
        
        return pixels
    }
    
    /**
     * Run inference on an image.
     */
    fun detect(image: BufferedImage): List<DetectedPhoto> {
        val inputData = preprocess(image)
        
        val inputTensor = org.onnxruntime.OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(inputData),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )
        
        val outputs = session.run(mapOf("images" to inputTensor))
        val outputBuffer = (outputs[0] as org.onnxruntime.OnnxTensor).floatBuffer
        
        // Post-process output
        return postprocess(outputBuffer, image.width, image.height)
    }
    
    /**
     * Post-process model output to extract keypoints.
     */
    private fun postprocess(output: FloatBuffer, origWidth: Int, origHeight: Int): List<DetectedPhoto> {
        val detections = mutableListOf<DetectedPhoto>()
        
        // This depends on the exact model output format
        // For YOLO26-pose, output is (batch, 4 + num_classes + keypoint_dims, num_predictions)
        // Keypoint dims = 2 (x, y) * 4 keypoints = 8, plus visibility = 12 total
        
        // Simplified extraction - actual implementation needs to match model output
        val numPredictions = 8400  // Common for YOLO models at 640x640
        
        for (i in 0 until minOf(numPredictions, 100)) {  // Limit for demo
            val offset = i * (4 + 1 + 12)  // box + class + keypoints
            
            // Extract box coordinates
            val x = output.get(offset)
            val y = output.get(offset + 1)
            val w = output.get(offset + 2)
            val h = output.get(offset + 3)
            
            // Extract confidence
            val conf = output.get(offset + 4)
            
            if (conf < confidenceThreshold) continue
            
            // Extract keypoints
            val keypointOffset = offset + 5
            val corners = mutableListOf<Point2D_F32>()
            
            for (k in 0 until numKeypoints) {
                val kx = output.get(keypointOffset + k * 3)
                val ky = output.get(keypointOffset + k * 3 + 1)
                // Scale to original image size
                corners.add(Point2D_F32(kx * origWidth, ky * origHeight))
            }
            
            detections.add(
                DetectedPhoto(
                    confidence = conf,
                    corners = corners,
                    boundingBox = BoundingBox(
                        x * origWidth, y * origHeight,
                        w * origWidth, h * origHeight
                    )
                )
            )
        }
        
        return detections
    }
    
    /**
     * Apply perspective transform to extract a photo.
     */
    fun extractPhoto(image: BufferedImage, corners: List<Point2D_F32>): BufferedImage {
        require(corners.size == 4) { "Expected 4 corners" }
        
        // Calculate output dimensions based on average edge lengths
        val topWidth = distance(corners[0], corners[1])
        val bottomWidth = distance(corners[3], corners[2])
        val leftHeight = distance(corners[0], corners[3])
        val rightHeight = distance(corners[1], corners[2])
        
        val outWidth = ((topWidth + bottomWidth) / 2).toInt()
        val outHeight = ((leftHeight + rightHeight) / 2).toInt()
        
        // Create destination corners (rectangle)
        val dstCorners = listOf(
            Point2D_F32(0f, 0f),
            Point2D_F32(outWidth.toFloat(), 0f),
            Point2D_F32(outWidth.toFloat(), outHeight.toFloat()),
            Point2D_F32(0f, outHeight.toFloat())
        )
        
        // Estimate homography
        val homography = estimateHomography(corners, dstCorners)
        
        // Apply transform using BoofCV
        val src = ConvertBufferedImage.convertFromSingle(image, InterleavedF32::class.java)
        val dst = InterleavedF32(outWidth, outHeight, 3)
        
        DistortImageOps.distort(
            src, dst, homography,
            InterpolateType.BILINEAR, BorderType.ZERO, null
        )
        
        return ConvertBufferedImage.convertTo(dst, null)
    }
    
    /**
     * Estimate homography matrix between two sets of points.
     */
    private fun estimateHomography(src: List<Point2D_F32>, dst: List<Point2D_F32>): ModelHomography2D_F32 {
        // Simplified - actual implementation uses DLT algorithm
        // For production, use georegression library's HomographyEstimator4Point
        val model = ModelHomography2D_F32()
        
        // Compute using DLT (Direct Linear Transform)
        // This is a simplified placeholder
        model.set(
            floatArrayOf(
                1f, 0f, 0f,
                0f, 1f, 0f,
                0f, 0f, 1f
            )
        )
        
        return model
    }
    
    private fun distance(p1: Point2D_F32, p2: Point2D_F32): Float {
        val dx = p2.x - p1.x
        val dy = p2.y - p1.y
        return sqrt(dx * dx + dy * dy)
    }
    
    fun close() {
        session.close()
        env.close()
    }
}

fun main() {
    // Load model
    val modelPath = "models/photo-corner-detector/best.onnx"
    val detector = PhotoCornerDetector(modelPath)
    
    // Load test image
    val imagePath = "data/images/val/val_00000.jpg"
    val image = ImageIO.read(File(imagePath))
    
    // Detect corners
    val detections = detector.detect(image)
    
    println("Found ${detections.size} photos")
    
    for ((i, photo) in detections.withIndex()) {
        println("Photo $i:")
        println("  Confidence: ${photo.confidence}")
        println("  Corners:")
        photo.corners.forEachIndexed { j, corner ->
            println("    ${j}: (${corner.x}, ${corner.y})")
        }
        
        // Extract and save
        if (photo.confidence > 0.7f) {
            val extracted = detector.extractPhoto(image, photo.corners)
            ImageIO.write(extracted, "jpg", File("output_photo_$i.jpg"))
            println("  Saved: output_photo_$i.jpg")
        }
    }
    
    detector.close()
}
```

---

## Step 7: Troubleshooting

### Issue: CUDA Out of Memory

```python
# Reduce batch size in train.py
results = model.train(
    batch=8,  # or even 4
    ...
)
```

### Issue: Model Overfitting

```python
# Reduce augmentation in train.py
mosaic=0.3,
scale=0.3,
mixup=0.0,
copy_paste=0.0,
```

### Issue: Low Keypoint Accuracy

1. Verify annotations are correct:
   ```bash
   python -c "
   from PIL import Image, ImageDraw
   img = Image.open('data/images/train/train_00000.jpg')
   draw = ImageDraw.Draw(img)
   with open('data/labels/train/train_00000.txt') as f:
       for line in f:
           parts = line.strip().split()
           # Parse and draw keypoints
           ...
   img.save('debug_annotation.jpg')
   "
   ```

2. Generate more training data

3. Increase training epochs

### Issue: ONNX Export Fails

```bash
# Install onnxruntime
pip install onnxruntime

# Or with GPU support
pip install onnxruntime-gpu
```

### Issue: Kotlin ONNX Runtime Crashes

- Ensure native libraries are properly loaded
- Check that the ONNX model opset version is supported (12+)
- Verify the model was exported correctly

---

## Next Steps

1. **Evaluate Model Performance**
   - Run validation on test set
   - Measure keypoint accuracy
   - Adjust confidence threshold

2. **Improve Model**
   - Generate more training data
   - Fine-tune hyperparameters
   - Try larger model (yolo26s-pose)

3. **Integrate with Your App**
   - Add to your existing Kotlin project
   - Implement UI for displaying results
   - Add perspective transform preview

4. **Optimize for Deployment**
   - Consider TensorRT export for NVIDIA
   - Quantize model for mobile
   - Add batch processing for videos

---

## Support

If you encounter issues:
1. Check the Ultralytics documentation
2. Verify your dataset format
3. Test with pretrained model first
4. Check training logs for errors
