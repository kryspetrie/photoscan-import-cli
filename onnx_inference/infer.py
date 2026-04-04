#!/usr/bin/env python3
"""
Photo Pose Detector - ONNX Inference Testing

Tests ONNX model inference and keypoint extraction.

Usage:
    python infer.py --model ../models/photo-corner-detector/best.onnx --image ../data/images/val/val_00000.jpg
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_onnx_model(model_path):
    """Load ONNX model with onnxruntime."""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return session
    except ImportError:
        print("Error: onnxruntime not installed")
        print("Install with: pip install onnxruntime")
        sys.exit(1)


def preprocess(image, target_size=640):
    """Preprocess image for ONNX inference."""
    # Resize
    img = image.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to float and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Transpose to CHW format
    img_array = img_array.transpose(2, 0, 1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def postprocess_pose(output, conf_threshold=0.5, orig_size=(640, 640)):
    """
    Post-process YOLO pose model output.
    
    Output format for pose model:
    - Shape: (batch, 4 + num_classes + keypoint_dims, num_predictions)
    - For 4 keypoints: (1, 4 + 1 + 12, 8400) = (1, 17, 8400)
    """
    output = output[0]  # Remove batch dimension
    
    results = []
    
    # Parameters
    num_predictions = output.shape[1]
    stride = 17  # 4 (box) + 1 (class) + 12 (4 keypoints * 3)
    
    orig_w, orig_h = orig_size
    
    for i in range(min(num_predictions, 8400)):
        offset = i * stride
        
        # Get box coordinates
        cx = output[offset]
        cy = output[offset + 1]
        w = output[offset + 2]
        h = output[offset + 3]
        
        # Get confidence
        conf = output[offset + 4]
        
        if conf < conf_threshold:
            continue
        
        # Get keypoints
        keypoints = []
        keypoint_offset = offset + 5
        
        for k in range(4):  # 4 corners
            kx = output[keypoint_offset + k * 3]
            ky = output[keypoint_offset + k * 3 + 1]
            visibility = output[keypoint_offset + k * 3 + 2]
            
            if visibility > 0:
                # Scale to original image coordinates
                keypoints.append((kx * orig_w, ky * orig_h, visibility))
            else:
                keypoints.append((0, 0, 0))
        
        results.append({
            'box': (cx * orig_w, cy * orig_h, w * orig_w, h * orig_h),
            'confidence': conf,
            'keypoints': keypoints
        })
    
    return results


def draw_keypoints(image, detections, save_path=None):
    """Draw keypoints on image."""
    draw = ImageDraw.Draw(image)
    
    colors = ['red', 'green', 'blue', 'yellow']
    keypoint_names = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
    
    for det in detections:
        box = det['box']
        corners = det['keypoints']
        
        # Draw bounding box
        x, y, w, h = box
        draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], 
                      outline='cyan', width=2)
        
        # Draw keypoints
        for i, (kx, ky, vis) in enumerate(corners):
            if vis > 0:
                color = colors[i % len(colors)]
                # Draw circle
                r = 8
                draw.ellipse([kx-r, ky-r, kx+r, ky+r], fill=color, outline='black')
                # Draw label
                draw.text((kx+10, ky-10), f"{keypoint_names[i]}", fill=color)
        
        # Draw connecting lines (corners)
        valid_corners = [(c[0], c[1]) for c in corners if c[2] > 0]
        if len(valid_corners) >= 4:
            # Draw quadrilateral
            for j in range(4):
                p1 = valid_corners[j]
                p2 = valid_corners[(j + 1) % 4]
                draw.line([p1, p2], fill='white', width=3)
    
    if save_path:
        image.save(save_path)
        print(f"Saved visualization: {save_path}")
    
    return image


def infer_onnx(model_path, image_path, output_path=None, conf_threshold=0.5):
    """Run inference on an image."""
    
    # Load model
    session = load_onnx_model(model_path)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Model input: {input_name}")
    print(f"Model output: {output_name}")
    
    # Load and preprocess image
    image = Image.open(image_path)
    orig_size = image.size
    print(f"Image size: {orig_size}")
    
    input_data = preprocess(image)
    
    # Run inference
    print("Running inference...")
    outputs = session.run([output_name], {input_name: input_data})
    
    # Post-process
    print("Post-processing...")
    detections = postprocess_pose(outputs[0], conf_threshold, orig_size)
    
    # Print results
    print(f"\nFound {len(detections)} detections:")
    for i, det in enumerate(detections):
        box = det['box']
        conf = det['confidence']
        print(f"\nDetection {i + 1}:")
        print(f"  Confidence: {conf:.4f}")
        print(f"  Bounding box: ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")
        print("  Corners:")
        for j, (kx, ky, vis) in enumerate(det['keypoints']):
            print(f"    {j}: ({kx:.1f}, {ky:.1f}) - visibility: {vis}")
    
    # Draw and save
    if output_path:
        draw_keypoints(image, detections, output_path)
    else:
        # Generate default output path
        base = Path(image_path).stem
        output_path = f"output_{base}_pose.jpg"
        draw_keypoints(image, detections, output_path)
    
    return detections


def main():
    parser = argparse.ArgumentParser(
        description="Test ONNX pose model inference"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="../models/photo-corner-detector/best.onnx",
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="../data/images/val/val_00000.jpg",
        help="Path to test image"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output image path"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Run inference
    infer_onnx(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        conf_threshold=args.conf
    )


if __name__ == "__main__":
    main()
