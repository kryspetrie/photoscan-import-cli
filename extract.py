#!/usr/bin/env python3
"""
Photo Extract — Detect and extract photographs from scanned images.

Uses the two-model YOLO pipeline (detection + pose) to find photos in
scanned images and extract them as clean, perspective-corrected crops.

Modes:
  detect   Run detection model only, output bounding-box crops
  pose     Run pose model (with optional detection pre-filter),
           output perspective-corrected or axis-aligned crops

The pose mode applies a distortion threshold: if the detected quadrilateral
is close enough to axis-aligned (within N pixels per corner), it crops
instead of warping — avoiding unnecessary perspective distortion on
images that are already nearly rectangular.

Usage:
    # Detection mode — axis-aligned bounding boxes
    python extract.py detect --input scan.jpg --output ./extracted/

    # Pose mode — corner keypoints with smart crop/warp
    python extract.py pose --input scan.jpg --output ./extracted/

    # Pose mode with custom distortion threshold
    python extract.py pose --input scan.jpg --output ./extracted/ --threshold 5

    # Pose mode with detection pre-filter (pipeline)
    python extract.py pose --input scan.jpg --output ./extracted/ --use-detection

    # Adjust confidence
    python extract.py pose --input scan.jpg --output ./extracted/ --conf 0.7
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keypoint order from the training configuration:
#   kp0 = LL (Lower-Left), kp1 = UL (Upper-Left),
#   kp2 = UR (Upper-Right), kp3 = LR (Lower-Right)
#
# For perspective warp we need corners in clockwise order starting from
# a top-left corner.  Mapping:
#   dst[0] = UL (kp1)    →  (0, 0)
#   dst[1] = UR (kp2)    →  (w, 0)
#   dst[2] = LR (kp3)    →  (w, h)
#   dst[3] = LL (kp0)    →  (0, h)
KP_UL, KP_UR, KP_LR, KP_LL = 1, 2, 3, 0

# Default model paths (relative to project root)
DEFAULT_DETECTION_MODEL = "models/detection_model.onnx"
DEFAULT_POSE_MODEL = "models/pose_model.onnx"

INPUT_SIZE = 640  # Model input size


# ---------------------------------------------------------------------------
# Model loading and preprocessing
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load an ONNX model, returning the session and input name."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    return session, input_name


def preprocess(image: np.ndarray, input_size: int = INPUT_SIZE):
    """Preprocess an image for YOLO ONNX inference with letterboxing.

    Returns:
        tensor: (1, 3, H, W) float32 normalized to [0, 1]
        scale: (scale_x, scale_y) to map model coords back to original
        pad: (pad_x, pad_y) letterbox padding in model space
        original_size: (orig_w, orig_h)
    """
    orig_h, orig_w = image.shape[:2]

    # Compute letterbox scale (fit within input_size, preserving aspect ratio)
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded canvas (grey 114, matching YOLO default)
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # Convert BGR → RGB, normalize, CHW
    tensor = canvas[:, :, ::-1].astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis, :, :, :]

    # Inverse scale factors: model coords → original image coords
    scale_x = 1.0 / scale
    scale_y = 1.0 / scale

    return tensor, (scale_x, scale_y), (pad_x, pad_y), (orig_w, orig_h)


def model_coords_to_original(x, y, scale, pad):
    """Map coordinates from model space (640×640 letterboxed) back to original image."""
    ox = (x - pad[0]) * scale[0]
    oy = (y - pad[1]) * scale[1]
    return ox, oy


# ---------------------------------------------------------------------------
# Detection model
# ---------------------------------------------------------------------------

def run_detection(image: np.ndarray, session, input_name: str, conf_threshold: float = 0.5):
    """Run the detection model, returning NMS-filtered bounding boxes.

    Returns:
        list of dict with keys: x1, y1, x2, y2, confidence
    """
    tensor, scale, pad, orig_size = preprocess(image)
    outputs = session.run(None, {input_name: tensor})
    output = outputs[0]  # (1, 5, 8400)

    raw_detections = []
    num_anchors = output.shape[2]

    for i in range(num_anchors):
        cx = output[0, 0, i]
        cy = output[0, 1, i]
        w = output[0, 2, i]
        h = output[0, 3, i]
        conf = output[0, 4, i]

        if conf < conf_threshold:
            continue

        # Convert center+size to corners in model space
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Map to original image coords
        ox1, oy1 = model_coords_to_original(x1, y1, scale, pad)
        ox2, oy2 = model_coords_to_original(x2, y2, scale, pad)

        raw_detections.append({
            "x1": ox1, "y1": oy1,
            "x2": ox2, "y2": oy2,
            "confidence": float(conf),
        })

    # Apply NMS
    return _nms(raw_detections, iou_threshold=0.45)


def _nms(detections: list, iou_threshold: float = 0.45) -> list:
    """Non-Maximum Suppression for detection results."""
    if not detections:
        return []

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    keep = []

    for det in detections:
        should_suppress = False
        for kept in keep:
            if _iou(det, kept) > iou_threshold:
                should_suppress = True
                break
        if not should_suppress:
            keep.append(det)

    return keep


def _iou(a: dict, b: dict) -> float:
    """Compute Intersection over Union of two axis-aligned boxes."""
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Pose model
# ---------------------------------------------------------------------------

def run_pose(image: np.ndarray, session, input_name: str, conf_threshold: float = 0.5):
    """Run the pose model, returning detected corners per photo.

    The ONNX pose-model output has shape (1, 300, 18) with this layout
    per row:

        [0:4]  x1, y1, x2, y2     — bounding box (xyxy, pixel coords)
        [4]    objectness confidence
        [5]    class probability    (≈0 for single-class models)
        [6:18] 4 keypoints × (x, y, vis) in order: LL, UL, UR, LR

    Returns:
        list of dict with keys:
            x1, y1, x2, y2  — bounding box in original image coords
            confidence       — detection confidence
            corners          — list of 4 (x, y, vis) in model order: LL, UL, UR, LR
    """
    tensor, scale, pad, orig_size = preprocess(image)
    outputs = session.run(None, {input_name: tensor})
    output = outputs[0]  # (1, 300, 18)

    results = []
    num_detections = output.shape[1]

    for i in range(num_detections):
        conf = output[0, i, 4]
        if conf < conf_threshold:
            continue

        # Bounding box in model space (xyxy format)
        mx1 = output[0, i, 0]
        my1 = output[0, i, 1]
        mx2 = output[0, i, 2]
        my2 = output[0, i, 3]

        # Map box to original coords
        bx1, by1 = model_coords_to_original(mx1, my1, scale, pad)
        bx2, by2 = model_coords_to_original(mx2, my2, scale, pad)

        # Keypoints in model order: LL, UL, UR, LR
        # Offset by 6 (4 box + 1 conf + 1 class), then 3 values per keypoint
        corners = []
        for k in range(4):
            kx = output[0, i, 6 + k * 3]
            ky = output[0, i, 6 + k * 3 + 1]
            kv = output[0, i, 6 + k * 3 + 2]

            # Map to original image coords
            ox, oy = model_coords_to_original(kx, ky, scale, pad)
            corners.append((ox, oy, float(kv)))

        results.append({
            "x1": bx1, "y1": by1,
            "x2": bx2, "y2": by2,
            "confidence": float(conf),
            "corners": corners,  # [LL, UL, UR, LR]
        })

    return results


# ---------------------------------------------------------------------------
# Distortion check & extraction
# ---------------------------------------------------------------------------

def max_quadrilateral_distortion(corners):
    """Compute the maximum pixel distance each corner deviates from the
    nearest point on the axis-aligned bounding box of the quadrilateral.

    If all corners are within threshold pixels of axis-aligned, a simple
    crop will produce the same result as a perspective warp (within threshold).

    Args:
        corners: list of 4 (x, y) tuples in any order

    Returns:
        max_dist: the largest corner-to-rect distance in pixels
    """
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]

    # Axis-aligned bounding box of the four corners
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)

    # For each corner, compute distance to the nearest point on the AAB rect
    max_dist = 0.0
    for cx, cy in corners:
        # Nearest point on the rect boundary to (cx, cy)
        # The rect has edges: left, right, top, bottom
        nx = max(left, min(cx, right))
        ny = max(top, min(cy, bottom))
        # If point is inside the rect, (nx, ny) == (cx, cy)
        # But we want distance to the rect *boundary*, not the interior
        # Actually what we really want: distance from (cx, cy) to the
        # "ideal" axis-aligned rectangle that best fits these corners.
        # The ideal rect has corners at (left,top), (right,top), etc.
        # Distance from corner to the nearest ideal corner:
        dist = min(
            _point_dist(cx, cy, left, top),
            _point_dist(cx, cy, right, top),
            _point_dist(cx, cy, right, bottom),
            _point_dist(cx, cy, left, bottom),
        )
        max_dist = max(max_dist, dist)

    return max_dist


def _point_dist(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def extract_crop(image, corners_ll_ul_ur_lr, margin=0):
    """Simple axis-aligned crop containing all four corners, with optional margin.

    Args:
        image: original image (numpy array, BGR)
        corners_ll_ul_ur_lr: 4 corners in model order [LL, UL, UR, LR]
        margin: extra pixels to add around the crop

    Returns:
        cropped image (numpy array, BGR)
    """
    pts = [(c[0], c[1]) for c in corners_ll_ul_ur_lr]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    left = max(0, int(min(xs)) - margin)
    top = max(0, int(min(ys)) - margin)
    right = min(image.shape[1], int(max(xs)) + margin + 1)
    bottom = min(image.shape[0], int(max(ys)) + margin + 1)

    return image[top:bottom, left:right]


def extract_perspective(image, corners_ll_ul_ur_lr, max_output_dim=4000):
    """Perspective-warp the quadrilateral defined by 4 corners into a rectangle.

    The output rectangle has the same aspect ratio as the detected quad,
    with dimensions derived from the edge lengths.

    Args:
        image: original image (numpy array, BGR)
        corners_ll_ul_ur_lr: 4 corners in model order [LL, UL, UR, LR]
        max_output_dim: maximum dimension for the output image (prevent huge crops)

    Returns:
        warped image (numpy array, BGR)
    """
    # Reorder to clockwise from top-left: UL, UR, LR, LL
    src = np.float32([
        corners_ll_ul_ur_lr[KP_UL][:2],  # UL → top-left
        corners_ll_ul_ur_lr[KP_UR][:2],  # UR → top-right
        corners_ll_ul_ur_lr[KP_LR][:2],  # LR → bottom-right
        corners_ll_ul_ur_lr[KP_LL][:2],  # LL → bottom-left
    ])

    # Compute output dimensions from edge lengths
    top_width = _point_dist(src[0][0], src[0][1], src[1][0], src[1][1])
    bottom_width = _point_dist(src[3][0], src[3][1], src[2][0], src[2][1])
    left_height = _point_dist(src[0][0], src[0][1], src[3][0], src[3][1])
    right_height = _point_dist(src[1][0], src[1][1], src[2][0], src[2][1])

    out_w = int(max(top_width, bottom_width))
    out_h = int(max(left_height, right_height))

    # Cap output size
    if max(out_w, out_h) > max_output_dim:
        scale_factor = max_output_dim / max(out_w, out_h)
        out_w = int(out_w * scale_factor)
        out_h = int(out_h * scale_factor)

    # Ensure minimum size
    out_w = max(out_w, 10)
    out_h = max(out_h, 10)

    dst = np.float32([
        [0, 0],
        [out_w, 0],
        [out_w, out_h],
        [0, out_h],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped


def extract_photo(image, corners_ll_ul_ur_lr, threshold=3.0, margin=0):
    """Decide whether to crop or warp based on distortion threshold.

    If every corner of the detected quadrilateral is within `threshold` pixels
    of the nearest ideal axis-aligned corner, we do a simple crop.  Otherwise,
    we apply a perspective warp to correct the distortion.

    Args:
        image: original image (numpy array, BGR)
        corners_ll_ul_ur_lr: 4 corners in model order [LL, UL, UR, LR]
            Each corner is (x, y, visibility) or (x, y).
        threshold: max pixel distance from axis-aligned to trigger warp (default 3)
        margin: extra pixels around crop (only used for axis-aligned crop)

    Returns:
        (extracted_image, method_str):
            extracted_image — the cropped or warped image
            method_str      — "crop" or "warp"
    """
    # Strip visibility, keep just (x, y)
    pts = [(c[0], c[1]) for c in corners_ll_ul_ur_lr]

    distortion = max_quadrilateral_distortion(pts)

    if distortion <= threshold:
        result = extract_crop(image, corners_ll_ul_ur_lr, margin=margin)
        return result, "crop"
    else:
        result = extract_perspective(image, corners_ll_ul_ur_lr)
        return result, "warp"


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def draw_detection_boxes(image, detections, color=(0, 255, 0)):
    """Draw detection bounding boxes on an image."""
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{det['confidence']:.2f}"
        cv2.putText(output, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return output


def draw_pose_corners(image, poses, draw_quad=True):
    """Draw pose model keypoints and quadrilaterals on an image."""
    output = image.copy()
    corner_names = ["LL", "UL", "UR", "LR"]
    corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for pose in poses:
        corners = pose["corners"]
        pts = [(int(c[0]), int(c[1])) for c in corners]

        # Draw keypoints
        for i, (pt, name, color) in enumerate(zip(pts, corner_names, corner_colors)):
            cv2.circle(output, pt, 6, color, -1)
            cv2.circle(output, pt, 6, (255, 255, 255), 1)
            cv2.putText(output, name, (pt[0] + 8, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw quadrilateral (UL → UR → LR → LL → UL)
        if draw_quad and all(c[2] > 0.3 for c in corners):
            quad = [pts[KP_UL], pts[KP_UR], pts[KP_LR], pts[KP_LL]]
            cv2.polylines(output, [np.array(quad)], True, (0, 255, 0), 2)

        # Confidence label
        cv2.putText(output, f"{pose['confidence']:.2f}",
                     (int(pose["x1"]), int(pose["y1"]) - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return output


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def find_model(mode, model_arg=None):
    """Resolve model path, checking argument then default locations."""
    if model_arg:
        p = Path(model_arg)
        if p.exists():
            return str(p)
        print(f"Error: Model not found: {model_arg}")
        sys.exit(1)

    # Try default locations relative to this script's directory
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "models" / ("detection_model.onnx" if mode == "detect" else "pose_model.onnx"),
        script_dir / ".." / "models" / ("detection_model.onnx" if mode == "detect" else "pose_model.onnx"),
    ]

    for c in candidates:
        if c.exists():
            return str(c.resolve())

    default_name = "detection_model.onnx" if mode == "detect" else "pose_model.onnx"
    print(f"Error: Cannot find {default_name}")
    print(f"  Searched: {[str(c) for c in candidates]}")
    print(f"  Use --model to specify the path explicitly.")
    sys.exit(1)


def process_detect(args):
    """Process images using detection model only."""
    session, input_name = load_model(args.model)

    image_paths = resolve_inputs(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        print(f"\n{'=' * 50}")
        print(f"Processing: {img_path}")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Warning: Could not read {img_path}, skipping.")
            continue

        detections = run_detection(image, session, input_name, conf_threshold=args.conf)

        print(f"  Found {len(detections)} photo(s)")

        if args.annotate:
            annotated = draw_detection_boxes(image, detections)
            ann_path = output_dir / f"{img_path.stem}_detected{img_path.suffix}"
            cv2.imwrite(str(ann_path), annotated)
            print(f"  Saved annotation: {ann_path}")

        for i, det in enumerate(detections):
            x1 = max(0, int(det["x1"]))
            y1 = max(0, int(det["y1"]))
            x2 = min(image.shape[1], int(det["x2"]))
            y2 = min(image.shape[0], int(det["y2"]))

            crop = image[y1:y2, x1:x2]

            stem = img_path.stem
            ext = img_path.suffix or ".jpg"
            out_path = output_dir / f"{stem}_photo{i+1}{ext}"
            cv2.imwrite(str(out_path), crop)
            print(f"  Saved crop: {out_path} ({crop.shape[1]}×{crop.shape[0]})")


def process_pose(args):
    """Process images using pose model, with optional detection pre-filter."""
    pose_session, pose_input = load_model(args.model)

    det_session = None
    det_input = None
    if args.use_detection:
        det_model_path = find_model("detect", args.detection_model)
        det_session, det_input = load_model(det_model_path)

    image_paths = resolve_inputs(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        print(f"\n{'=' * 50}")
        print(f"Processing: {img_path}")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Warning: Could not read {img_path}, skipping.")
            continue

        # If using detection pre-filter, first find regions
        if det_session is not None:
            detections = run_detection(image, det_session, det_input, conf_threshold=args.detection_conf)
            print(f"  Detection found {len(detections)} region(s)")

            if not detections:
                print(f"  No photos detected, skipping.")
                continue

            poses = []
            for det in detections:
                # Crop a region with margin for the pose model
                x1 = max(0, int(det["x1"]) - 20)
                y1 = max(0, int(det["y1"]) - 20)
                x2 = min(image.shape[1], int(det["x2"]) + 20)
                y2 = min(image.shape[0], int(det["y2"]) + 20)

                region = image[y1:y2, x1:x2]
                if region.size == 0:
                    continue

                region_poses = run_pose(region, pose_session, pose_input, conf_threshold=args.conf)

                # Remap keypoints back to full image coordinates
                for p in region_poses:
                    remapped_corners = []
                    for c in p["corners"]:
                        rx, ry, v = c
                        remapped_corners.append((rx + x1, ry + y1, v))
                    p["corners"] = remapped_corners
                    p["x1"] += x1
                    p["y1"] += y1
                    p["x2"] += x1
                    p["y2"] += y1
                poses.extend(region_poses)
        else:
            # Run pose directly on full image
            poses = run_pose(image, pose_session, pose_input, conf_threshold=args.conf)

        print(f"  Found {len(poses)} photo(s)")

        if args.annotate:
            annotated = draw_pose_corners(image, poses)
            ann_path = output_dir / f"{img_path.stem}_pose{img_path.suffix}"
            cv2.imwrite(str(ann_path), annotated)
            print(f"  Saved annotation: {ann_path}")

        for i, pose in enumerate(poses):
            corners = pose["corners"]

            # Check that all 4 keypoints have sufficient visibility
            visible = [c for c in corners if c[2] > 0.3]
            if len(visible) < 4:
                print(f"  Photo {i+1}: Only {len(visible)}/4 visible corners, skipping extraction")
                continue

            extracted, method = extract_photo(
                image, corners,
                threshold=args.threshold,
                margin=args.margin,
            )

            stem = img_path.stem
            ext = img_path.suffix or ".jpg"
            out_path = output_dir / f"{stem}_photo{i+1}_{method}{ext}"
            cv2.imwrite(str(out_path), extracted)

            distortion = max_quadrilateral_distortion(
                [(c[0], c[1]) for c in corners]
            )
            print(f"  Photo {i+1}: {method} (distortion={distortion:.1f}px, threshold={args.threshold}px) → {out_path} ({extracted.shape[1]}×{extracted.shape[0]})")


def resolve_inputs(inputs):
    """Resolve input paths, expanding directories and glob patterns."""
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"):
                paths.extend(sorted(p.glob(ext)))
        else:
            paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Photo Extract — Detect and extract photos from scanned images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Detection mode")

    # Shared arguments
    def add_common_args(sub):
        sub.add_argument("--input", "-i", nargs="+", required=True,
                          help="Input image path(s) or directory")
        sub.add_argument("--output", "-o", default="./output",
                          help="Output directory (default: ./output)")
        sub.add_argument("--model", "-m", default=None,
                          help="Path to ONNX model (auto-detected if omitted)")
        sub.add_argument("--conf", "-c", type=float, default=0.5,
                          help="Confidence threshold (default: 0.5)")
        sub.add_argument("--annotate", "-a", action="store_true",
                          help="Save annotated image with detections overlaid")

    # --- detect subcommand ---
    detect_parser = subparsers.add_parser(
        "detect", help="Detection mode — axis-aligned bounding box crops"
    )
    add_common_args(detect_parser)

    # --- pose subcommand ---
    pose_parser = subparsers.add_parser(
        "pose", help="Pose mode — corner keypoints with smart crop/warp"
    )
    add_common_args(pose_parser)
    pose_parser.add_argument(
        "--threshold", "-t", type=float, default=3.0,
        help="Distortion threshold in pixels: if all corners are within this "
             "distance of axis-aligned, crop instead of warp (default: 3.0)"
    )
    pose_parser.add_argument(
        "--margin", type=int, default=0,
        help="Extra pixels around axis-aligned crop (default: 0)"
    )
    pose_parser.add_argument(
        "--use-detection", action="store_true",
        help="Use detection model to pre-filter regions before pose model"
    )
    pose_parser.add_argument(
        "--detection-model", default=None,
        help="Path to detection ONNX model (only used with --use-detection)"
    )
    pose_parser.add_argument(
        "--detection-conf", type=float, default=0.5,
        help="Confidence threshold for detection pre-filter (default: 0.5)"
    )

    args = parser.parse_args()

    # Resolve model path
    args.model = find_model(args.mode, args.model)

    if args.mode == "detect":
        process_detect(args)
    elif args.mode == "pose":
        if args.use_detection and not args.detection_model:
            args.detection_model = find_model("detect")
        process_pose(args)


if __name__ == "__main__":
    main()