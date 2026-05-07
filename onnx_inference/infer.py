#!/usr/bin/env python3
"""
Photo Pose Detector — Two-Stage ONNX Inference CLI
===================================================

Two-stage pipeline for detecting photo corners in multi-photo images:

  Stage 1 (Detection):  Detect photo bounding boxes in the full image
  Stage 2 (Pose):       Crop each detected box → detect 4 corner keypoints
                        Map keypoints back to original image coordinates
  Stage 3 (Dedup):      Greedy deduplication by keypoint-center proximity

Single-photo mode is also supported (skip Stage 1, run Stage 2 directly).

Overlapping Detections
---------------------
The detection model can produce overlapping boxes for the same photo and
"combined" boxes spanning multiple photos. Standard NMS handles exact
duplicates well but doesn't fully suppress these. Instead of trying to
fix this with ever-more-complex NMS heuristics, we run pose on ALL
NMS-surviving boxes and then **deduplicate by keypoint-center proximity**:
sort by pose confidence, keep each result only if its center is far enough
from all previously kept results.

Model Output Formats
--------------------
  Detection model:  [1, 5, N_anchors]   — raw YOLO output (no NMS)
                    rows = cx, cy, w, h, class_confidence
  Pose model:       [1, 300, 18]         — NMS-enabled (end-to-end)
                    cols = x1,y1,x2,y2,conf,class, kp0_x,kp0_y,kp0_vis,
                           kp1_x,kp1_y,kp1_vis, kp2_x,kp2_y,kp2_vis,
                           kp3_x,kp3_y,kp3_vis

Keypoint Order (matches training convention):
  kp0 = Lower-Left  (LL) — high Y, low  X
  kp1 = Upper-Left  (UL) — low  Y, low  X
  kp2 = Upper-Right (UR) — low  Y, high X
  kp3 = Lower-Right (LR) — high Y, high X

Cropping
--------
    --crop simple          Bounding-box crop from detection model bbox
    --crop simple-corners  Bounding-box crop from detected corner keypoints
                           (tighter, more accurate than --crop simple)
    --crop warp            Perspective warp (inward): average of opposite
                           edge lengths → may lose small edge slivers
    --crop warp-stretch    Perspective warp (outward): max of opposite edge
                           lengths → preserves ALL photo content
    --crop-dir DIR         Output directory for crops (default: ./crops/)
    --crop-margin N        Expand crops outward by N pixels on each side.
                           For simple/simple-corners: expands the bounding box.
                           For warp/warp-stretch: pushes source corners
                           outward from the quad center before warping.
    --crop-transparent     Save crops as transparent PNG. For simple crops,
                           area outside keypoint quad is transparent. For
                           warp, out-of-bounds areas are transparent.
    --border-fill COLOR    Fill color for warp background (default: grey).
                           Accepts: R,G,B  #RRGGBB  #RGB  white/black/grey/
                           red/green/blue. Ignored with --crop-transparent.

    Output naming: {original_stem}_{photo_id}.{ext}
    Example: scan_001_1.jpg, scan_001_2.jpg

Usage
-----
    # Two-stage pipeline (multi-photo image)
    python3 infer.py --detection-model ../models/detection_model.onnx \
                     --pose-model ../models/pose_model_v2.onnx \
                     --image scan.jpg

    # Single-stage (single-photo image, skip detection)
    python3 infer.py --pose-model ../models/pose_model_v2.onnx --image crop.jpg

    # Batch directory
    python3 infer.py --detection-model ../models/detection_model.onnx \
                     --pose-model ../models/pose_model_v2.onnx \
                     --image ../data_pose_multi/images/val/ --batch --limit 5

    # With perspective warp cropping
    python3 infer.py --detection-model ../models/detection_model.onnx \
                     --pose-model ../models/pose_model_v2.onnx \
                     --image scan.jpg --crop warp

    # With warp stretching (outward, preserves all content)
    python3 infer.py --image scan.jpg --crop warp-stretch --border-fill white

    # Simple crop as transparent PNG
    python3 infer.py --image scan.jpg --crop simple --crop-transparent

    # Corner-based crop with 10px margin to avoid clipping edges
    python3 infer.py --image scan.jpg --crop simple-corners --crop-margin 10

    # Adjust thresholds
    python3 infer.py --image photo.jpg --det-conf 0.3 --pose-conf 0.25
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEYPOINT_NAMES = ["LL", "UL", "UR", "LR"]
KEYPOINT_COLORS = ["#FF4444", "#44DD44", "#4488FF", "#FFCC00"]
KEYPOINT_FULL_NAMES = {
    "LL": "Lower-Left",
    "UL": "Upper-Left",
    "UR": "Upper-Right",
    "LR": "Lower-Right",
}
BOX_COLOR = "#FF00FF"
EDGE_COLOR = "#00FFFF"
CROP_PAD = 4              # Pixels of padding around detection box for crop
DEFAULT_IMG_SIZE = 640
DEDUP_MIN_DIST_RATIO = 0.12  # 12% of image min-dimension
_VIS_THRESH_DEDUP = 0.25      # visibility threshold for center-based dedup


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_onnx_model(model_path: str):
    """Load an ONNX model with onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed.")
        print("Install with: pip install onnxruntime")
        sys.exit(1)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    for inp in session.get_inputs():
        print(f"  Input:  {inp.name}  shape={inp.shape}  dtype={inp.type}")
    for out in session.get_outputs():
        print(f"  Output: {out.name}  shape={out.shape}  dtype={out.type}")
    return session


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_letterbox(image: Image.Image, target_size: int = DEFAULT_IMG_SIZE):
    """
    Letterbox resize: preserves aspect ratio, pads with (114,114,114).

    Returns:
        input_array: (1, 3, H, W) float32 normalized [0,1]
        scale_info:  dict for mapping coords back to original image space
    """
    orig_w, orig_h = image.size
    ratio = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    resized = image.resize((new_w, new_h), Image.BILINEAR)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    canvas.paste(resized, (pad_w, pad_h))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

    scale_info = {
        "ratio": ratio,
        "pad_w": pad_w,
        "pad_h": pad_h,
        "orig_w": orig_w,
        "orig_h": orig_h,
    }
    return arr, scale_info


def preprocess_crop(crop: Image.Image, target_size: int = DEFAULT_IMG_SIZE):
    """
    Resize a crop to target_size×target_size (stretches, no letterbox).

    The pose model expects the photo to fill the frame (single-photo
    training distribution), so we stretch rather than letterbox.

    Returns:
        input_array: (1, 3, H, W) float32 normalized [0,1]
        crop_w, crop_h: original crop dimensions for coordinate mapping
    """
    crop_w, crop_h = crop.size
    resized = crop.resize((target_size, target_size), Image.BILINEAR)

    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

    return arr, crop_w, crop_h


# ---------------------------------------------------------------------------
# NMS (for raw detection model output)
# ---------------------------------------------------------------------------

def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45):
    """Non-maximum suppression on xyxy-format boxes."""
    if len(scores) == 0:
        return []

    x1, y1 = boxes_xyxy[:, 0], boxes_xyxy[:, 1]
    x2, y2 = boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_threshold]
    return keep


# ---------------------------------------------------------------------------
# Stage 1: Detection
# ---------------------------------------------------------------------------

def run_detection(session, image: Image.Image, conf_threshold: float = 0.5,
                  iou_threshold: float = 0.45, img_size: int = DEFAULT_IMG_SIZE):
    """
    Run the detection model on a full image.

    Returns list of dicts with keys: 'box' (x1,y1,x2,y2 in original coords),
    'confidence'.
    """
    orig_w, orig_h = image.size
    input_arr, scale_info = preprocess_letterbox(image, img_size)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_arr})[0]

    # Raw output: [1, 5, N_anchors] → cx, cy, w, h, class_conf
    cx = output[0, 0]
    cy = output[0, 1]
    w = output[0, 2]
    h = output[0, 3]
    conf = output[0, 4]

    # Filter by confidence
    mask = conf > conf_threshold
    if not mask.any():
        return []

    # Convert to xyxy in original image space
    ratio = scale_info["ratio"]
    pad_w = scale_info["pad_w"]
    pad_h = scale_info["pad_h"]

    x1 = (cx[mask] - w[mask] / 2 - pad_w) / ratio
    y1 = (cy[mask] - h[mask] / 2 - pad_h) / ratio
    x2 = (cx[mask] + w[mask] / 2 - pad_w) / ratio
    y2 = (cy[mask] + h[mask] / 2 - pad_h) / ratio
    filtered_conf = conf[mask]

    # NMS
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes_xyxy, filtered_conf, iou_threshold)

    results = []
    for idx in keep:
        bx1 = max(0, boxes_xyxy[idx, 0])
        by1 = max(0, boxes_xyxy[idx, 1])
        bx2 = min(orig_w, boxes_xyxy[idx, 2])
        by2 = min(orig_h, boxes_xyxy[idx, 3])

        results.append({
            "box": {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2},
            "confidence": float(filtered_conf[idx]),
        })

    return results


# ---------------------------------------------------------------------------
# Stage 2: Pose
# ---------------------------------------------------------------------------

def run_pose(session, crop: Image.Image, conf_threshold: float = 0.5,
             img_size: int = DEFAULT_IMG_SIZE):
    """
    Run the pose model on a cropped photo region.

    Returns list of dicts with keys: 'confidence', 'keypoints' (list of
    dicts with 'name', 'x', 'y', 'visibility' in crop pixel coords),
    or empty list if no detection.
    """
    crop_w, crop_h = crop.size
    input_arr, _, _ = preprocess_crop(crop, img_size)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_arr})[0]

    # Output: [1, 300, 18] — NMS-enabled
    rows = output[0]

    results = []
    for row in rows:
        conf = row[4]
        if conf < conf_threshold:
            continue

        # Map keypoints from 640×640 → crop pixel coords
        scale_x = crop_w / img_size
        scale_y = crop_h / img_size

        keypoints = []
        for k in range(4):
            kx = row[6 + k * 3] * scale_x
            ky = row[6 + k * 3 + 1] * scale_y
            kvis = row[6 + k * 3 + 2]
            keypoints.append({
                "name": KEYPOINT_NAMES[k],
                "x": kx,
                "y": ky,
                "visibility": float(kvis),
            })

        results.append({
            "confidence": float(conf),
            "keypoints": keypoints,
        })

    return results


# ---------------------------------------------------------------------------
# Stage 3: Pose-based deduplication
# ---------------------------------------------------------------------------

def dedup_pose_results(pose_results: list, min_center_dist: float):
    """
    Greedy deduplication: sort by dedup priority descending, keep each
    result only if its keypoint center is at least min_center_dist pixels
    away from all previously kept results.

    This handles overlapping detection boxes and "combined" boxes that
    span multiple photos — the pose model correctly localizes whichever
    photo is dominant in each crop, and dedup ensures we report each
    unique photo only once.

    Results are sorted by ``dedup_priority`` (pose confidence with a
    visibility penalty) rather than raw pose confidence. Results with
    fewer than 3 visible corners get demoted because 2-corner centers
    from only one side of the photo are unreliable for dedup.

    Args:
        pose_results: list of dicts with 'center' and 'dedup_priority'
        min_center_dist: minimum distance between kept photo centers (px)
    """
    pose_results.sort(key=lambda r: r.get("dedup_priority", r["pose_confidence"]), reverse=True)

    kept = []
    for r in pose_results:
        too_close = any(
            np.sqrt((r["center"][0] - k["center"][0]) ** 2 +
                    (r["center"][1] - k["center"][1]) ** 2) < min_center_dist
            for k in kept
        )
        if not too_close:
            kept.append(r)

    return kept


# ---------------------------------------------------------------------------
# Two-stage pipeline
# ---------------------------------------------------------------------------

def pipeline(detection_session, pose_session, image: Image.Image,
            det_conf: float = 0.5, pose_conf: float = 0.5,
            iou_threshold: float = 0.45, crop_pad: int = CROP_PAD,
            img_size: int = DEFAULT_IMG_SIZE,
            dedup_min_dist_ratio: float = DEDUP_MIN_DIST_RATIO):
    """
    Full two-stage pipeline: detect photos → find corners → dedup.

    Returns list of dicts:
        {
            'detection': {'box': {...}, 'confidence': float},
            'pose_confidence': float,
            'keypoints': [                     # In ORIGINAL image coordinates
                {'name': 'LL', 'x': float, 'y': float, 'visibility': float},
                ...
            ],
            'center': (float, float),          # Mean of visible keypoints
        }
    """
    orig_w, orig_h = image.size

    # Stage 1: Detect photos
    if detection_session is not None:
        detections = run_detection(
            detection_session, image, det_conf, iou_threshold, img_size
        )
    else:
        # Single-photo mode: treat entire image as one detection
        detections = [{
            "box": {"x1": 0, "y1": 0, "x2": orig_w, "y2": orig_h},
            "confidence": 1.0,
        }]

    # Stage 2: Find corners for each detected photo
    pose_results = []
    for det in detections:
        box = det["box"]
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        # Crop with padding
        crop_x1 = max(0, int(x1) - crop_pad)
        crop_y1 = max(0, int(y1) - crop_pad)
        crop_x2 = min(orig_w, int(x2) + crop_pad)
        crop_y2 = min(orig_h, int(y2) + crop_pad)

        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Run pose on crop
        pose_dets = run_pose(pose_session, crop, pose_conf, img_size)

        if not pose_dets:
            continue

        # Take highest-confidence pose result
        best_pose = max(pose_dets, key=lambda p: p["confidence"])

        # Map keypoints from crop coords → original image coords
        visible_xs, visible_ys = [], []
        mapped_keypoints = []
        for kp in best_pose["keypoints"]:
            mk = {
                "name": kp["name"],
                "x": kp["x"] + crop_x1,
                "y": kp["y"] + crop_y1,
                "visibility": kp["visibility"],
            }
            mapped_keypoints.append(mk)
            if kp["visibility"] >= _VIS_THRESH_DEDUP:
                visible_xs.append(mk["x"])
                visible_ys.append(mk["y"])

        if len(visible_xs) < 2:
            continue  # Skip if too few visible corners

        center = (float(np.mean(visible_xs)), float(np.mean(visible_ys)))

        # Visibility penalty for dedup: results with only 2 visible corners
        # get demoted so they're less likely to suppress a better detection.
        # (2 visible corners from one side of the photo give a misleading center.)
        vis_count = len(visible_xs)
        dedup_priority = best_pose["confidence"]
        if vis_count < 3:
            dedup_priority *= 0.5  # Penalize low-visibility results

        pose_results.append({
            "detection": det,
            "pose_confidence": best_pose["confidence"],
            "keypoints": mapped_keypoints,
            "center": center,
            "dedup_priority": dedup_priority,
        })

    # Stage 3: Deduplicate by keypoint-center proximity
    if detection_session is not None and len(pose_results) > 1:
        min_dist = min(orig_w, orig_h) * dedup_min_dist_ratio
        pose_results = dedup_pose_results(pose_results, min_dist)

    return pose_results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_results(image: Image.Image, results: list):
    """Draw detection boxes, keypoints, and corner quadrilaterals."""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, res in enumerate(results):
        box = res["detection"]["box"]
        det_conf = res["detection"]["confidence"]
        pose_conf = res.get("pose_confidence", 0)
        kps = res.get("keypoints", [])

        # Bounding box
        draw.rectangle(
            [box["x1"], box["y1"], box["x2"], box["y2"]],
            outline=BOX_COLOR, width=2,
        )
        label = f"photo {det_conf:.2f}"
        if pose_conf > 0:
            label += f" | pose {pose_conf:.2f}"
        draw.text((box["x1"] + 4, box["y1"] + 2), label, fill=BOX_COLOR, font=font)

        # Index label
        draw.text((box["x2"] - 20, box["y1"] + 2), f"#{i+1}", fill=BOX_COLOR, font=font)

        # Keypoint circles and labels
        valid_points = []
        for kp in kps:
            kx, ky = kp["x"], kp["y"]
            name = kp["name"]
            vis = kp["visibility"]
            color = KEYPOINT_COLORS[KEYPOINT_NAMES.index(name)]
            r = 6
            draw.ellipse(
                [kx - r, ky - r, kx + r, ky + r],
                fill=color, outline="black", width=1,
            )
            draw.text(
                (kx + 10, ky - 8),
                f"{name} ({kx:.0f},{ky:.0f}) v={vis:.2f}",
                fill=color, font=font,
            )
            valid_points.append((kx, ky))

        # Draw corner quadrilateral: LL→UL→UR→LR→LL
        if len(valid_points) == 4:
            for j in range(4):
                p1 = valid_points[j]
                p2 = valid_points[(j + 1) % 4]
                draw.line([p1, p2], fill=EDGE_COLOR, width=2)

    return image


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def _get_keypoints_bbox(result: dict, margin: float = 0):
    """
    Compute the tight axis-aligned bounding box around the 4 detected
    corner keypoints. Falls back to the detection model's bounding box
    if keypoints are unavailable or insufficiently visible.

    This is typically tighter and more accurate than the detection
    box because the pose model directly localizes the photo corners.

    Args:
        result: Pipeline result dict with 'keypoints' list and
            'detection'/'box' dict.
        margin: Extra pixels to expand the box outward in each
            direction. Positive values add padding around the crop.

    Returns:
        (x1, y1, x2, y2) in original image pixel coordinates.
    """
    kps = result.get("keypoints", [])
    kp_by_name = {kp["name"]: kp for kp in kps} if kps else {}

    # Need at least 2 visible corners for a meaningful keypoint bbox
    visible = [kp for kp in kps if kp.get("visibility", 0) >= 0.1]
    if len(visible) >= 2:
        xs = [kp["x"] for kp in visible]
        ys = [kp["y"] for kp in visible]
        x1 = min(xs) - margin
        y1 = min(ys) - margin
        x2 = max(xs) + margin
        y2 = max(ys) + margin
        return x1, y1, x2, y2
    else:
        # Fallback to detection bounding box
        box = result["detection"]["box"]
        x1 = box["x1"] - margin
        y1 = box["y1"] - margin
        x2 = box["x2"] + margin
        y2 = box["y2"] + margin
        return x1, y1, x2, y2


def crop_simple(image: Image.Image, result: dict,
                transparent: bool = False,
                use_corners: bool = False,
                margin: float = 0) -> Image.Image:
    """
    Simple crop: extract the detected photo region as an axis-aligned
    rectangle, no geometric transforms.

    By default uses the detection model's bounding box. When
    use_corners=True, uses the tight bounding box around the detected
    corner keypoints instead, which is typically more accurate at
    finding the actual photo boundary.

    When transparent=True, the crop is returned as RGBA with the
    region outside the detected corner quadrilateral set to fully
    transparent. This allows the crop to be saved as a transparent
    PNG where only the actual photo area is opaque.

    Args:
        image: Original full-size PIL image
        result: Pipeline result dict with 'detection'/'box' and
            'keypoints' list.
        transparent: If True, make area outside the keypoint
            quadrilateral transparent (RGBA output).
        use_corners: If True, compute the crop box from the detected
            corner keypoints instead of the detection bounding box.
            More accurate at finding photo boundaries.
        margin: Extra pixels to expand the crop outward in each
            direction (positive = more of the surrounding area
            included in the crop, negative = tighter crop).

    Returns:
        Cropped PIL image (RGB or RGBA if transparent)
    """
    if use_corners:
        x1, y1, x2, y2 = _get_keypoints_bbox(result, margin=margin)
    else:
        box = result["detection"]["box"]
        x1 = box["x1"] - margin
        y1 = box["y1"] - margin
        x2 = box["x2"] + margin
        y2 = box["y2"] + margin

    # Clamp to image bounds
    ix1 = max(0, int(x1))
    iy1 = max(0, int(y1))
    ix2 = min(image.size[0], int(round(x2)))
    iy2 = min(image.size[1], int(round(y2)))

    # Check for valid crop dimensions
    if ix2 <= ix1 or iy2 <= iy1:
        ix2 = max(ix2, ix1 + 1)
        iy2 = max(iy2, iy1 + 1)

    cropped = image.crop((ix1, iy1, ix2, iy2))

    if not transparent:
        return cropped

    # Build a mask that is opaque inside the keypoint quadrilateral
    # and transparent everywhere else.
    kps = result.get("keypoints", [])
    if len(kps) < 4 or any(kp["visibility"] < 0.1 for kp in kps):
        # Not enough reliable corners for a mask — return opaque crop
        return cropped.convert("RGBA")

    # Keypoint coords relative to the crop origin
    offset_x, offset_y = ix1, iy1
    # Order for polygon: UL → UR → LR → LL (clockwise from top-left)
    kp_by_name = {kp["name"]: kp for kp in kps}
    poly_points = [
        (kp_by_name["UL"]["x"] - offset_x, kp_by_name["UL"]["y"] - offset_y),
        (kp_by_name["UR"]["x"] - offset_x, kp_by_name["UR"]["y"] - offset_y),
        (kp_by_name["LR"]["x"] - offset_x, kp_by_name["LR"]["y"] - offset_y),
        (kp_by_name["LL"]["x"] - offset_x, kp_by_name["LL"]["y"] - offset_y),
    ]

    # Draw mask using PIL: white inside polygon, black outside
    mask = Image.new("L", cropped.size, 0)
    from PIL import ImageDraw as _ID
    mask_draw = _ID.Draw(mask)
    mask_draw.polygon(poly_points, fill=255)

    # Composite: photo through mask onto transparent background
    rgba = cropped.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def crop_perspective_warp(image: Image.Image, result: dict,
                          min_visibility: float = 0.3,
                          warp_mode: str = "inward",
                          border_fill: tuple = (114, 114, 114),
                          transparent: bool = False,
                          margin: float = 0) -> Image.Image:
    """
    Perspective warp crop: use the 4 detected corner keypoints to
    compute a homography that dewarps the photo into a rectangular
    output image.

    Uses cv2.warpPerspective with INTER_LANCZOS4 (8x8 Lanczos) for
    the highest-quality interpolation available in OpenCV.

    Warp modes:
      inward  — Output rectangle dimensions from the average of opposite
                edge lengths. This "inscribes" the rect in the detected
                quadrilateral, which may lose small slivers of content
                at the edges but produces a clean crop. (Default)
      outward — Output rectangle dimensions from the maximum of opposite
                edge lengths. This "circumscribes" the detected quad,
                guaranteeing ALL photo content is preserved at the cost
                of small filler areas in the corners of the output.

    Margin:
      When margin > 0, the 4 source corner keypoints are expanded outward
      along the edges of the quadrilateral before computing the homography.
      This adds real pixels of surrounding area to the dewarped output,
      ensuring the photo edges are fully included even if the keypoint
      detection slightly underestimates the true corner positions.

    Args:
        image: Original full-size PIL image
        result: Pipeline result dict with 'keypoints' list
        min_visibility: Minimum keypoint visibility to use that
            corner for the warp. Falls back to bbox-derived corner
            if a keypoint is below this threshold.
        warp_mode: 'inward' (average edge lengths) or 'outward'
            (max edge lengths) for output dimension calculation.
        border_fill: RGB tuple for out-of-bounds fill color used by
            cv2.warpPerspective. Ignored when transparent=True.
        transparent: If True, produce an RGBA image with transparent
            background instead of border_fill. The warp is done with
            a transparent (0,0,0,0) border, then converted back to PIL
            RGBA.
        margin: Extra pixels to expand the source corner points outward
            along the quadrilateral edges before warping. Adds real
            surrounding content to the output.

    Returns:
        Dewarped (rectangular) PIL image, or None if insufficient
        visible corners for a perspective transform.
    """
    kps = result.get("keypoints", [])
    if len(kps) < 4:
        return None

    # Build source points in the order: UL, UR, LR, LL
    # (clockwise from top-left, which is what cv2.getPerspectiveTransform expects)
    kp_by_name = {kp["name"]: kp for kp in kps}

    corner_names = ["UL", "UR", "LR", "LL"]
    src_points = []

    for name in corner_names:
        kp = kp_by_name.get(name)
        if kp is not None and kp["visibility"] >= min_visibility:
            src_points.append([kp["x"], kp["y"]])
        else:
            # Fallback: derive corner position from bounding box
            # This is approximate but better than skipping the warp entirely
            box = result["detection"]["box"]
            bbox_corners = {
                "UL": [box["x1"], box["y1"]],
                "UR": [box["x2"], box["y1"]],
                "LR": [box["x2"], box["y2"]],
                "LL": [box["x1"], box["y2"]],
            }
            src_points.append(bbox_corners[name])

    src_pts = np.array(src_points, dtype=np.float32)

    # Expand source corners outward by margin pixels along edges
    # Each corner is pushed outward along the two edges it connects to.
    # For corner i, the outward direction is the average of the two
    # edge normals pointing away from the quadrilateral center.
    if margin > 0:
        center = src_pts.mean(axis=0)
        # Edges: UL→UR, UR→LR, LR→LL, LL→UL
        # Edge vectors (unnormalized direction from each corner toward the next)
        edges_next = np.roll(src_pts, -1, axis=0) - src_pts
        edges_prev = np.roll(src_pts, 1, axis=0) - src_pts
        # Outward normal for each edge: rotate edge 90° away from center.
        # For edge (x,y), outward normal is (y, -x) rotated if center is inside.
        # Simpler approach: push each corner away from center.
        for i in range(4):
            direction = src_pts[i] - center
            dist = np.linalg.norm(direction)
            if dist > 0:
                src_pts[i] += (margin / dist) * direction

    # Compute output dimensions from detected edge lengths
    w_top = np.linalg.norm(src_pts[1] - src_pts[0])
    w_bot = np.linalg.norm(src_pts[3] - src_pts[2])
    h_left = np.linalg.norm(src_pts[3] - src_pts[0])
    h_right = np.linalg.norm(src_pts[2] - src_pts[1])

    if warp_mode == "outward":
        # Use max of opposite edges → circumscribes the detected quad
        # Guarantees no photo content is lost; may include small filler
        # areas in the corners of the output rectangle.
        out_w = int(round(max(w_top, w_bot)))
        out_h = int(round(max(h_left, h_right)))
    else:
        # Inward (default): average of opposite edges → inscribes the rect
        out_w = int(round((w_top + w_bot) / 2))
        out_h = int(round((h_left + h_right) / 2))

    # Minimum output size guard
    out_w = max(out_w, 32)
    out_h = max(out_h, 32)

    # Destination rectangle: same clockwise order (UL, UR, LR, LL)
    dst_pts = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    # Compute perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Convert PIL image to cv2 format (RGB → BGR)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if transparent:
        # Add alpha channel: BGRA with source alpha = 255 everywhere
        img_bgra = cv2.cvtColor(img_cv, cv2.COLOR_BGR2BGRA)
        # Apply warp with transparent border
        warped = cv2.warpPerspective(
            img_bgra,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),  # Transparent
        )
        # Convert BGRA → RGBA for PIL
        warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(warped_rgba)
    else:
        # Apply warp with specified border fill color
        border_bgr = (border_fill[2], border_fill[1], border_fill[0])  # RGB → BGR
        warped = cv2.warpPerspective(
            img_cv,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_bgr,
        )
        # Convert back to PIL (BGR → RGB)
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(warped_rgb)


def parse_border_fill(value: str) -> tuple:
    """Parse a border fill color string into an RGB tuple.

    Accepts: 'R,G,B' (0-255), hex '#RRGGBB', hex '#RGB', or
    named colors 'white', 'black', 'grey', 'gray', 'red', etc.
    """
    value = value.strip().lower()

    # Named colors
    named = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "grey": (114, 114, 114),
        "gray": (114, 114, 114),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "none": (114, 114, 114),  # default grey
    }
    if value in named:
        return named[value]

    # Hex: #RRGGBB or #RGB
    if value.startswith("#"):
        hex_str = value[1:]
        if len(hex_str) == 3:
            return (int(hex_str[0]*2, 16), int(hex_str[1]*2, 16), int(hex_str[2]*2, 16))
        if len(hex_str) == 6:
            return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
        raise ValueError(f"Invalid hex color: {value!r}")

    # R,G,B decimal
    parts = value.split(",")
    if len(parts) == 3:
        try:
            return (int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip()))
        except ValueError:
            pass

    raise ValueError(
        f"Invalid border fill color: {value!r}. "
        "Use 'R,G,B', '#RRGGBB', '#RGB', or named color (white/black/grey/red/green/blue)."
    )


def save_crops(image: Image.Image, results: list, image_path: str,
               crop_mode: str = "simple", crop_dir: str = None,
               transparent: bool = False,
               warp_mode: str = "inward",
               border_fill: tuple = (114, 114, 114),
               source_exif: bytes = None,
               margin: float = 0):
    """
    Save cropped photos for each detected photo in results.

    Args:
        image: Original full-size PIL image
        results: List of pipeline result dicts
        image_path: Original image file path (for naming)
        crop_mode: 'simple' for bbox crop, 'simple-corners' for
                   keypoint-based bbox crop, 'warp' or 'warp-stretch'
                   for perspective warp
        crop_dir: Output directory (default: same as input image)
        transparent: If True, save as transparent PNG (area outside
                     photo quadrilateral is transparent). Forces .png
                     output format.
        warp_mode: 'inward' (avg edge dims) or 'outward' (max edge
                   dims) for perspective warp output size.
        border_fill: RGB tuple for warp background fill, or None
                     for default grey (114,114,114).
        source_exif: Raw EXIF bytes from the source image to preserve
                     in cropped outputs. If None, no EXIF is written.
        margin: Extra pixels to expand crops outward on each side.
                For simple crops, expands the bounding box. For warp
                crops, pushes corner keypoints outward from the
                quadrilateral center before computing the homography.
    """
    if not results:
        return

    stem = Path(image_path).stem

    # When transparent, always output as PNG regardless of input format
    if transparent:
        ext = ".png"
    else:
        ext = Path(image_path).suffix or ".jpg"

    if crop_dir:
        out_dir = Path(crop_dir)
    else:
        out_dir = Path(image_path).parent / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract EXIF from source image for preservation in crops
    # (caller passes it in since image.convert() may have stripped it)

    for i, res in enumerate(results):
        photo_id = i + 1
        out_name = f"{stem}_{photo_id}{ext}"
        out_path = out_dir / out_name

        if crop_mode.startswith("warp"):
            wm = "outward" if crop_mode == "warp-stretch" else "inward"
            cropped = crop_perspective_warp(
                image, res, warp_mode=wm,
                border_fill=border_fill, transparent=transparent,
                margin=margin,
            )
            if cropped is None:
                # Fallback to simple crop if warp fails (insufficient keypoints)
                print(f"    Photo #{photo_id}: warp failed (insufficient keypoints), falling back to simple crop")
                cropped = crop_simple(image, res, transparent=transparent, margin=margin)
        else:
            use_corners = (crop_mode == "simple-corners")
            cropped = crop_simple(image, res, transparent=transparent,
                                  use_corners=use_corners, margin=margin)

        # Save with high quality, preserving EXIF from source
        save_kwargs = {}
        if source_exif:
            # PIL supports EXIF for both JPEG and PNG, including RGBA PNG.
            # For JPEG, only include EXIF for RGB mode (JPEG doesn't support RGBA).
            if out_path.suffix.lower() in (".jpg", ".jpeg"):
                save_kwargs["quality"] = 95
                if cropped.mode == "RGB":
                    save_kwargs["exif"] = source_exif
            elif out_path.suffix.lower() == ".png":
                save_kwargs["exif"] = source_exif
        else:
            if out_path.suffix.lower() in (".jpg", ".jpeg"):
                save_kwargs["quality"] = 95

        cropped.save(str(out_path), **save_kwargs)

        box = res["detection"]["box"]
        kps = res.get("keypoints", [])
        vis_corners = sum(1 for kp in kps if kp["visibility"] >= 0.3)
        dim_info = f"{cropped.size[0]}×{cropped.size[1]}"
        mode_info = cropped.mode
        print(f"    Photo #{photo_id} → {out_path}  "
              f"({dim_info} {mode_info}  "
              f"box=({box['x1']:.0f},{box['y1']:.0f})→({box['x2']:.0f},{box['y2']:.0f})  "
              f"visible_corners={vis_corners}/4)")


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def infer_single(detection_session, pose_session, image_path: str,
                 output_path: str = None, det_conf: float = 0.5,
                 pose_conf: float = 0.5, iou_threshold: float = 0.45,
                 crop_pad: int = CROP_PAD, img_size: int = DEFAULT_IMG_SIZE,
                 dedup_min_dist_ratio: float = DEDUP_MIN_DIST_RATIO,
                 crop_mode: str = None, crop_dir: str = None,
                 transparent: bool = False,
                 border_fill: tuple = (114, 114, 114),
                 margin: float = 0):
    """Run the full pipeline on a single image."""

    # Open image and capture EXIF before converting (convert creates
    # a new image that loses the .info dict including exif)
    _raw_image = Image.open(image_path)
    _source_exif = _raw_image.info.get("exif")
    image = _raw_image.convert("RGB")
    orig_w, orig_h = image.size

    results = pipeline(
        detection_session, pose_session, image,
        det_conf, pose_conf, iou_threshold, crop_pad, img_size,
        dedup_min_dist_ratio,
    )

    # Print results
    mode = "two-stage" if detection_session else "single-stage (pose only)"
    print(f"\n{'=' * 60}")
    print(f"Image: {image_path}")
    print(f"Size: {orig_w}×{orig_h}")
    print(f"Mode: {mode}")
    print(f"Photos found: {len(results)}")
    print(f"{'=' * 60}")

    for i, res in enumerate(results):
        box = res["detection"]["box"]
        det_c = res["detection"]["confidence"]
        pose_c = res.get("pose_confidence", 0)
        kps = res.get("keypoints", [])

        print(f"\n  Photo #{i+1}:  detection_conf={det_c:.4f}  pose_conf={pose_c:.4f}")
        print(f"    Box: ({box['x1']:.1f}, {box['y1']:.1f}) → ({box['x2']:.1f}, {box['y2']:.1f})")
        if kps:
            for kp in kps:
                full = KEYPOINT_FULL_NAMES[kp["name"]]
                print(f"    {kp['name']} ({full}): "
                      f"({kp['x']:.1f}, {kp['y']:.1f})  vis={kp['visibility']:.3f}")
        else:
            print("    No pose detection in this crop")

    # Draw and save
    vis = image.copy()
    draw_results(vis, results)

    if output_path is None:
        base = Path(image_path).stem
        parent = Path(image_path).parent
        output_path = str(parent / f"{base}_detected.jpg")

    vis.save(output_path, quality=95)
    print(f"\n  Saved: {output_path}")

    # Save crops if requested
    if crop_mode:
        print(f"\n  Cropping ({crop_mode}):")
        wm = "outward" if crop_mode == "warp-stretch" else ("inward" if crop_mode.startswith("warp") else None)
        save_crops(image, results, image_path, crop_mode, crop_dir,
                   transparent=transparent, warp_mode=wm,
                   border_fill=border_fill,
                   source_exif=_source_exif,
                   margin=margin)

    return results


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def infer_batch(detection_session, pose_session, image_dir: str,
               output_dir: str = None, det_conf: float = 0.5,
               pose_conf: float = 0.5, iou_threshold: float = 0.45,
               crop_pad: int = CROP_PAD, img_size: int = DEFAULT_IMG_SIZE,
               dedup_min_dist_ratio: float = DEDUP_MIN_DIST_RATIO,
               limit: int = 0,
               crop_mode: str = None, crop_dir: str = None,
               transparent: bool = False,
               border_fill: tuple = (114, 114, 114),
               margin: float = 0):
    """Run the full pipeline on all images in a directory."""

    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions and not p.name.endswith("_detected.jpg")
    )

    if limit > 0:
        images = images[:limit]

    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = image_dir / "detected"
    out_path.mkdir(parents=True, exist_ok=True)

    mode = "two-stage" if detection_session else "pose-only"
    print(f"Processing {len(images)} images ({mode})")
    print(f"Output: {out_path}")

    summary = []
    for img_path in images:
        out_file = out_path / f"{img_path.stem}_detected.jpg"
        results = infer_single(
            detection_session, pose_session, str(img_path), str(out_file),
            det_conf, pose_conf, iou_threshold, crop_pad, img_size,
            dedup_min_dist_ratio,
            crop_mode, crop_dir,
            transparent, border_fill,
            margin,
        )
        photo_count = len(results)
        pose_ok = sum(1 for r in results if r.get("keypoints"))
        summary.append({
            "image": img_path.name,
            "photos": photo_count,
            "with_pose": pose_ok,
        })

    # Summary
    print(f"\n{'=' * 60}")
    print("BATCH SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Image':<30} {'Photos':>8} {'W/Pose':>8}")
    print(f"{'-' * 30} {'-' * 8} {'-' * 8}")
    for e in summary:
        print(f"{e['image']:<30} {e['photos']:>8} {e['with_pose']:>8}")
    total_photos = sum(e["photos"] for e in summary)
    total_pose = sum(e["with_pose"] for e in summary)
    print(f"{'-' * 30} {'-' * 8} {'-' * 8}")
    print(f"{'Total':<30} {total_photos:>8} {total_pose:>8}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Photo Pose Detector — Two-Stage ONNX Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two-stage pipeline (multi-photo image)
  python3 infer.py \\
    --detection-model ../models/detection_model.onnx \\
    --pose-model ../models/pose_model_v2.onnx \\
    --image scan.jpg

  # Single-stage (skip detection, image is already a single photo)
  python3 infer.py --pose-model ../models/pose_model_v2.onnx --image crop.jpg

  # Batch validation images
  python3 infer.py \\
    --detection-model ../models/detection_model.onnx \\
    --pose-model ../models/pose_model_v2.onnx \\
    --image ../data_pose_multi/images/val/ --batch --limit 5

  # With perspective warp cropping
  python3 infer.py \\
    --detection-model ../models/detection_model.onnx \\
    --pose-model ../models/pose_model_v2.onnx \\
    --image scan.jpg --crop warp

  # Outward warp (preserves all photo content)
  python3 infer.py \\
    --image scan.jpg --crop warp-stretch --border-fill white

  # Simple crop as transparent PNG
  python3 infer.py --image scan.jpg --crop simple --crop-transparent

  # Corner-based crop with 10px margin
  python3 infer.py --image scan.jpg --crop simple-corners --crop-margin 10
""",
    )

    parser.add_argument(
        "--detection-model", "-d", type=str, default=None,
        help="Path to detection ONNX model (omit for single-photo/pose-only mode)",
    )
    parser.add_argument(
        "--pose-model", "-p", type=str,
        default="../models/pose_model_v2.onnx",
        help="Path to pose ONNX model (default: ../models/pose_model_v2.onnx)",
    )
    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="Path to image file or directory (with --batch)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for annotated image or directory (batch)",
    )
    parser.add_argument(
        "--det-conf", type=float, default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--pose-conf", type=float, default=0.5,
        help="Pose confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold for detection stage (default: 0.45)",
    )
    parser.add_argument(
        "--crop-pad", type=int, default=CROP_PAD,
        help="Padding around detection box for pose crop (default: 4)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMG_SIZE,
        help="Model input image size (default: 640)",
    )
    parser.add_argument(
        "--dedup-dist", type=float, default=DEDUP_MIN_DIST_RATIO,
        help="Min center distance for dedup, as fraction of image min-dimension (default: 0.08)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all images in directory specified by --image",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=0,
        help="Limit number of images in batch mode (0 = all)",
    )
    parser.add_argument(
        "--crop", type=str, default=None,
        choices=["simple", "simple-corners", "warp", "warp-stretch"],
        help="Crop detected photos: "
             "'simple' (detection bbox crop, no transforms), "
             "'simple-corners' (tighter bbox from detected corner keypoints, "
             "more accurate), "
             "'warp' (perspective warp, inward — average edge dims), or "
             "'warp-stretch' (perspective warp, outward — max edge dims, "
             "preserves all photo content). "
             "Output saved as {stem}_{photo_id}.{ext}",
    )
    parser.add_argument(
        "--crop-dir", type=str, default=None,
        help="Output directory for cropped photos (default: 'crops' subdirectory "
             "next to input image)",
    )
    parser.add_argument(
        "--crop-margin", type=float, default=0,
        help="Extra pixels to expand the crop outward on each side. For simple "
             "crops, expands the bounding box. For warp crops, pushes corner "
             "keypoints outward from the quadrilateral center. Useful to ensure "
             "photo edges are fully included. (default: 0)",
    )
    parser.add_argument(
        "--crop-transparent", action="store_true",
        help="Save crops as transparent PNG. For simple crops, the area outside "
             "the keypoint quadrilateral is transparent. For warp crops, areas "
             "outside the source image are transparent instead of filled with "
             "the border color. Forces .png output format.",
    )
    parser.add_argument(
        "--border-fill", type=str, default="grey",
        help="Border fill color for perspective warp (areas outside source "
             "image). Accepts 'R,G,B' (0-255), '#RRGGBB', '#RGB', or named "
             "colors: white, black, grey/gray, red, green, blue. "
             "Ignored when --crop-transparent is set. (default: grey)",
    )

    args = parser.parse_args()

    # Validate pose model
    if not Path(args.pose_model).exists():
        print(f"Error: Pose model not found: {args.pose_model}")
        sys.exit(1)

    # Load pose model
    print(f"Loading pose model: {args.pose_model}")
    pose_session = load_onnx_model(args.pose_model)

    # Load detection model (optional)
    detection_session = None
    if args.detection_model:
        if not Path(args.detection_model).exists():
            print(f"Error: Detection model not found: {args.detection_model}")
            sys.exit(1)
        print(f"Loading detection model: {args.detection_model}")
        detection_session = load_onnx_model(args.detection_model)

    # Validate image path
    image_path = Path(args.image)
    if args.batch:
        if not image_path.is_dir():
            print("Error: --batch requires --image to be a directory")
            sys.exit(1)
    else:
        if not image_path.exists():
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)

    # Parse border fill color
    border_fill = parse_border_fill(args.border_fill)

    # Run inference
    if args.batch:
        infer_batch(
            detection_session, pose_session, args.image, args.output,
            det_conf=args.det_conf,
            pose_conf=args.pose_conf,
            iou_threshold=args.iou,
            crop_pad=args.crop_pad,
            img_size=args.imgsz,
            dedup_min_dist_ratio=args.dedup_dist,
            limit=args.limit,
            crop_mode=args.crop,
            crop_dir=args.crop_dir,
            transparent=args.crop_transparent,
            border_fill=border_fill,
            margin=args.crop_margin,
        )
    else:
        infer_single(
            detection_session, pose_session, args.image, args.output,
            det_conf=args.det_conf,
            pose_conf=args.pose_conf,
            iou_threshold=args.iou,
            crop_pad=args.crop_pad,
            img_size=args.imgsz,
            dedup_min_dist_ratio=args.dedup_dist,
            crop_mode=args.crop,
            crop_dir=args.crop_dir,
            transparent=args.crop_transparent,
            border_fill=border_fill,
            margin=args.crop_margin,
        )


if __name__ == "__main__":
    main()