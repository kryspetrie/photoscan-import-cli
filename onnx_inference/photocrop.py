#!/usr/bin/env python3
"""
photocrop -- Detect & Extract Photos from Multi-Photo Scans
===========================================================

Pipeline for detecting photo corners in multi-photo images:

  Stage 1 (Detection):  Detect photo bounding boxes in the full image
  Stage 2 (Pose):        Crop each detected box -> detect 4 corner keypoints
                         Map keypoints back to original image coordinates
  Stage 2b (Refine):     [optional, --pose-refine] Re-run pose on tighter crop
  Stage 3 (Dedup):       Greedy deduplication by keypoint-center proximity
  Stage 4 (Rescue):      [automatic] Recover invisible/low-vis corners using
                         Sobel edge detection + line intersection. Only runs
                         on photos with < 3 visible corners -- zero cost on
                         well-detected images.

Optional post-rescue stages:
  --corner-refine:       Crop around each corner -> run corner regression
                          model for vis=1.0. Specialized 320x320 model detects
                          tight bounding boxes around corner points, picks the
                          detection closest to the expected position.
  --cv-refine:           Sobel edge refinement on ALL corners (sub-pixel)

The rescue stage (always on) prevents degenerate crops (e.g., "strip" crops
from only 2 visible corners) and avoids warp->simple fallback when edges are
visible but the neural network missed the corner position.

Pose Refine (--pose-refine)
---------------------------
After the first pose pass, derive a tighter bounding box from the detected
keypoints and re-run the pose model with a smaller crop. This helps when
the detection box is loose or misaligned -- the second pass gets a crop that's
tightly centered on the actual photo, improving corner visibility.

The first pass uses --pose-crop-expand (default 15%) to give the pose model
enough context. The refine pass uses --pose-refine-expand (default 5%) since
the keypoint-derived box is already well-centered.

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
  Detection model:  [1, 5, N_anchors] (legacy) -- raw YOLO output (no NMS)
                    rows = cx, cy, w, h, class_confidence
                    [1, N, 6] (NMS-enabled) -- end-to-end output
                    cols = x1, y1, x2, y2, conf, class
  Pose model:       [1, 300, 18]         -- NMS-enabled (end-to-end)
                    cols = x1,y1,x2,y2,conf,class, kp0_x,kp0_y,kp0_vis,
                           kp1_x,kp1_y,kp1_vis, kp2_x,kp2_y,kp2_vis,
                           kp3_x,kp3_y,kp3_vis

Keypoint Order (matches training convention):
  kp0 = Lower-Left  (LL) -- high Y, low  X
  kp1 = Upper-Left  (UL) -- low  Y, low  X
  kp2 = Upper-Right (UR) -- low  Y, high X
  kp3 = Lower-Right (LR) -- high Y, high X

Cropping
--------
    --crop simple          Bounding-box crop from detection model bbox
    --crop simple-corners  Bounding-box crop from detected corner keypoints
                           (tighter, more accurate than --crop simple)
    --crop warp            Perspective warp (inward): average of opposite
                           edge lengths -> may lose small edge slivers
    --crop warp-stretch    Perspective warp (outward): max of opposite edge
                           lengths -> preserves ALL photo content
    --crop-dir DIR         Output directory for crops (default: ./crops/)
    --crop-margin F        Expand crops outward by F x photo-diagonal on each
                           side. 0.02 = 2% of the photo's diagonal (~20px on a
                           1000px photo). For simple/simple-corners: expands the
                           bounding box. For warp/warp-stretch: pushes source
                           corners outward from the quad center before warping.
    --crop-transparent     Save crops as transparent PNG. For simple crops,
                           area outside keypoint quad is transparent. For
                           warp, out-of-bounds areas are transparent.
    --border-fill COLOR    Fill color for warp background (default: grey).
                           Accepts: R,G,B  #RRGGBB  #RGB  white/black/grey/
                           red/green/blue. Ignored with --crop-transparent.

    Output naming: {original_stem}_{crop_tag}_{photo_id}.{ext}
                  crop_tag: 'warp', 'crop', or 'box' (warp->fallback shows 'crop')
    Example: scan_001_1.jpg, scan_001_2.jpg

Recommended Commands
-------------------
    # Best simple crop -- keypoint-based bbox + 2% margin so edges aren't clipped
    photocrop --image scan.jpg --crop simple-corners --crop-margin 0.02

    # Best warp crop -- outward warp + margin + white fill for clean edges
    photocrop --image scan.jpg --crop warp-stretch \
                     --crop-margin 0.02 --border-fill white

    # Best transparent crop -- corner-based with margin, for compositing
    photocrop --image scan.jpg --crop simple-corners \
                     --crop-margin 0.02 --crop-transparent

    # Crop a whole folder of images
    photocrop --image ./scans/ --output ./crops/ \
                     --crop simple-corners --crop-margin 0.02

    # Best accuracy -- 3-stage refine for improved corner detection
    photocrop --image scan.jpg --pose-refine \
                     --crop warp-stretch --crop-margin 0.02 --border-fill white

Usage
-----
    # Single image (models auto-detected)
    photocrop --image scan.jpg

    # Folder of images
    photocrop --image ./scans/ --output ./crops/

    # Override model paths
    photocrop --detection-model /path/to/det.onnx \
                     --pose-model /path/to/pose.onnx \
                     --image scan.jpg

    # Adjust thresholds
    photocrop --image photo.jpg --det-conf 0.3 --pose-conf 0.25
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Logging -- all diagnostic output goes through _log() which writes to
# stderr.  Coordinate output (--coords) goes to stdout, making it easy
# to capture in scripts:  photocrop --coords json --no-image img.jpg
# ---------------------------------------------------------------------------


def _log(*args, **kwargs):
    """Print diagnostic output to stderr. Never mixed with --coords stdout."""
    print(*args, file=sys.stderr, **kwargs)


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
POSE_CROP_EXPAND = 0.15    # Expand detection box by 15% of larger dim for pose crop
POSE_REFINE_EXPAND = 0.05  # After first pose, re-crop from keypoints + 5% for refine pass
SWEEP_CROP_EXPANDS = [0.05, 0.10, 0.15, 0.20]      # values to try for --pose-sweep first-pass
SWEEP_REFINE_EXPANDS = [0.03, 0.05, 0.10, 0.15]     # values to try for --pose-sweep refine-pass
SWEEP_XY_EXPANDS = [0.05, 0.10, 0.15, 0.20, 0.25]  # per-axis expand values for --pose-sweep-xy
DEFAULT_IMG_SIZE = 640
DEDUP_MIN_DIST_RATIO = 0.12  # 12% of image min-dimension
_VIS_THRESH_DEDUP = 0.25      # visibility threshold for center-based dedup
_VIS_THRESH_SWEEP = 0.30      # visibility threshold for sweep scoring

# Default model paths (relative to this script's location)
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DETECTION_MODEL = _SCRIPT_DIR / ".." / "models" / "detection_ep47.onnx"
DEFAULT_POSE_MODEL = _SCRIPT_DIR / ".." / "models" / "pose_single_ep42.onnx"
DEFAULT_CORNER_REGRESSION_MODEL = _SCRIPT_DIR / ".." / "models" / "corner-regression-v2.onnx"
CORNER_REGRESSION_SIZE = 320  # Corner regression model input size

# Corner refinement: crop size and edge-contact classification
CORNER_CROP_SIZE_MIN = 320   # Minimum crop size for corner regression model
CORNER_REGRESSION_SIZE = 320  # Corner regression model input size


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_onnx_model(model_path: str):
    """Load an ONNX model with onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        _log("Error: onnxruntime not installed.")
        _log("Install with: pip install onnxruntime")
        sys.exit(1)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    for inp in session.get_inputs():
        _log(f"  Input:  {inp.name}  shape={inp.shape}  dtype={inp.type}")
    for out in session.get_outputs():
        _log(f"  Output: {out.name}  shape={out.shape}  dtype={out.type}")
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
    Resize a crop to target_sizextarget_size (stretches, no letterbox).

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

    Supports two detection model output formats:
      - Legacy: [1, 5, N_anchors] -- raw YOLO with columns (cx, cy, w, h, conf).
        Requires NMS post-processing.
      - NMS-enabled: [1, N, 6] -- end-to-end with columns
        (x1, y1, x2, y2, conf, cls). NMS is already applied.

    Returns list of dicts with keys: 'box' (x1,y1,x2,y2 in original coords),
    'confidence'.
    """
    orig_w, orig_h = image.size
    input_arr, scale_info = preprocess_letterbox(image, img_size)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_arr})[0]

    ratio = scale_info["ratio"]
    pad_w = scale_info["pad_w"]
    pad_h = scale_info["pad_h"]

    # Detect output format from shape
    # Legacy: [1, 5, N] -- transposed, 5 rows for cx,cy,w,h,conf
    # NMS-enabled: [1, N, 6] -- row-oriented, 6 cols for x1,y1,x2,y2,conf,cls
    if output.shape[1] == 5 and len(output.shape) == 3:
        # Legacy format: [1, 5, N_anchors] -> cx, cy, w, h, class_conf
        cx = output[0, 0]
        cy = output[0, 1]
        w = output[0, 2]
        h = output[0, 3]
        conf = output[0, 4]

        mask = conf > conf_threshold
        if not mask.any():
            return []

        # Convert cxcywh to xyxy in original image space
        x1 = (cx[mask] - w[mask] / 2 - pad_w) / ratio
        y1 = (cy[mask] - h[mask] / 2 - pad_h) / ratio
        x2 = (cx[mask] + w[mask] / 2 - pad_w) / ratio
        y2 = (cy[mask] + h[mask] / 2 - pad_h) / ratio
        filtered_conf = conf[mask]

        # NMS (needed for legacy format)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(boxes_xyxy, filtered_conf, iou_threshold)

        results = []
        for idx in keep:
            bx1 = max(0, boxes_xyxy[idx, 0])
            by1 = max(0, boxes_xyxy[idx, 1])
            bx2 = min(orig_w, boxes_xyxy[idx, 2])
            by2 = min(orig_h, boxes_xyxy[idx, 3])

            results.append({
                "box": {"x1": float(bx1), "y1": float(by1),
                        "x2": float(bx2), "y2": float(by2)},
                "confidence": float(filtered_conf[idx]),
            })

    else:
        # NMS-enabled format: [1, N, 6] -> x1, y1, x2, y2, conf, cls
        # Coordinates are in letterboxed 640-space
        detections = output[0]  # [N, 6]

        # Filter by confidence
        conf = detections[:, 4]
        mask = conf > conf_threshold
        if not mask.any():
            return []

        filtered = detections[mask]

        # Convert from letterboxed 640-space to original image space
        x1 = (filtered[:, 0] - pad_w) / ratio
        y1 = (filtered[:, 1] - pad_h) / ratio
        x2 = (filtered[:, 2] - pad_w) / ratio
        y2 = (filtered[:, 3] - pad_h) / ratio
        filtered_conf = filtered[:, 4]

        # NMS already applied by model, but apply again for safety
        # (some models may still produce overlapping detections)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(boxes_xyxy, filtered_conf, iou_threshold)

        results = []
        for idx in keep:
            bx1 = max(0, boxes_xyxy[idx, 0])
            by1 = max(0, boxes_xyxy[idx, 1])
            bx2 = min(orig_w, boxes_xyxy[idx, 2])
            by2 = min(orig_h, boxes_xyxy[idx, 3])

            results.append({
                "box": {"x1": float(bx1), "y1": float(by1),
                        "x2": float(bx2), "y2": float(by2)},
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

    Returns list of dicts with keys:
        confidence: detection confidence
        keypoints: list of dicts with 'name', 'x', 'y', 'visibility' in crop pixel coords
        box: dict with x1, y1, x2, y2 in crop pixel coords (bounding box)
    Or empty list if no detection.
    """
    crop_w, crop_h = crop.size
    input_arr, _, _ = preprocess_crop(crop, img_size)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_arr})[0]

    # Output: [1, 300, 18] -- NMS-enabled
    # Columns: x1, y1, x2, y2, conf, cls, kp0_x, kp0_y, kp0_vis, ...
    rows = output[0]

    results = []
    for row in rows:
        conf = row[4]
        if conf < conf_threshold:
            continue

        # Map from 640x640 -> crop pixel coords
        scale_x = crop_w / img_size
        scale_y = crop_h / img_size

        # Bounding box in crop pixel coords
        box_x1 = float(row[0] * scale_x)
        box_y1 = float(row[1] * scale_y)
        box_x2 = float(row[2] * scale_x)
        box_y2 = float(row[3] * scale_y)

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
            "box": {
                "x1": box_x1,
                "y1": box_y1,
                "x2": box_x2,
                "y2": box_y2,
            },
        })

    return results


def run_corner_regression(session, crop: Image.Image,
                          conf_threshold: float = 0.3,
                          img_size: int = CORNER_REGRESSION_SIZE):
    """Run the corner regression model on a 320x320 crop.

    The model detects small bounding boxes around corner points, each with
    one keypoint at the exact corner position.

    Args:
        session: ONNX inference session for the corner regression model.
        crop: PIL Image (should be 320x320, will be resized if not).
        conf_threshold: Minimum detection confidence (default 0.3, lower than
            pose model since corner regression works on tight crops where
            the corner is expected to be present).
        img_size: Model input size (default 320).

    Returns:
        List of dicts with keys: 'confidence', 'keypoint' (dict with x, y,
        visibility), 'box' (dict with x1, y1, x2, y2). Coordinates are in
        crop pixel space. Empty list if no detections above threshold.
    """
    crop_w, crop_h = crop.size
    input_arr, _, _ = preprocess_crop(crop, img_size)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_arr})[0]

    # Output: [1, 300, 9] -- x1, y1, x2, y2, conf, cls, kp_x, kp_y, kp_vis
    rows = output[0]

    # Scale from model input size to crop pixel coords
    scale_x = crop_w / img_size
    scale_y = crop_h / img_size

    results = []
    for row in rows:
        conf = row[4]
        if conf < conf_threshold:
            continue

        box_x1 = float(row[0] * scale_x)
        box_y1 = float(row[1] * scale_y)
        box_x2 = float(row[2] * scale_x)
        box_y2 = float(row[3] * scale_y)

        kp_x = float(row[6] * scale_x)
        kp_y = float(row[7] * scale_y)
        kp_vis = float(row[8])

        results.append({
            "confidence": float(conf),
            "keypoint": {
                "x": kp_x,
                "y": kp_y,
                "visibility": kp_vis,
            },
            "box": {
                "x1": box_x1,
                "y1": box_y1,
                "x2": box_x2,
                "y2": box_y2,
            },
        })

    # Sort by confidence descending
    results.sort(key=lambda d: d["confidence"], reverse=True)
    return results


def _corner_crop(image: Image.Image, x, y, crop_size: int = CORNER_CROP_SIZE_MIN):
    """Extract a square crop centered on (x, y) from a PIL Image.

    If the crop would extend beyond the image boundary, it is shifted
    to stay within bounds. The returned crop is always crop_size x crop_size
    (padded with grey if the image is too small).

    Args:
        image: PIL Image
        x, y: Center point in original image coordinates (float)
        crop_size: Size of the square crop (default: CORNER_CROP_SIZE_MIN=320)

    Returns:
        crop: PIL Image (crop_size x crop_size)
        offset_x, offset_y: top-left corner of the crop in original image coords
    """
    orig_w, orig_h = image.size
    half = crop_size // 2

    # Center the crop on (x, y) but clamp to image bounds
    x1 = int(round(x - half))
    y1 = int(round(y - half))

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x1 + crop_size > orig_w:
        x1 = orig_w - crop_size
    if y1 + crop_size > orig_h:
        y1 = orig_h - crop_size

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(orig_w, x1 + crop_size)
    y2 = min(orig_h, y1 + crop_size)

    crop = image.crop((x1, y1, x2, y2))

    if crop.size[0] < crop_size or crop.size[1] < crop_size:
        padded = Image.new("RGB", (crop_size, crop_size), (114, 114, 114))
        padded.paste(crop, (0, 0))
        crop = padded

    return crop, x1, y1


def _bbox_corner_coord(box: dict, corner_name: str) -> tuple:
    """Return the bbox corner that approximates the fiducial position.

    For a detection model (no keypoints), the relevant bbox corner is a
    better approximation of the photo corner than the bbox center:
      UL -> (x1, y1)    UR -> (x2, y1)
      LL -> (x1, y2)    LR -> (x2, y2)
    """
    cx = box["x1"] if corner_name in ("UL", "LL") else box["x2"]
    cy = box["y1"] if corner_name in ("UL", "UR") else box["y2"]
    return cx, cy


def refine_corners_regression(image: Image.Image, result: dict,
                               corner_regression_session,
                               iterations: int = 2,
                               conf_threshold: float = 0.3,
                               crop_size: int = CORNER_REGRESSION_SIZE,
                               max_shift_ratio: float = 0.3):
    """Refine corner positions using the corner regression model.

    The corner regression model is a specialized YOLO26n-pose model trained
    on 320x320 crops of corner regions. It detects small bounding boxes around
    corner points, with the keypoint at the exact corner position.

    Strategy:
      1. Use the pose model's approximate corner position as a starting point
      2. Crop a 320x320 region around that approximate corner
      3. Run the corner regression model on the crop
      4. If multiple corners are detected, pick the one closest to the
         expected corner position (based on pose model approximation)
      5. Use the detection's keypoint as the refined corner position
      6. Optionally iterate: re-crop around the refined position

    This is more accurate than the pose model alone because the regression
    model was specifically trained on tight corner crops and knows exactly
    what a corner looks like at close range. It's also more robust than
    geometric edge classification because it doesn't rely on which edges of
    the crop the bbox touches.

    Args:
        image: Full input PIL Image
        result: Pipeline result dict with 'keypoints' and 'detection'/'box'
        corner_regression_session: ONNX session for the corner regression model
        iterations: Number of refinement iterations (default 2)
        conf_threshold: Detection confidence threshold (default 0.3, lower
            than pose since we expect a corner to be present)
        crop_size: Crop size for the regression model (default 320)
        max_shift_ratio: Max allowed shift as ratio of crop_size (default 0.3)

    Returns:
        The same result dict with updated keypoint coordinates and
        visibility boosted to 1.0 for successfully refined corners.
    """
    kps = {kp["name"]: kp for kp in result.get("keypoints", [])}
    box = result["detection"]["box"]
    max_shift = crop_size * max_shift_ratio

    # Minimum confidence to accept a detection without further iteration.
    # When confidence is low (e.g., adjacent photos sharing an edge), the
    # model may be picking up the wrong corner. Re-cropping around the
    # detection gives the model a tighter view, often boosting confidence
    # and accuracy on the next iteration.
    MIN_CONF_ACCEPT = 0.7
    MAX_EXTRA_ITERS = 3  # Extra iterations beyond the base count
    # Lower confidence threshold for candidate detection search. When two
    # photos share an edge, the correct corner may have very low confidence
    # (0.1-0.4) while an adjacent photo's corner scores high (0.9+). Using
    # a lower threshold for discovery ensures we find the right corner, and
    # then proximity to the pose reference (not confidence) determines which
    # one we pick.
    SEARCH_CONF = 0.05

    # Use pose keypoints as starting points, fall back to bbox corners
    approx_corners = {}
    for name in ["UL", "UR", "LL", "LR"]:
        if name in kps and kps[name]["visibility"] >= 0.1:
            approx_corners[name] = (kps[name]["x"], kps[name]["y"])
        else:
            # Fall back to bbox corner
            approx_corners[name] = _bbox_corner_coord(box, name)

    refined_corners = {}

    for corner_name, (ax, ay) in approx_corners.items():
        orig_x, orig_y = ax, ay  # Pose model's original position

        # Keep iterating until we get a confident detection or run out of
        # iterations. Base iterations cover the initial search, then extra
        # iterations kick in when confidence is low to keep refining.
        max_iters = iterations + MAX_EXTRA_ITERS
        best_result = None  # Track closest+confident result across iterations

        for iteration in range(max_iters):
            # Crop a 320x320 region around the approximate corner
            crop, offset_x, offset_y = _corner_crop(image, ax, ay, crop_size)

            # Run the corner regression model with a low confidence threshold
            # for finding candidates. We'll use proximity to pick the right
            # one, not the model's confidence score.
            detections = run_corner_regression(
                corner_regression_session, crop,
                conf_threshold=SEARCH_CONF,
                img_size=CORNER_REGRESSION_SIZE,
            )

            if not detections:
                _log(f"  {corner_name}: no corner regression detections "
                     f"at ({ax:.0f},{ay:.0f}) iter={iteration}")
                break

            # Find the detection whose keypoint is closest to the ORIGINAL
            # pose model position (orig_x, orig_y), not the crop center.
            # Using the original position ensures we pick the corner belonging
            # to the target photo, not an adjacent photo's corner that may
            # appear in the same crop with higher confidence.
            ref_x = orig_x - offset_x  # Original pose position in crop coords
            ref_y = orig_y - offset_y

            best_det = None
            best_dist = float('inf')

            for det in detections:
                kp = det["keypoint"]
                if kp["visibility"] < 0.3:
                    continue
                dx = kp["x"] - ref_x
                dy = kp["y"] - ref_y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_det = det

            if best_det is None:
                _log(f"  {corner_name}: no visible keypoint in regression "
                     f"detections at iter={iteration}")
                break

            # Map keypoint back to original image coordinates
            new_x = best_det["keypoint"]["x"] + offset_x
            new_y = best_det["keypoint"]["y"] + offset_y

            # Validate: reject if refinement moved too far from original
            if (abs(new_x - orig_x) > max_shift or
                    abs(new_y - orig_y) > max_shift):
                _log(f"  {corner_name}: regression moved too far "
                     f"({new_x:.0f},{new_y:.0f}) vs "
                     f"({orig_x:.0f},{orig_y:.0f})")
                break

            conf = best_det["confidence"]
            _log(f"  {corner_name}: regression refined to ({new_x:.1f},{new_y:.1f}) "
                 f"(conf={conf:.3f}, dist_from_ref={best_dist:.1f}px, iter={iteration})")

            # Track the best result for this corner. Prefer the closest
            # detection to the reference with confidence above threshold.
            if best_result is None or conf > best_result[2]:
                best_result = (new_x, new_y, conf)

            # Accept the result if confidence is high enough
            if conf >= MIN_CONF_ACCEPT:
                ax, ay = new_x, new_y
                break  # Confident — no need to iterate further

            # Low confidence: re-crop around this detection and try again.
            # The tighter crop gives the model a better view of just the
            # target corner, often boosting confidence significantly.
            ax, ay = new_x, new_y

        # Use the best result found (even if confidence is low)
        if best_result is not None:
            ax, ay = best_result[0], best_result[1]

        refined_corners[corner_name] = (ax, ay)

    # Update the result dict with refined positions    # Update the result dict with refined positions
    for name in ["UL", "UR", "LL", "LR"]:
        if name in refined_corners:
            rx, ry = refined_corners[name]
            if name in kps:
                # Save original visibility before refinement boosted it to 1.0
                if "nn_vis" not in kps[name]:
                    kps[name]["nn_vis"] = kps[name]["visibility"]
                kps[name]["x"] = rx
                kps[name]["y"] = ry
                kps[name]["visibility"] = 1.0
            else:
                result.setdefault("keypoints", []).append({
                    "name": name,
                    "x": rx,
                    "y": ry,
                    "visibility": 1.0,
                    "nn_vis": 0.0,
                })

    return result


def dedup_pose_results(pose_results: list, min_center_dist: float):
    """
    Greedy deduplication: sort by dedup priority descending, keep each
    result only if its keypoint center is at least min_center_dist pixels
    away from all previously kept results.

    This handles overlapping detection boxes and "combined" boxes that
    span multiple photos -- the pose model correctly localizes whichever
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

def _compute_crop_limits(detections: list, image_w: int, image_h: int,
                          max_intrusion_ratio: float = 0.15) -> list:
    """
    For each detection box, compute how far the crop can expand in each
    direction (left, right, up, down) before including too much content
    from a neighboring detection box.

    For each direction, the limit is the minimum of:
      1. The distance to the image edge (hard boundary).
      2. The gap to the adjacent box + a small intrusion allowance.

    The intrusion allowance is ``max_intrusion_ratio`` of the current box's
    dimension -- this allows the crop to extend a small amount into the
    neighboring photo (which provides useful edge context) without pulling
    in so much that the pose model gets confused by seeing corners from
    the adjacent photo.

    For example, with a 2x2 grid where boxes are ~14px apart and each
    box is ~700px wide, max_intrusion_ratio=0.15 allows expansion up to
    14px + 0.15*700 = 119px past the box edge -- enough context for the
    pose model without including too much of the adjacent photo.

    Args:
        detections: list of detection dicts with 'box' keys
        image_w, image_h: image dimensions
        max_intrusion_ratio: max fraction of this box's dimension that
            can extend into an adjacent box (default 0.15 = 15%).

    Returns:
        list of dicts with keys 'left', 'right', 'up', 'down' (pixels)
        for each detection, capping expansion at the image boundary.
    """
    boxes = [d["box"] for d in detections]
    limits = []

    for i, box in enumerate(boxes):
        box_w = box["x2"] - box["x1"]
        box_h = box["y2"] - box["y1"]

        # Default: expand up to image edges
        left_limit = box["x1"]            # pixels available left of box
        right_limit = image_w - box["x2"]  # pixels available right of box
        up_limit = box["y1"]              # pixels available above box
        down_limit = image_h - box["y2"]   # pixels available below box

        # For each other box, compute how far we can expand toward it
        # without including too much of its content.
        for j, other in enumerate(boxes):
            if i == j:
                continue

            other_w = other["x2"] - other["x1"]
            other_h = other["y2"] - other["y1"]

            # Other is to the left and vertically overlaps
            if other["x2"] <= box["x1"]:
                # We can expand left by: gap + small intrusion into other box
                gap = box["x1"] - other["x2"]
                intrusion = max(box_w * max_intrusion_ratio,
                                other_w * max_intrusion_ratio)
                limit = int(gap + intrusion)
                left_limit = min(left_limit, limit)

            # Other is to the right and vertically overlaps
            if other["x1"] >= box["x2"]:
                gap = other["x1"] - box["x2"]
                intrusion = max(box_w * max_intrusion_ratio,
                                other_w * max_intrusion_ratio)
                limit = int(gap + intrusion)
                right_limit = min(right_limit, limit)

            # Other is above and horizontally overlaps
            if other["y2"] <= box["y1"]:
                gap = box["y1"] - other["y2"]
                intrusion = max(box_h * max_intrusion_ratio,
                                other_h * max_intrusion_ratio)
                limit = int(gap + intrusion)
                up_limit = min(up_limit, limit)

            # Other is below and horizontally overlaps
            if other["y1"] >= box["y2"]:
                gap = other["y1"] - box["y2"]
                intrusion = max(box_h * max_intrusion_ratio,
                                other_h * max_intrusion_ratio)
                limit = int(gap + intrusion)
                down_limit = min(down_limit, limit)

        limits.append({
            "left": left_limit,
            "right": right_limit,
            "up": up_limit,
            "down": down_limit,
        })

    return limits


def _center_biased_expand(box: dict, orig_w: int, orig_h: int,
                          expand_px: int, crop_limits: dict = None) -> tuple:
    """
    Expand a detection box asymmetrically, biasing expansion toward the
    image center.

    Instead of expanding equally in all directions (which wastes context
    toward image edges where there's usually just background), this
    shifts more of the expansion budget toward the image center, where
    the photo content and useful context typically are.

    The bias is proportional to how far each box edge is from the
    corresponding image edge. For a box near the left edge, the left
    side gets less expansion (not much useful content there) and the
    right side gets more (toward the center where other photos or
    context may be).

    If ``crop_limits`` is provided, expansion is also capped per-side
    to avoid including too much of adjacent photos.

    Returns:
        (crop_x1, crop_y1, crop_x2, crop_y2)
    """
    x1, y1 = box["x1"], box["y1"]
    x2, y2 = box["x2"], box["y2"]

    # Distance from each box edge to the corresponding image edge
    dist_left = x1
    dist_right = orig_w - x2
    dist_up = y1
    dist_down = orig_h - y2

    # Weight each direction by how far it is from the image edge
    # (closer to center = more expansion budget)
    total_h = dist_left + dist_right
    total_v = dist_up + dist_down

    # Allocate expansion budget proportionally
    # If both sides are equally far, we get 50/50 (same as symmetric)
    # If one side is near the edge, most budget goes to the other side
    if total_h > 0:
        weight_left = dist_left / total_h
        weight_right = dist_right / total_h
    else:
        weight_left = weight_right = 0.5

    if total_v > 0:
        weight_up = dist_up / total_v
        weight_down = dist_down / total_v
    else:
        weight_up = weight_down = 0.5

    # Total expansion budget per axis = 2 * expand_px (half on each side
    # when symmetric). Distribute according to weights.
    expand_left = int(2 * expand_px * weight_left)
    expand_right = int(2 * expand_px * weight_right)
    expand_up = int(2 * expand_px * weight_up)
    expand_down = int(2 * expand_px * weight_down)

    # Apply crop_limits if provided
    if crop_limits is not None:
        expand_left = min(expand_left, crop_limits["left"])
        expand_right = min(expand_right, crop_limits["right"])
        expand_up = min(expand_up, crop_limits["up"])
        expand_down = min(expand_down, crop_limits["down"])

    crop_x1 = max(0, int(x1) - expand_left)
    crop_y1 = max(0, int(y1) - expand_up)
    crop_x2 = min(orig_w, int(x2) + expand_right)
    crop_y2 = min(orig_h, int(y2) + expand_down)

    return crop_x1, crop_y1, crop_x2, crop_y2


def _run_pose_on_crop(pose_session, image: Image.Image, box: dict,
                     pose_conf: float, img_size: int,
                     expand_ratio: float = None,
                     expand_ratio_x: float = None,
                     expand_ratio_y: float = None,
                     crop_limits: dict = None,
                     center_bias: bool = False) -> dict | None:
    """
    Run the pose model on a single detection box, expanding the crop
    before running pose.

    Expansion can be specified in three ways:
      - ``expand_ratio`` (scalar): expand both axes by this ratio of the
        box's larger dimension (uniform, backward-compatible).
      - ``expand_ratio_x`` + ``expand_ratio_y``: expand independently.
        Each is a ratio of the respective box dimension (width/height).
      - If only one of expand_ratio_x/y is given, the other defaults
        to the ``expand_ratio`` value (or 0).

    Returns a result dict with keypoints mapped to original image coords,
    or None if no pose detection above threshold.
    """
    orig_w, orig_h = image.size
    x1, y1 = box["x1"], box["y1"]
    x2, y2 = box["x2"], box["y2"]

    box_w = x2 - x1
    box_h = y2 - y1

    # Resolve expansion ratios
    if expand_ratio is not None and expand_ratio_x is None and expand_ratio_y is None:
        # Legacy scalar mode: uniform expansion based on larger dim
        expand_px_x = int(max(box_w, box_h) * expand_ratio)
        expand_px_y = expand_px_x
    else:
        # Independent X/Y mode
        rx = expand_ratio_x if expand_ratio_x is not None else (expand_ratio or 0)
        ry = expand_ratio_y if expand_ratio_y is not None else (expand_ratio or 0)
        expand_px_x = int(box_w * rx)
        expand_px_y = int(box_h * ry)

    # Compute crop region with optional center-biased expansion
    if center_bias:
        # Center-biased: shift more expansion budget toward image center
        crop_x1, crop_y1, crop_x2, crop_y2 = _center_biased_expand(
            box, orig_w, orig_h, max(expand_px_x, expand_px_y),
            crop_limits=crop_limits,
        )
    else:
        # Symmetric per-side expansion with crop limits (default)
        expand_left = expand_px_x
        expand_right = expand_px_x
        expand_up = expand_px_y
        expand_down = expand_px_y
        if crop_limits is not None:
            expand_left = min(expand_px_x, crop_limits["left"])
            expand_right = min(expand_px_x, crop_limits["right"])
            expand_up = min(expand_px_y, crop_limits["up"])
            expand_down = min(expand_px_y, crop_limits["down"])

        crop_x1 = max(0, int(x1) - expand_left)
        crop_y1 = max(0, int(y1) - expand_up)
        crop_x2 = min(orig_w, int(x2) + expand_right)
        crop_y2 = min(orig_h, int(y2) + expand_down)

    crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    pose_dets = run_pose(pose_session, crop, pose_conf, img_size)
    if not pose_dets:
        return None

    best_pose = max(pose_dets, key=lambda p: p["confidence"])

    # Map keypoints from crop coords -> original image coords
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
        return None

    center = (float(np.mean(visible_xs)), float(np.mean(visible_ys)))
    vis_count = len(visible_xs)
    dedup_priority = best_pose["confidence"]
    if vis_count < 3:
        dedup_priority *= 0.5

    return {
        "pose_confidence": best_pose["confidence"],
        "keypoints": mapped_keypoints,
        "center": center,
        "dedup_priority": dedup_priority,
        "crop_box": {
            "x1": crop_x1, "y1": crop_y1,
            "x2": crop_x2, "y2": crop_y2,
        },
    }


def _keypoints_to_bbox(keypoints: list, margin: float = 0,
                       margin_x: float = None, margin_y: float = None) -> dict:
    """
    Derive a bounding box from detected corner keypoints.

    If at least 2 visible corners are available, uses their positions.
    Falls back to the detection bbox corners for invisible keypoints.

    Expansion can be uniform (``margin``) or independent (``margin_x``, ``margin_y``).
    If margin_x/y are given, they override margin on the respective axis.
    """
    kp_by_name = {kp["name"]: kp for kp in keypoints}
    corners = {}
    for name in ["LL", "UL", "UR", "LR"]:
        kp = kp_by_name.get(name)
        if kp and kp["visibility"] >= _VIS_THRESH_DEDUP:
            corners[name] = (kp["x"], kp["y"])

    if len(corners) < 2:
        return None

    xs = [c[0] for c in corners.values()]
    ys = [c[1] for c in corners.values()]

    mx = margin_x if margin_x is not None else margin
    my = margin_y if margin_y is not None else margin

    x1 = min(xs) - mx
    y1 = min(ys) - my
    x2 = max(xs) + mx
    y2 = max(ys) + my

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def pipeline(detection_session, pose_session, image: Image.Image,
            det_conf: float = 0.5, pose_conf: float = 0.5,
            iou_threshold: float = 0.45,
            pose_crop_expand: float = POSE_CROP_EXPAND,
            pose_refine: bool = False,
            pose_refine_expand: float = POSE_REFINE_EXPAND,
            img_size: int = DEFAULT_IMG_SIZE,
            dedup_min_dist_ratio: float = DEDUP_MIN_DIST_RATIO,
            center_bias: bool = False,
            cv_refine: bool = False,
            cv_refine_radius: int = 40):
    """
    Two- or three-stage pipeline: detect -> pose -> [refine ->] dedup -> rescue.

    Two-stage (default):
        Detection model finds bounding boxes -> pose model finds corners.

    Three-stage refine (``pose_refine=True``):
        Detection -> pose (coarse, with expanded crop) -> derive a tighter
        bounding box from the first pass keypoints -> pose again (refined,
        with a smaller crop centered on the actual photo). This helps
        when the first pose pass has low-visibility corners because the
        detection box is too loose or misaligned -- the second pass gets
        a crop that's tightly centered on the actual photo.

    CV refinement (optional):
        ``cv_refine``: Sobel edge detection + line intersection on ALL
        corners, regardless of visibility. Useful for fine-tuning already-
        good corners. Applied before the automatic rescue step.

    Automatic rescue (always on):
        After pose (and optional CV refinement), photos with fewer than 3
        visible corners are rescued using Sobel edge detection + line
        intersection. This prevents degenerate crops (e.g., "strips" from
        only 2 visible corners) and avoids warp->simple fallback. Only
        photos that need it are processed.

    Returns list of dicts:
        {
            'detection': {'box': {...}, 'confidence': float},
            'pose_confidence': float,
            'keypoints': [...],
            'center': (float, float),
        }
    """
    orig_w, orig_h = image.size

    # Stage 1: Detect photos
    if detection_session is not None:
        detections = run_detection(
            detection_session, image, det_conf, iou_threshold, img_size
        )
    else:
        detections = [{
            "box": {"x1": 0, "y1": 0, "x2": orig_w, "y2": orig_h},
            "confidence": 1.0,
        }]

    # Compute crop limits: cap expansion so we don't pull in content
    # from adjacent detection boxes (which confuses the pose model)
    crop_limits_list = _compute_crop_limits(detections, orig_w, orig_h)

    # Stage 2: Find corners for each detected photo
    pose_results = []
    for det_idx, det in enumerate(detections):
        result = _run_pose_on_crop(
            pose_session, image, det["box"],
            pose_conf, img_size, pose_crop_expand,
            crop_limits=crop_limits_list[det_idx],
            center_bias=center_bias,
        )
        if result is None:
            continue

        # Stage 2b (optional): Refine by re-running on a tighter crop
        # derived from the first pass keypoints. Refine crops are derived
        # from keypoints, not detection boxes, so they don't need crop limits.
        if pose_refine:
            refined_box = _keypoints_to_bbox(result["keypoints"])
            if refined_box is not None:
                # Use standard 4-keypoint pose model for refinement
                refined = _run_pose_on_crop(
                    pose_session, image, refined_box,
                    pose_conf, img_size, pose_refine_expand,
                )
                if refined is not None:
                    result = refined

        result["detection"] = det
        pose_results.append(result)

    # Stage 3: Deduplicate by keypoint-center proximity
    if detection_session is not None and len(pose_results) > 1:
        min_dist = min(orig_w, orig_h) * dedup_min_dist_ratio
        pose_results = dedup_pose_results(pose_results, min_dist)

    # Stage 4 (optional): CV corner refinement on all photos
    if cv_refine:
        pose_results = refine_corners_cv(image, pose_results,
                                          search_radius=cv_refine_radius)

    # Stage 5: Rescue low-visibility corners (always on)
    # When photos have fewer than 3 visible corners, the results are
    # degenerate (e.g., "strip" crops, or warp->simple fallback). CV
    # edge refinement can recover invisible corners using Sobel edges
    # and neighbor-anchored projection. This runs automatically -- no
    # flag needed -- because leaving a photo with 2/4 corners is always
    # worse than attempting recovery.
    pose_results = rescue_low_vis_corners(image, pose_results)

    return pose_results


# ---------------------------------------------------------------------------
# Parameter sweep -- search for best pose-crop-expand / pose-refine-expand
# ---------------------------------------------------------------------------

def pose_sweep(detection_session, pose_session, image: Image.Image,
               det_conf: float = 0.5, pose_conf: float = 0.5,
               iou_threshold: float = 0.45,
               crop_expands: list = None,
               refine_expands: list = None,
               img_size: int = DEFAULT_IMG_SIZE,
               dedup_min_dist_ratio: float = DEDUP_MIN_DIST_RATIO):
    """
    Search for the best pose-crop-expand and pose-refine-expand values
    by trying a grid of combinations and scoring each by corner visibility.

    Strategy:
      1. Run detection once (shared across all combos).
      2. For each detection box, try every (crop_expand, refine_expand)
         combination.  We also include refine_expand=None (two-stage only)
         for each crop_expand.
      3. Score each combo per-photo using:
           - min corner visibility across all 4 corners (higher is better)
           - number of visible corners (vis >= threshold, higher is better)
           - pose confidence as tiebreaker
      4. Pick the best combo for each photo independently.
      5. Report a summary table and return results using the best params.

    Returns:
        (results, best_params_per_photo)
        where best_params_per_photo[i] = {"crop_expand": float, "refine_expand": float|None}
    """
    if crop_expands is None:
        crop_expands = list(SWEEP_CROP_EXPANDS)
    if refine_expands is None:
        refine_expands = list(SWEEP_REFINE_EXPANDS)

    orig_w, orig_h = image.size

    # Stage 1: Detect photos (shared)
    if detection_session is not None:
        detections = run_detection(
            detection_session, image, det_conf, iou_threshold, img_size
        )
    else:
        detections = [{
            "box": {"x1": 0, "y1": 0, "x2": orig_w, "y2": orig_h},
            "confidence": 1.0,
        }]

    # Compute crop limits so sweeps don't expand past adjacent photos
    crop_limits_list = _compute_crop_limits(detections, orig_w, orig_h)

    # Build list of (crop_expand, refine_expand_or_None) combos
    combos = []
    for ce in crop_expands:
        combos.append((ce, None))  # two-stage (no refine)
        for re in refine_expands:
            combos.append((ce, re))  # three-stage refine

    # For each detection, run all combos and find the best
    best_results = []
    best_params = []

    _log(f"\n{'=' * 80}")
    _log(f"POSE PARAMETER SWEEP  --  {len(detections)} detection(s), "
          f"{len(combos)} combo(s) per detection")
    _log(f"{'=' * 80}")
    _log(f"  crop_expands:   {crop_expands}")
    _log(f"  refine_expands: {refine_expands}")
    _log(f"  (also trying 2-stage [no refine] for each crop_expand)\n")

    for det_idx, det in enumerate(detections):
        box = det["box"]
        _log(f"  Photo #{det_idx+1}:  box=({box['x1']:.0f},{box['y1']:.0f})"
              f"->({box['x2']:.0f},{box['y2']:.0f})  det_conf={det['confidence']:.3f}")
        _log(f"  {'crop_exp':>9}  {'refn_exp':>9}  "
              f"{'pose_conf':>9}  "
              f"{'vis_corners':>11}  "
              f"{'min_vis':>7}  "
              f"{'LL':>6}  {'UL':>6}  {'UR':>6}  {'LR':>6}")
        _log(f"  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*11}  {'-'*7}  "
              f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

        best_score = None
        best_result = None
        best_combo = None

        for ce, re in combos:
            # Run pose with this crop_expand (crop_limits cap expansion)
            result = _run_pose_on_crop(
                pose_session, image, box, pose_conf, img_size, ce,
                crop_limits=crop_limits_list[det_idx],
            )
            if result is None:
                label = f"{'--':>6}" if re is None else f"{re:.2f}"
                _log(f"  {ce:9.2f}  {label:>9}  {'--':>9}  {'--':>11}  {'--':>7}  "
                      f"{'--':>6}  {'--':>6}  {'--':>6}  {'--':>6}")
                continue

            # Optionally refine
            if re is not None:
                refined_box = _keypoints_to_bbox(result["keypoints"])
                if refined_box is not None:
                    refined = _run_pose_on_crop(
                        pose_session, image, refined_box,
                        pose_conf, img_size, re
                    )
                    if refined is not None:
                        result = refined

            # Score this result
            kps = result["keypoints"]
            vis_values = {kp["name"]: kp["visibility"] for kp in kps}
            vis_count = sum(1 for v in vis_values.values() if v >= _VIS_THRESH_SWEEP)
            min_vis = min(vis_values.values()) if vis_values else 0.0
            pose_con = result["pose_confidence"]

            # Scoring: prioritize (vis_count, min_vis), then pose_conf as tiebreak
            score = (vis_count, min_vis, pose_con)

            re_label = f"{re:.2f}" if re is not None else "  --"
            ll = vis_values.get("LL", 0)
            ul = vis_values.get("UL", 0)
            ur = vis_values.get("UR", 0)
            lr = vis_values.get("LR", 0)
            _log(f"  {ce:9.2f}  {re_label:>9}  {pose_con:9.3f}  "
                  f"{vis_count:>7}/4  {min_vis:7.3f}  "
                  f"{ll:6.3f}  {ul:6.3f}  {ur:6.3f}  {lr:6.3f}")

            if best_score is None or score > best_score:
                best_score = score
                best_result = result
                best_combo = (ce, re)

        # Print winner for this photo
        if best_result is not None:
            ce_best, re_best = best_combo
            kps = best_result["keypoints"]
            vis_values = {kp["name"]: kp["visibility"] for kp in kps}
            vis_count = sum(1 for v in vis_values.values() if v >= _VIS_THRESH_SWEEP)
            min_vis = min(vis_values.values())
            re_str = f"refine={re_best:.2f}" if re_best is not None else "no refine"
            _log(f"  >>> BEST: crop_expand={ce_best:.2f}  {re_str}  "
                  f"vis={vis_count}/4  min_vis={min_vis:.3f}\n")

            best_result["detection"] = det
            best_results.append(best_result)
            best_params.append({
                "crop_expand": ce_best,
                "refine_expand": re_best,
            })
        else:
            _log(f"  >>> NO valid pose result for this photo\n")

    # Dedup the final results
    if detection_session is not None and len(best_results) > 1:
        min_dist = min(orig_w, orig_h) * dedup_min_dist_ratio
        best_results = dedup_pose_results(best_results, min_dist)

    # Summary
    _log(f"\n{'=' * 80}")
    _log("SWEEP SUMMARY -- best params per photo:")
    _log(f"{'=' * 80}")
    for i, (res, params) in enumerate(zip(best_results, best_params)):
        kps = res["keypoints"]
        vis_values = {kp["name"]: kp["visibility"] for kp in kps}
        vis_count = sum(1 for v in vis_values.values() if v >= _VIS_THRESH_SWEEP)
        re_str = (f"refine={params['refine_expand']:.2f}"
                  if params["refine_expand"] is not None else "no refine")
        _log(f"  Photo #{i+1}:  crop_expand={params['crop_expand']:.2f}  {re_str}  "
              f"vis={vis_count}/4  min_vis={min(vis_values.values()):.3f}  "
              f"pose_conf={res['pose_confidence']:.3f}")

    return best_results, best_params


# ---------------------------------------------------------------------------
# Independent X/Y sweep -- search per-axis expansion values
# ---------------------------------------------------------------------------

def pose_sweep_xy(detection_session, pose_session, image: Image.Image,
                  det_conf: float = 0.5, pose_conf: float = 0.5,
                  iou_threshold: float = 0.45,
                  x_expands: list = None,
                  y_expands: list = None,
                  refine_expands: list = None,
                  img_size: int = DEFAULT_IMG_SIZE,
                  dedup_min_dist_ratio: float = DEDUP_MIN_DIST_RATIO,
                  early_stop_vis: float = 0.70):
    """
    Adaptive X/Y parameter sweep with early stopping.

    Strategy -- 3 tiers, stopping as soon as a "good enough" result is found:

      Tier 1 (quick): Try a small grid of moderate values without refine.
        If any combo achieves 4/4 visible corners with all vis >= early_stop_vis,
        we're done -- skip refine and wider search entirely.

      Tier 2 (refine): If tier 1 found 4/4 corners but some are below threshold,
        add a refine pass on the best candidates from tier 1.

      Tier 3 (wide): If tier 1 didn't find 4/4 corners at all, expand the
        search to more extreme values.

    Each tier reuses cached first-pass results for refine -- no redundant
    pose inference calls.

    Scoring: (visible_corners, min_visibility, pose_confidence).
    A result is "good enough" when vis_count == 4 AND min_vis >= early_stop_vis.

    Returns:
        (results, best_params_per_photo)
        best_params[i] = {"expand_x": float, "expand_y": float,
                          "refine_x": float|None, "refine_y": float|None}
    """
    if x_expands is None:
        x_expands = list(SWEEP_XY_EXPANDS)
    if y_expands is None:
        y_expands = list(SWEEP_XY_EXPANDS)
    if refine_expands is None:
        refine_expands = list(SWEEP_REFINE_EXPANDS)

    # Tier 1: moderate "likely good" values -- typically hits early stop
    tier1_values = [v for v in [0.10, 0.15, 0.20] if v in x_expands or v in y_expands]
    tier1_x = [v for v in [0.10, 0.15, 0.20] if v in x_expands]
    tier1_y = [v for v in [0.10, 0.15, 0.20] if v in y_expands]
    if not tier1_x:
        tier1_x = x_expands[:3]
    if not tier1_y:
        tier1_y = y_expands[:3]

    # Tier 3: everything else (more extreme values)
    tier3_x = [v for v in x_expands if v not in tier1_x]
    tier3_y = [v for v in y_expands if v not in tier1_y]

    orig_w, orig_h = image.size

    # Stage 1: Detect photos (shared across all tiers)
    if detection_session is not None:
        detections = run_detection(
            detection_session, image, det_conf, iou_threshold, img_size
        )
    else:
        detections = [{
            "box": {"x1": 0, "y1": 0, "x2": orig_w, "y2": orig_h},
            "confidence": 1.0,
        }]

    # Compute crop limits so expansions don't bleed into adjacent photos
    crop_limits_list = _compute_crop_limits(detections, orig_w, orig_h)

    # Refine candidates: only the top-N first-pass results that need help
    max_refine_candidates = 3

    best_results = []
    best_params = []

    _log(f"\n{'=' * 90}")
    _log(f"ADAPTIVE X/Y SWEEP  --  {len(detections)} detection(s)")
    _log(f"{'=' * 90}")
    total_tier1 = len(tier1_x) * len(tier1_y)
    _log(f"  Tier 1 (quick): {tier1_x} x {tier1_y} = {total_tier1} combos")
    if tier3_x or tier3_y:
        _log(f"  Tier 3 (wide):  expand to full grid if needed")
    _log(f"  Refine candidates: top {max_refine_candidates} from tier 1, using {refine_expands}")
    _log(f"  Early stop: vis=4/4 and min_vis >= {early_stop_vis:.2f}")

    # ── "Good enough" threshold for skipping Tier 2 refine ──
    # If Tier 1 finds 4/4 corners with min_vis at or above this value,
    # Tier 2 refine is skipped -- the result is already good enough for
    # accurate perspective warp. This defaults to 0.5, which is below
    # early_stop_vis (0.7) but above the minimum for reliable cropping.
    _TIER2_SKIP_VIS = 0.50

    for det_idx, det in enumerate(detections):
        box = det["box"]
        bw = box["x2"] - box["x1"]
        bh = box["y2"] - box["y1"]
        _log(f"\n  Photo #{det_idx+1}:  box=({box['x1']:.0f},{box['y1']:.0f})"
              f"->({box['x2']:.0f},{box['y2']:.0f})  "
              f"size={bw:.0f}x{bh:.0f}  det_conf={det['confidence']:.3f}")

        # ── Tier 1: Quick grid, no refine ──
        tier1_results = []  # [(ex, ey, result), ...]
        best_score = None
        best_result = None
        best_combo = None
        early_stop = False

        _log(f"  --- Tier 1: quick grid (no refine) ---")
        _log(f"  {'exp_x':>6} {'exp_y':>6} {'ref_x':>6} {'ref_y':>6}  "
              f"{'p_conf':>6} {'vis':>4} {'min_v':>6}  "
              f"{'LL':>6} {'UL':>6} {'UR':>6} {'LR':>6}")

        for ex in tier1_x:
            for ey in tier1_y:
                result = _run_pose_on_crop(
                    pose_session, image, box, pose_conf, img_size,
                    expand_ratio_x=ex, expand_ratio_y=ey,
                    crop_limits=crop_limits_list[det_idx],
                )
                if result is None:
                    _log(f"  {ex:6.2f} {ey:6.2f} {'--':>6} {'--':>6}  "
                          f"{'--':>6} {'--':>4} {'--':>6}  "
                          f"{'--':>6} {'--':>6} {'--':>6} {'--':>6}")
                    continue

                kps = result["keypoints"]
                vis_values = {kp["name"]: kp["visibility"] for kp in kps}
                vis_count = sum(1 for v in vis_values.values()
                                if v >= _VIS_THRESH_SWEEP)
                min_vis = min(vis_values.values()) if vis_values else 0.0
                pose_con = result["pose_confidence"]
                score = (vis_count, min_vis, pose_con)

                ll = vis_values.get("LL", 0)
                ul = vis_values.get("UL", 0)
                ur = vis_values.get("UR", 0)
                lr = vis_values.get("LR", 0)
                _log(f"  {ex:6.2f} {ey:6.2f} {'--':>6} {'--':>6}  "
                      f"{pose_con:6.3f} {vis_count:4}/4 {min_vis:6.3f}  "
                      f"{ll:6.3f} {ul:6.3f} {ur:6.3f} {lr:6.3f}")

                tier1_results.append((ex, ey, result, score))

                if best_score is None or score > best_score:
                    best_score = score
                    best_result = result
                    best_combo = (ex, ey, None, None)

                # Early stop: 4/4 corners all above threshold
                if vis_count == 4 and min_vis >= early_stop_vis:
                    early_stop = True

            if early_stop:
                break

        if not early_stop:
            # ── Post-Tier-1 refinement ──
            if best_result is not None:
                # Compute best Tier 1 visibility to decide whether to skip Tier 2
                kps = best_result["keypoints"]
                vis_values = {kp["name"]: kp["visibility"] for kp in kps}
                best_vis_count = sum(1 for v in vis_values.values() if v >= _VIS_THRESH_SWEEP)
                best_min_vis = min(vis_values.values()) if vis_values else 0.0

                if best_vis_count == 4 and best_min_vis >= _TIER2_SKIP_VIS:
                    # ── Tier 1 found 4/4 corners with "good enough" visibility ──
                    # min_vis is below early_stop_vis but above the practical threshold
                    # for reliable perspective warp. Tier 2 refine rarely helps here
                    # because the corners are already well-located.
                    _log(f"  >>> TIER 2 SKIP: 4/4 corners with min_vis={best_min_vis:.3f} "
                          f">= {_TIER2_SKIP_VIS:.2f} (good enough for warp)")
                else:
                    # ── Tier 2: Refine the best tier-1 candidates ──
                    tier1_results.sort(key=lambda t: t[3], reverse=True)
                    candidates = tier1_results[:max_refine_candidates]

                    _log(f"  --- Tier 2: refine top {len(candidates)} candidate(s) ---")
                    for ex, ey, t1_result, t1_score in candidates:
                        for re in refine_expands:
                            refined_box = _keypoints_to_bbox(
                                t1_result["keypoints"],
                                margin_x=bw * re, margin_y=bh * re,
                            )
                            if refined_box is None:
                                continue
                            refined = _run_pose_on_crop(
                                pose_session, image, refined_box, pose_conf, img_size,
                                expand_ratio_x=re, expand_ratio_y=re,
                            )
                            if refined is None:
                                continue

                            kps = refined["keypoints"]
                            vis_values = {kp["name"]: kp["visibility"] for kp in kps}
                            vis_count = sum(1 for v in vis_values.values()
                                            if v >= _VIS_THRESH_SWEEP)
                            min_vis = min(vis_values.values()) if vis_values else 0.0
                            pose_con = refined["pose_confidence"]
                            score = (vis_count, min_vis, pose_con)

                            ll = vis_values.get("LL", 0)
                            ul = vis_values.get("UL", 0)
                            ur = vis_values.get("UR", 0)
                            lr = vis_values.get("LR", 0)
                            _log(f"  {ex:6.2f} {ey:6.2f} {re:6.2f} {re:6.2f}  "
                                  f"{pose_con:6.3f} {vis_count:4}/4 {min_vis:6.3f}  "
                                  f"{ll:6.3f} {ul:6.3f} {ur:6.3f} {lr:6.3f}")

                            if score > best_score:
                                best_score = score
                                best_result = refined
                                best_combo = (ex, ey, re, re)

                            if vis_count == 4 and min_vis >= early_stop_vis:
                                early_stop = True
                                break
                        if early_stop:
                            break

            # ── Tier 3: Wider search if still not good enough ──
            if not early_stop and (tier3_x or tier3_y):
                _log(f"  --- Tier 3: wider search ---")
                for ex in tier3_x + tier1_x:
                    for ey in tier3_y + tier1_y:
                        # Skip if already tried in tier 1
                        if ex in tier1_x and ey in tier1_y:
                            continue
                        result = _run_pose_on_crop(
                            pose_session, image, box, pose_conf, img_size,
                            expand_ratio_x=ex, expand_ratio_y=ey,
                            crop_limits=crop_limits_list[det_idx],
                        )
                        if result is None:
                            continue

                        kps = result["keypoints"]
                        vis_values = {kp["name"]: kp["visibility"] for kp in kps}
                        vis_count = sum(1 for v in vis_values.values()
                                        if v >= _VIS_THRESH_SWEEP)
                        min_vis = min(vis_values.values()) if vis_values else 0.0
                        pose_con = result["pose_confidence"]
                        score = (vis_count, min_vis, pose_con)

                        ll = vis_values.get("LL", 0)
                        ul = vis_values.get("UL", 0)
                        ur = vis_values.get("UR", 0)
                        lr = vis_values.get("LR", 0)
                        _log(f"  {ex:6.2f} {ey:6.2f} {'--':>6} {'--':>6}  "
                              f"{pose_con:6.3f} {vis_count:4}/4 {min_vis:6.3f}  "
                              f"{ll:6.3f} {ul:6.3f} {ur:6.3f} {lr:6.3f}")

                        if score > best_score:
                            best_score = score
                            best_result = result
                            best_combo = (ex, ey, None, None)

                        if vis_count == 4 and min_vis >= early_stop_vis:
                            early_stop = True
                            break
                    if early_stop:
                        break

                # Also try refine on any new tier-3 candidates that are good
                if not early_stop:
                    # Quick check: did tier 3 improve on tier 1 at all?
                    # If the best result has < 4 visible corners, try refine
                    # on the top tier-3 candidates
                    pass  # tier 3 combos are already without refine;
                          # doing refine here would be very expensive
                          # and is unlikely to help if tier 2 already failed

        # Print winner for this photo
        if best_result is not None:
            ex_b, ey_b, rx_b, ry_b = best_combo
            kps = best_result["keypoints"]
            vis_values = {kp["name"]: kp["visibility"] for kp in kps}
            vis_count = sum(1 for v in vis_values.values() if v >= _VIS_THRESH_SWEEP)
            min_vis = min(vis_values.values())
            refine_str = ("no refine" if rx_b is None
                         else f"refine=({rx_b:.2f},{ry_b:.2f})")
            tier_str = " (early-stop)" if early_stop else ""
            _log(f"  >>> BEST: expand=({ex_b:.2f},{ey_b:.2f})  {refine_str}  "
                  f"vis={vis_count}/4  min_vis={min_vis:.3f}{tier_str}\n")

            best_result["detection"] = det
            best_results.append(best_result)
            best_params.append({
                "expand_x": ex_b, "expand_y": ey_b,
                "refine_x": rx_b, "refine_y": ry_b,
            })
        else:
            _log(f"  >>> NO valid pose result for this photo\n")

    # Dedup the final results
    if detection_session is not None and len(best_results) > 1:
        min_dist = min(orig_w, orig_h) * dedup_min_dist_ratio
        best_results = dedup_pose_results(best_results, min_dist)

    # Summary
    _log(f"\n{'=' * 90}")
    _log("X/Y SWEEP SUMMARY -- best params per photo:")
    _log(f"{'=' * 90}")
    for i, (res, params) in enumerate(zip(best_results, best_params)):
        kps = res["keypoints"]
        vis_values = {kp["name"]: kp["visibility"] for kp in kps}
        vis_count = sum(1 for v in vis_values.values() if v >= _VIS_THRESH_SWEEP)
        refine_str = ("no refine" if params["refine_x"] is None
                      else f"refine=({params['refine_x']:.2f},{params['refine_y']:.2f})")
        _log(f"  Photo #{i+1}:  expand=({params['expand_x']:.2f}, {params['expand_y']:.2f})  "
              f"{refine_str}  "
              f"vis={vis_count}/4  min_vis={min(vis_values.values()):.3f}  "
              f"pose_conf={res['pose_confidence']:.3f}")

    return best_results, best_params


# ---------------------------------------------------------------------------
# CV Corner Refinement -- refine NN keypoints using local edge analysis
# ---------------------------------------------------------------------------

# Corner geometry for orientation-aware refinement.
# Each corner of a photo has two edges meeting at ~90°. The edge pixels
# belonging to each edge lie on a specific side of the corner:
#   - Horizontal edge pixels: lie ABOVE the corner for bottom-row corners
#     (LL, LR), BELOW for top-row corners (UL, UR).
#   - Vertical edge pixels: lie to the RIGHT for left-column corners
#     (LL, UL), to the LEFT for right-column corners (UR, LR).
#
# This table describes, for each corner type, which side each edge group
# should be on relative to the corner position.
_CORNER_ORIENTATION = {
    #  name  h_edge_side  v_edge_side
    "LL": {"h": "above", "v": "right"},   # horizontal edge above, vertical edge right
    "UL": {"h": "below", "v": "right"},   # horizontal edge below, vertical edge right
    "UR": {"h": "below", "v": "left"},    # horizontal edge below, vertical edge left
    "LR": {"h": "above", "v": "left"},    # horizontal edge above, vertical edge left
}

# Corner adjacency: which two other corners share an edge with this one.
# Each corner shares a horizontal edge with one neighbor and a vertical
# edge with the other. The neighbor that shares the horizontal edge provides
# the y-coordinate projection; the neighbor sharing the vertical edge
# provides the x-coordinate projection.
_CORNER_NEIGHBORS = {
    #  name  h_neighbor (shares horizontal edge)  v_neighbor (shares vertical edge)
    "LL": {"h": "LR", "v": "UL"},  # bottom edge shared with LR, left edge shared with UL
    "UL": {"h": "UR", "v": "LL"},  # top edge shared with UR, left edge shared with LL
    "UR": {"h": "UL", "v": "LR"},  # top edge shared with UL, right edge shared with LR
    "LR": {"h": "LL", "v": "UR"},  # bottom edge shared with LL, right edge shared with UR
}

# Minimum visibility for a neighbor corner to be used for projection
_NEIGHBOR_VIS_THRESHOLD = 0.5


def _orientation_filter_edge_pixels(edge_abs_xs, edge_abs_ys, corner_name,
                                     search_cx, search_cy):
    """Filter edge pixels to only those on the expected side of the corner.

    For a photo rectangle, each corner has two edges that meet at ~90°.
    Edge pixels belonging to those edges lie on the interior side:
      - LL: horizontal edge above, vertical edge to the right
      - UL: horizontal edge below, vertical edge to the right
      - UR: horizontal edge below, vertical edge to the left
      - LR: horizontal edge above, vertical edge to the left

    This filtering prevents edges from adjacent photos or background
    clutter from contaminating the line fits.

    Args:
        edge_abs_xs, edge_abs_ys: Absolute pixel coordinates of edge pixels
        corner_name: One of "LL", "UL", "UR", "LR"
        search_cx, search_cy: The search center (projected or NN position)

    Returns:
        h_mask: Boolean mask for edge pixels suitable for the horizontal edge line
        v_mask: Boolean mask for edge pixels suitable for the vertical edge line
    """
    orient = _CORNER_ORIENTATION[corner_name]

    # Horizontal edge pixels: based on y position relative to center
    if orient["h"] == "above":
        h_mask = edge_abs_ys <= search_cy
    else:  # "below"
        h_mask = edge_abs_ys >= search_cy

    # Vertical edge pixels: based on x position relative to center
    if orient["v"] == "left":
        v_mask = edge_abs_xs <= search_cx
    else:  # "right"
        v_mask = edge_abs_xs >= search_cx

    return h_mask, v_mask


def _project_from_neighbors(kps, corner_name, corner_idx):
    """Project a better search center using high-vis neighbor corners.

    For a low-visibility corner, the two adjacent corners share its edges:
      - The neighbor sharing the horizontal edge constrains the y-coordinate
      - The neighbor sharing the vertical edge constrains the x-coordinate

    For example, for LR: LL shares the bottom edge (projects y), and UR
    shares the right edge (projects x). If LL is at (100, 1974) and UR is
    at (750, 100), the projected center for LR is (750, 1974).

    Only uses neighbors with visibility >= _NEIGHBOR_VIS_THRESHOLD.

    Args:
        kps: List of keypoint dicts from one result
        corner_name: "LL", "UL", "UR", or "LR"
        corner_idx: Index of this corner in the kps list

    Returns:
        dict with keys:
          proj_x, proj_y: Projected coordinates (NN position used for
            any axis without a reliable neighbor)
          confidence: 0.0–1.0 based on how many neighbors contributed
          projected_axis: "x", "y", "both", or "none" -- which axes were
            projected from reliable neighbors (not from the NN fallback)
          proj_y_from_h: True if y was projected from h-neighbor
          proj_x_from_v: True if x was projected from v-neighbor
    """
    neighbors = _CORNER_NEIGHBORS[corner_name]
    h_neighbor_name = neighbors["h"]
    v_neighbor_name = neighbors["v"]

    # Build a lookup by name
    kp_by_name = {kp["name"]: kp for kp in kps}

    proj_x = None
    proj_y = None
    proj_y_from_h = False
    proj_x_from_v = False

    # Horizontal-edge neighbor provides y-coordinate projection
    h_kp = kp_by_name.get(h_neighbor_name)
    if h_kp and h_kp["visibility"] >= _NEIGHBOR_VIS_THRESHOLD:
        proj_y = h_kp["y"]
        proj_y_from_h = True

    # Vertical-edge neighbor provides x-coordinate projection
    v_kp = kp_by_name.get(v_neighbor_name)
    if v_kp and v_kp["visibility"] >= _NEIGHBOR_VIS_THRESHOLD:
        proj_x = v_kp["x"]
        proj_x_from_v = True

    if not (proj_y_from_h or proj_x_from_v):
        return {"proj_x": None, "proj_y": None, "confidence": 0.0,
                "projected_axis": "none", "proj_y_from_h": False, "proj_x_from_v": False}

    # If only one neighbor contributed, use the NN position for the missing axis
    nn_kp = kps[corner_idx]
    if proj_x is None:
        proj_x = nn_kp["x"]
    if proj_y is None:
        proj_y = nn_kp["y"]

    contributions = int(proj_y_from_h) + int(proj_x_from_v)
    confidence = contributions / 2.0

    if proj_y_from_h and proj_x_from_v:
        projected_axis = "both"
    elif proj_y_from_h:
        projected_axis = "y"
    else:
        projected_axis = "x"

    return {"proj_x": proj_x, "proj_y": proj_y, "confidence": confidence,
            "projected_axis": projected_axis,
            "proj_y_from_h": proj_y_from_h, "proj_x_from_v": proj_x_from_v}


def _strip_search_corner(grad_x_img, grad_y_img, grad_mag_img,
                         corner_name, proj_axis, proj_val,
                         strip_half_width=15,
                         perpendicular_range=200,
                         edge_threshold=50.0,
                         nn_other_axis=None,
                         box_hint=None):
    """Search for a corner position using 1D strip scans along a projected axis.

    When only one neighbor provides a projection (partial projection), the
    2D window search can fail because the unprojected axis from the NN is
    wrong, pulling the search center into a contaminated region.

    This function does two 1D searches using gradient profiles and peak
    detection (scipy.signal.find_peaks) to find the sharpest, narrowest
    edge -- which corresponds to the photo boundary rather than internal
    photo content:

      1. Scan perpendicular to the projected axis: find the strongest
         narrow peak in the perpendicular gradient profile. A real photo
         boundary produces a sharp, high-prominence spike, while internal
         content produces broader, lower-prominence structure.
      2. Scan parallel to the projected axis at the found position:
         find the corresponding edge to confirm/refine the projected
         coordinate.

    Peak selection strategy: always select the **highest-prominence peak**.
    A real photo boundary produces a sharp, high-prominence spike in the
    gradient profile because it's the strongest brightness transition.
    Internal photo content and inter-photo gaps produce weaker, less
    prominent structure. This makes highest-prominence the most reliable
    selector regardless of corner type.

    Args:
        grad_x_img: Full-image Sobel x gradient (CV_64F)
        grad_y_img: Full-image Sobel y gradient (CV_64F)
        grad_mag_img: Full-image gradient magnitude
        corner_name: "LL", "UL", "UR", or "LR"
        proj_axis: "x" or "y" -- which axis the neighbor projected
        proj_val: The projected coordinate value on that axis
        strip_half_width: Half-width of the strip for 1D search (pixels)
        perpendicular_range: How far to search perpendicular to the projected
            axis from the NN position (pixels)
        edge_threshold: Minimum gradient magnitude for edge pixels
        nn_other_axis: The NN-predicted value on the unprojected axis. Used
            to center the perpendicular search range.
        box_hint: Optional tuple (x1, y1, x2, y2) of the detection bounding
            box. Used to constrain the perpendicular search range so we don't
            pick up edges from adjacent photos.

    Returns:
        (found_x, found_y, confidence) or (None, None, 0.0) if search fails.
        confidence is 0.5 (1D search only) or 1.0 (both axes confirmed).
    """
    from scipy.signal import find_peaks as _find_peaks

    h, w = grad_mag_img.shape
    orient = _CORNER_ORIENTATION[corner_name]
    # - If proj_axis == "x" (neighbor gave us x), we're scanning y for horizontal edge
    #   The horizontal edge side tells us which direction

    if proj_axis == "y":
        # Neighbor projected Y (horizontal edge neighbor).
        # Search 1: horizontal strip at y≈proj_val -> find strongest vertical edge (grad_x)
        # to determine the X coordinate.
        y_center = int(round(proj_val))
        y_lo = max(0, y_center - strip_half_width)
        y_hi = min(h, y_center + strip_half_width + 1)

        if nn_other_axis is not None:
            x_center = int(round(nn_other_axis))
        else:
            x_center = w // 2
        x_lo = max(0, x_center - perpendicular_range)
        x_hi = min(w, x_center + perpendicular_range)

        # Constrain perpendicular search to detection box if available
        if box_hint is not None:
            bx1, by1, bx2, by2 = [int(round(v)) for v in box_hint]
            # Constrain perpendicular search to ~50px around the relevant
            # box boundary. This prevents picking up edges from adjacent
            # photos that happen to be stronger than the target edge.
            if orient["v"] == "left":
                # Right column corner (LR, UR): edge near box x2
                x_lo = max(x_lo, bx2 - 30)
                x_hi = min(x_hi, bx2 + 30)
            else:
                # Left column corner (LL, UL): edge near box x1
                x_lo = max(x_lo, bx1 - 30)
                x_hi = min(x_hi, bx1 + 30)

        if x_lo >= x_hi or y_lo >= y_hi:
            return None, None, 0.0

        strip = grad_x_img[y_lo:y_hi, x_lo:x_hi]
        profile = np.sum(np.abs(strip), axis=0)
        coords = np.arange(x_lo, x_hi)

        if profile.max() < edge_threshold * strip.shape[0] * 0.3:
            return None, None, 0.0

        # Find peaks: narrow edges have high prominence
        min_height = np.median(profile) * 2
        min_prominence = profile.max() * 0.1
        peaks, props = _find_peaks(profile, height=min_height,
                                   distance=5, prominence=min_prominence)

        if len(peaks) == 0:
            return None, None, 0.0

        # Select the highest-prominence peak -- real photo boundaries
        # produce the sharpest, most isolated peaks in the profile.
        prominences = props["prominences"]
        best_idx = peaks[prominences.argmax()]

        found_x = float(coords[best_idx])

        # Search 2: vertical strip at x≈found_x -> find strongest horizontal
        # edge (grad_y) to confirm/refine Y coordinate.
        x_center2 = int(round(found_x))
        x_lo2 = max(0, x_center2 - strip_half_width)
        x_hi2 = min(w, x_center2 + strip_half_width + 1)
        y_lo2 = max(0, y_center - perpendicular_range)
        y_hi2 = min(h, y_center + perpendicular_range)

        if x_lo2 >= x_hi2 or y_lo2 >= y_hi2:
            return found_x, proj_val, 0.5

        strip2 = grad_y_img[y_lo2:y_hi2, x_lo2:x_hi2]
        profile2 = np.sum(np.abs(strip2), axis=1)
        coords2 = np.arange(y_lo2, y_hi2)

        if profile2.max() < edge_threshold * strip2.shape[1] * 0.3:
            return found_x, proj_val, 0.5

        min_height2 = np.median(profile2) * 2
        min_prominence2 = profile2.max() * 0.1
        peaks2, props2 = _find_peaks(profile2, height=min_height2,
                                     distance=5, prominence=min_prominence2)

        if len(peaks2) == 0:
            return found_x, proj_val, 0.5

        # Select highest-prominence peak
        prominences2 = props2["prominences"]
        best_idx2 = peaks2[prominences2.argmax()]

        found_y = float(coords2[best_idx2])
        return found_x, found_y, 1.0

    else:
        # proj_axis == "x"
        # Neighbor projected X (vertical edge neighbor).
        # Search 1: vertical strip at x≈proj_val -> find strongest horizontal edge (grad_y)
        # to determine the Y coordinate.
        x_center = int(round(proj_val))
        x_lo = max(0, x_center - strip_half_width)
        x_hi = min(w, x_center + strip_half_width + 1)

        if nn_other_axis is not None:
            y_center = int(round(nn_other_axis))
        else:
            y_center = h // 2
        y_lo = max(0, y_center - perpendicular_range)
        y_hi = min(h, y_center + perpendicular_range)

        # Constrain perpendicular search to detection box if available
        if box_hint is not None:
            bx1, by1, bx2, by2 = [int(round(v)) for v in box_hint]
            if orient["h"] == "above":
                # Bottom-row corner (LL, LR): edge near box y2
                y_lo = max(y_lo, by2 - 30)
                y_hi = min(y_hi, by2 + 30)
            else:
                # Top-row corner (UL, UR): edge near box y1
                y_lo = max(y_lo, by1 - 30)
                y_hi = min(y_hi, by1 + 30)

        if x_lo >= x_hi or y_lo >= y_hi:
            return None, None, 0.0

        strip = grad_y_img[y_lo:y_hi, x_lo:x_hi]
        profile = np.sum(np.abs(strip), axis=1)
        coords = np.arange(y_lo, y_hi)

        if profile.max() < edge_threshold * strip.shape[1] * 0.3:
            return None, None, 0.0

        min_height = np.median(profile) * 2
        min_prominence = profile.max() * 0.1
        peaks, props = _find_peaks(profile, height=min_height,
                                   distance=5, prominence=min_prominence)

        if len(peaks) == 0:
            return None, None, 0.0

        # Select highest-prominence peak
        prominences = props["prominences"]
        best_idx = peaks[prominences.argmax()]

        found_y = float(coords[best_idx])

        # Search 2: horizontal strip at y≈found_y -> find strongest vertical
        # edge (grad_x) to confirm/refine X coordinate.
        y_center2 = int(round(found_y))
        y_lo2 = max(0, y_center2 - strip_half_width)
        y_hi2 = min(h, y_center2 + strip_half_width + 1)
        x_lo2 = max(0, x_center - perpendicular_range)
        x_hi2 = min(w, x_center + perpendicular_range)

        if x_lo2 >= x_hi2 or y_lo2 >= y_hi2:
            return proj_val, found_y, 0.5

        strip2 = grad_x_img[y_lo2:y_hi2, x_lo2:x_hi2]
        profile2 = np.sum(np.abs(strip2), axis=0)
        coords2 = np.arange(x_lo2, x_hi2)

        if profile2.max() < edge_threshold * strip2.shape[0] * 0.3:
            return proj_val, found_y, 0.5

        min_height2 = np.median(profile2) * 2
        min_prominence2 = profile2.max() * 0.1
        peaks2, props2 = _find_peaks(profile2, height=min_height2,
                                     distance=5, prominence=min_prominence2)

        if len(peaks2) == 0:
            return proj_val, found_y, 0.5

        # Select highest-prominence peak
        prominences2 = props2["prominences"]
        best_idx2 = peaks2[prominences2.argmax()]

        found_x = float(coords2[best_idx2])
        return found_x, found_y, 1.0


def _fit_weighted_line(group_xs, group_ys, group_mags):
    """Fit a weighted least-squares line to a set of points.

    Returns (a, b, c, linearity) where ax + by + c = 0, a²+b²=1,
    and linearity is the eigenvalue ratio (higher = more linear).
    Returns None if fitting fails.
    """
    if len(group_xs) < 3:
        return None

    total_weight = np.sum(group_mags)
    if total_weight < 1e-6:
        return None

    mean_x = np.average(group_xs, weights=group_mags)
    mean_y = np.average(group_ys, weights=group_mags)

    dx = group_xs - mean_x
    dy = group_ys - mean_y

    cov_xx = np.sum(group_mags * dx * dx) / total_weight
    cov_xy = np.sum(group_mags * dx * dy) / total_weight
    cov_yy = np.sum(group_mags * dy * dy) / total_weight

    cov_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Normal to the line (smallest variance direction)
    normal = eigenvectors[:, 0]
    a, b = normal
    c = -(a * mean_x + b * mean_y)

    linearity = np.max(eigenvalues) / max(np.min(eigenvalues), 1e-10)
    return (a, b, c, linearity)


def _intersect_lines(line1, line2):
    """Compute intersection of two lines in ax+by+c=0 form.

    Returns (ix, iy) or None if lines are parallel.
    """
    a1, b1, c1, _ = line1
    a2, b2, c2, _ = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    ix = (b1 * c2 - b2 * c1) / det
    iy = (a2 * c1 - a1 * c2) / det
    return (ix, iy)


def rescue_low_vis_corners(image: Image.Image, results: list,
                            vis_threshold: float = 0.3,
                            min_visible_corners: int = 3,
                            search_radius: int = 40) -> list:
    """Apply CV refinement only to photos with insufficient visible corners.

    For each result that has fewer than ``min_visible_corners`` corners
    above ``vis_threshold`` visibility, run Sobel edge detection + line
    intersection refinement on ALL of that photo's low-visibility corners.
    This recovers corners the neural network couldn't find, using the
    physical edge structure of the photo in the image.

    High-confidence corners (visibility >= the CV refinement vis_threshold
    of 0.7) are left untouched -- they're already accurate.

    This is the standard "rescue" step in the pipeline, always enabled.
    It prevents degenerate crops (e.g., "strip" crops from only 2 visible
    corners) and avoids warp->simple-crop fallback when edges are clearly
    visible but the NN missed the corner position.

    Args:
        image: Source image (RGB PIL Image)
        results: List of result dicts from pipeline/sweep
        vis_threshold: NN visibility below which a corner counts as
            "low". Also the threshold for counting "visible" corners
            to decide whether a photo needs rescue. (default 0.3)
        min_visible_corners: Minimum number of visible corners for a
            photo to be considered OK -- no rescue needed. (default 3)
        search_radius: Half-size of the search region for CV refinement.
            (default 40)

    Returns:
        The same results list with rescued corners updated in-place.
    """
    needs_rescue = []
    for i, res in enumerate(results):
        kps = res.get("keypoints", [])
        visible = sum(1 for kp in kps if kp["visibility"] >= vis_threshold)
        if visible < min_visible_corners:
            needs_rescue.append(i)

    if not needs_rescue:
        return results

    _RESCUE_VIS_THRESHOLD = 0.7  # CV refinement only processes corners below this
    low_vis_names = []
    for i in needs_rescue:
        kps = results[i].get("keypoints", [])
        low = [f"{kp['name']}={kp['visibility']:.2f}"
               for kp in kps if kp["visibility"] < vis_threshold]
        low_vis_names.append(f"Photo #{i+1}: [{', '.join(low)}]")
        _log(f"  Rescue: Photo #{i+1} has only "
             f"{sum(1 for kp in kps if kp['visibility'] >= vis_threshold)}/4 "
             f"visible corners (need {min_visible_corners}), "
             f"applying CV edge refinement")

    # Filter to only the results that need rescue -- avoids processing
    # all photos when only one has low-visibility corners.
    rescue_subset = [results[i] for i in needs_rescue]
    refined = refine_corners_cv(image, rescue_subset,
                                search_radius=search_radius,
                                vis_threshold=_RESCUE_VIS_THRESHOLD)

    # Copy refined results back
    for idx, i in enumerate(needs_rescue):
        results[i] = refined[idx]

    # Log rescue results
    for i in needs_rescue:
        kps = results[i].get("keypoints", [])
        vis_now = {kp["name"]: kp["visibility"] for kp in kps}
        rescued_corners = []
        for kp in kps:
            if "nn_vis" in kp and kp["nn_vis"] < vis_threshold:
                rescued_corners.append(
                    f"{kp['name']}: vis {kp['nn_vis']:.2f}->{kp['visibility']:.2f} "
                    f"pos ({kp['nn_x']:.0f},{kp['nn_y']:.0f})->({kp['x']:.0f},{kp['y']:.0f})")
        if rescued_corners:
            _log(f"  Rescue: Photo #{i+1} recovered: {'; '.join(rescued_corners)}")
        else:
            _log(f"  Rescue: Photo #{i+1} -- CV refinement could not recover any corners")

    return results


def refine_corners_cv(image: Image.Image, results: list,
                       search_radius: int = 40,
                       edge_threshold: float = 50.0,
                       vis_threshold: float = 0.7,
                       max_shift_ratio: float = 0.3,
                       detection_boxes: list = None) -> list:
    """
    Refine corner keypoint positions using classical computer vision edge
    detection and line intersection, with two enhancements over the naive
    approach:

    **Enhancement 1 -- Orientation-aware edge search:**
    Each corner type (LL/UL/UR/LR) has known edge geometry -- the two edges
    meeting at a corner lie on specific sides (e.g., for LR: horizontal edge
    above, vertical edge to the left). Edge pixels are filtered to only those
    on the expected side before line fitting, preventing edges from adjacent
    photos or background clutter from contaminating the fit.

    **Enhancement 2 -- Neighbor-anchored projection:**
    When a corner has very low visibility, its NN-predicted position may be
    far from the true location. The two adjacent corners share edges with
    this corner and can project a much better search center: the neighbor
    sharing the horizontal edge constrains the y-coordinate, and the neighbor
    sharing the vertical edge constrains the x-coordinate. The search is then
    centered on this projected position, allowing the CV edge search to reach
    the true corner even when the NN prediction is far off.

    **Only refines corners with NN visibility below vis_threshold** -- high-
    confidence NN corners are already accurate and don't need CV refinement.

    The refined position is constrained to stay within ``max_shift_ratio`` x
    ``search_radius`` of the NN prediction (when no projection is used) or
    within the search region centered on the projected position (when
    projection is used). This prevents drift to edges from distant photos.

    Args:
        image: Source image (RGB PIL Image)
        results: List of result dicts from pipeline/sweep (modified in-place)
        search_radius: Half-size of the search region around each corner (px).
            Larger values handle corners that are further from the true position,
            but risk finding edges from adjacent photos.
        edge_threshold: Minimum gradient magnitude to count as an edge pixel.
        vis_threshold: Only refine corners with NN visibility below this value.
            High-confidence corners (>= this) are left unchanged.
        max_shift_ratio: Maximum shift from NN position as a fraction of
            search_radius. E.g. 0.3 with radius=40 means the refined position
            can be at most 12px from the NN prediction (without projection).

    Returns:
        The same results list with refined keypoint positions.
        ``visibility`` is boosted for successfully refined corners, and
        original NN position and visibility are saved in ``nn_x``, 
        ``nn_y``, and ``nn_vis`` fields.
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Sobel gradients for edge detection
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_angle = np.arctan2(grad_y, grad_x)

    h, w = gray.shape

    # Two-pass refinement. Pass 1 refines corners that have reliable
    # neighbors. Their boosted visibility makes them available as neighbors
    # in pass 2 -- allowing corners that initially had no reliable neighbors
    # to benefit from projection (e.g., UR corner after LR is refined).
    for pass_num in range(2):
        for res in results:
            kps = res.get("keypoints", [])
            if not kps:
                continue

            # Extract detection box for this result (for strip search bounding)
            det_box = res.get("detection", {}).get("box", None)
            box_hint = None
            if det_box is not None:
                box_hint = (det_box["x1"], det_box["y1"], det_box["x2"], det_box[
"y2"])
            elif detection_boxes is not None:
                # Fallback: use detection_boxes list by result index
                ridx = results.index(res) if res in results else 0
                if ridx < len(detection_boxes) and detection_boxes[ridx] is not None:
                    box_hint = detection_boxes[ridx]

            for kp_idx, kp in enumerate(kps):
                # Skip corners that are already high-vis or already refined
                if "nn_x" in kp:
                    continue  # already refined in a previous iteration
                if kp["visibility"] >= vis_threshold:
                    continue

                cx, cy = kp["x"], kp["y"]
                vis = kp["visibility"]
                corner_name = kp.get("name", "")

                # --- Enhancement 2: Neighbor-anchored projection ---
                proj = _project_from_neighbors(kps, corner_name, kp_idx)
                proj_x = proj["proj_x"]
                proj_y = proj["proj_y"]
                proj_conf = proj["confidence"]
                projected_axis = proj["projected_axis"]

                use_projection = proj_conf >= 0.5
                if use_projection:
                    search_cx = proj_x
                    search_cy = proj_y
                else:
                    search_cx = cx
                    search_cy = cy

                # --- Strip search: preferred method when partial projection is available ---
                # When one neighbor provides a reliable projection, the strip search
                # is more robust than the 2D window search. It constrains one axis
                # FIRST (from the reliable neighbor), then searches for the other axis
                # via 1D peak detection. The 2D search, by contrast, can find wrong
                # line intersections from adjacent photos in its 2D window.
                #
                # Strategy:
                #   - projected_axis in ("x", "y"): strip search first, 2D as fallback
                #   - projected_axis == "both": 2D first (projection should center well),
                #     strip search as fallback if 2D fails
                #   - projected_axis == "none": 2D only with strict NN-distance constraint
                refined = None
                if projected_axis in ("x", "y"):
                    # Partial projection: strip search is the primary method
                    if projected_axis == "y":
                        proj_axis_val = proj_y
                        nn_other = cx
                    else:
                        proj_axis_val = proj_x
                        nn_other = cy

                    sx, sy, s_conf = _strip_search_corner(
                        grad_x, grad_y, grad_mag,
                        corner_name, projected_axis, proj_axis_val,
                        strip_half_width=15,
                        perpendicular_range=200,
                        edge_threshold=edge_threshold,
                        nn_other_axis=nn_other,
                        box_hint=box_hint)

                    if sx is not None and sy is not None and s_conf >= 0.5:
                        if 0 <= sx < w and 0 <= sy < h:
                            refined = (sx, sy, s_conf)

                    # Fallback: try 2D search if strip search failed
                    if refined is None:
                        refined = _refine_corner_2d(
                            grad_mag, grad_angle, grad_x, grad_y,
                            corner_name, search_cx, search_cy, cx, cy,
                            h, w, search_radius, edge_threshold, max_shift_ratio,
                            use_projection)

                elif projected_axis == "both":
                    # Full projection: 2D search should work well
                    refined = _refine_corner_2d(
                        grad_mag, grad_angle, grad_x, grad_y,
                        corner_name, search_cx, search_cy, cx, cy,
                        h, w, search_radius, edge_threshold, max_shift_ratio,
                        use_projection)
                else:
                    # No projection: 2D search with strict NN constraint
                    refined = _refine_corner_2d(
                        grad_mag, grad_angle, grad_x, grad_y,
                        corner_name, cx, cy, cx, cy,
                        h, w, search_radius, edge_threshold, max_shift_ratio,
                        False)

                if refined is None:
                    continue

                ix, iy, _ = refined

                # Update the keypoint in-place (modifies kps list directly)
                kp["nn_x"] = kp["x"]
                kp["nn_y"] = kp["y"]
                # Save original NN visibility before boosting -- this tells us
                # how confident the pose model was, regardless of CV refinement.
                # Used by adaptive margin and warp fallback to decide whether
                # to trust a corner for perspective warp.
                if "nn_vis" not in kp:
                    kp["nn_vis"] = kp["visibility"]
                if use_projection:
                    kp["proj_x"] = float(proj_x)
                    kp["proj_y"] = float(proj_y)
                kp["x"] = float(ix)
                kp["y"] = float(iy)
                kp["visibility"] = max(vis, 0.5)

            # After each pass, recompute center and dedup_priority
            visible_xs = [kp["x"] for kp in kps if kp["visibility"] >= _VIS_THRESH_DEDUP]
            visible_ys = [kp["y"] for kp in kps if kp["visibility"] >= _VIS_THRESH_DEDUP]
            if len(visible_xs) >= 2:
                res["center"] = (float(np.mean(visible_xs)), float(np.mean(visible_ys)))
                vis_count = len(visible_xs)
                pose_con = res.get("pose_confidence", 0)
                res["dedup_priority"] = pose_con * (0.5 if vis_count < 3 else 1.0)

    return results


def _refine_corner_2d(grad_mag, grad_angle, grad_x_img, grad_y_img,
                      corner_name, search_cx, search_cy,
                      nn_cx, nn_cy, img_h, img_w,
                      search_radius, edge_threshold, max_shift_ratio,
                      use_projection):
    """2D window search with orientation-aware edge line intersection.

    Searches in a square window centered on (search_cx, search_cy),
    filters edge pixels by orientation, fits two lines, and returns
    their intersection.

    Returns (ix, iy, confidence) on success, or None if refinement fails.
    """
    x1 = max(0, int(search_cx) - search_radius)
    y1 = max(0, int(search_cy) - search_radius)
    x2 = min(img_w, int(search_cx) + search_radius)
    y2 = min(img_h, int(search_cy) + search_radius)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    # Extract gradient data in the search region
    local_mag = grad_mag[y1:y2, x1:x2]
    local_angle = grad_angle[y1:y2, x1:x2]

    edge_mask = local_mag > edge_threshold
    edge_ys, edge_xs = np.where(edge_mask)

    if len(edge_xs) < 10:
        return None

    edge_angles = local_angle[edge_mask]
    edge_abs_xs = edge_xs + x1
    edge_abs_ys = edge_ys + y1

    # --- Orientation-aware edge filtering ---
    h_spatial, v_spatial = _orientation_filter_edge_pixels(
        edge_abs_xs, edge_abs_ys, corner_name, search_cx, search_cy)

    edge_dir = edge_angles + np.pi / 2
    dir_mod = edge_dir % np.pi
    h_angle = (dir_mod < np.pi / 4) | (dir_mod > 3 * np.pi / 4)
    v_angle = ~h_angle

    horizontal_mask = h_angle & h_spatial
    vertical_mask = v_angle & v_spatial

    lines = []
    for group_mask in [horizontal_mask, vertical_mask]:
        group_xs = edge_abs_xs[group_mask]
        group_ys = edge_abs_ys[group_mask]
        group_mags = local_mag[edge_mask][group_mask]
        line = _fit_weighted_line(group_xs, group_ys, group_mags)
        if line is not None:
            lines.append(line)

    # Fallback: angle-only filtering if orientation-aware was too aggressive
    if len(lines) < 2:
        fallback_lines = []
        for group_mask in [h_angle, v_angle]:
            group_xs = edge_abs_xs[group_mask]
            group_ys = edge_abs_ys[group_mask]
            group_mags = local_mag[edge_mask][group_mask]
            line = _fit_weighted_line(group_xs, group_ys, group_mags)
            if line is not None:
                fallback_lines.append(line)
        if len(fallback_lines) >= 2:
            lines = fallback_lines

    if len(lines) < 2:
        return None

    # Sort by linearity, take best two
    lines.sort(key=lambda l: l[3], reverse=True)
    best_two = lines[:2]

    result = _intersect_lines(best_two[0], best_two[1])
    if result is None:
        return None

    ix, iy = result

    # Validate
    if ix < 0 or iy < 0 or ix >= img_w or iy >= img_h:
        return None
    if (abs(ix - search_cx) > search_radius or
            abs(iy - search_cy) > search_radius):
        return None
    if not use_projection:
        max_shift = search_radius * max_shift_ratio
        if abs(ix - nn_cx) > max_shift or abs(iy - nn_cy) > max_shift:
            return None

    return (ix, iy, 1.0)


# ---------------------------------------------------------------------------
# Coordinate output helpers
# ---------------------------------------------------------------------------

# Canonical corner order for output: LL, UL, UR, LR
_CORNER_ORDER = ["LL", "UL", "UR", "LR"]


def format_coords(results: list, fmt: str = "json") -> str:
    """Format photo corner coordinates for output.

    Args:
        results: List of result dicts from pipeline/infer.
        fmt: Output format -- "json" or "text".

    Returns:
        Formatted string ready for stdout.
    """
    if fmt == "json":
        import json
        photos = []
        for res in results:
            kps = {kp["name"]: kp for kp in res.get("keypoints", [])}
            corners = []
            for name in _CORNER_ORDER:
                kp = kps.get(name)
                if kp is not None:
                    corners.append([round(kp["x"], 1), round(kp["y"], 1)])
                else:
                    corners.append(None)
            photos.append(corners)
        return json.dumps(photos)

    elif fmt == "text":
        lines = []
        for i, res in enumerate(results):
            kps = {kp["name"]: kp for kp in res.get("keypoints", [])}
            for name in _CORNER_ORDER:
                kp = kps.get(name)
                if kp is not None:
                    lines.append(f"{i + 1} {name} {kp['x']:.1f} {kp['y']:.1f} {kp['visibility']:.3f}")
                else:
                    lines.append(f"{i + 1} {name} - - 0.000")
        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown coords format: {fmt!r} (expected 'json' or 'text')")


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

        # Draw corner quadrilateral: LL->UL->UR->LR->LL
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
            direction (pixels, pre-converted from fraction of diagonal
            when called via the CLI). Positive values add padding.

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
            Note: When using the CLI, --crop-margin is a fraction of
            the photo diagonal (resolution-independent); this function
            receives the pre-converted pixel value.

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
        # Not enough reliable corners for a mask -- return opaque crop
        return cropped.convert("RGBA")

    # Keypoint coords relative to the crop origin
    offset_x, offset_y = ix1, iy1
    # Order for polygon: UL -> UR -> LR -> LL (clockwise from top-left)
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
                          margin: float = 0,
                          adaptive_margins: dict = None) -> Image.Image:
    """
    Perspective warp crop: use the 4 detected corner keypoints to
    compute a homography that dewarps the photo into a rectangular
    output image.

    Uses cv2.warpPerspective with INTER_LANCZOS4 (8x8 Lanczos) for
    the highest-quality interpolation available in OpenCV.

    Warp modes:
      inward  -- Output rectangle dimensions from the average of opposite
                edge lengths. This "inscribes" the rect in the detected
                quadrilateral, which may lose small slivers of content
                at the edges but produces a clean crop. (Default)
      outward -- Output rectangle dimensions from the maximum of opposite
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
            surrounding content to the output. Note: When using the
            CLI, --crop-margin is a fraction of the photo diagonal
            (resolution-independent); this function receives the
            pre-converted pixel value.

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
    # Expand source corners outward by margin pixels along edges.
    # When adaptive_margins is provided, each corner gets its own margin
    # based on keypoint visibility (lower visibility -> more margin).
    # Otherwise, uniform margin is applied to all corners.
    if margin > 0 or adaptive_margins:
        center = src_pts.mean(axis=0)
        for i in range(4):
            name = corner_names[i]
            # Determine this corner's margin
            if adaptive_margins and name in adaptive_margins:
                m = adaptive_margins[name]
            else:
                m = margin
            if m > 0:
                direction = src_pts[i] - center
                dist = np.linalg.norm(direction)
                if dist > 0:
                    src_pts[i] += (m / dist) * direction

    # Compute output dimensions from detected edge lengths
    w_top = np.linalg.norm(src_pts[1] - src_pts[0])
    w_bot = np.linalg.norm(src_pts[3] - src_pts[2])
    h_left = np.linalg.norm(src_pts[3] - src_pts[0])
    h_right = np.linalg.norm(src_pts[2] - src_pts[1])

    if warp_mode == "outward":
        # Use max of opposite edges -> circumscribes the detected quad
        # Guarantees no photo content is lost; may include small filler
        # areas in the corners of the output rectangle.
        out_w = int(round(max(w_top, w_bot)))
        out_h = int(round(max(h_left, h_right)))
    else:
        # Inward (default): average of opposite edges -> inscribes the rect
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

    # Convert PIL image to cv2 format (RGB -> BGR)
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
        # Convert BGRA -> RGBA for PIL
        warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(warped_rgba)
    else:
        # Apply warp with specified border fill color
        border_bgr = (border_fill[2], border_fill[1], border_fill[0])  # RGB -> BGR
        warped = cv2.warpPerspective(
            img_cv,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_bgr,
        )
        # Convert back to PIL (BGR -> RGB)
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
               margin: float = 0,
               adaptive_margin: bool = False,
               adaptive_margin_thresh: float = 0.5,
               adaptive_margin_max: float = 0.03,
               warp_fallback_thresh: float = 0.3):
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
        margin: Margins expressed as a fraction of the detected photo's
                diagonal. E.g. 0.02 = 2% of diagonal. For a 1000px
                diagonal photo, that's ~20px. Converted to absolute
                pixels internally based on each photo's detection bbox.
        adaptive_margin: If True, expand crop margin for low-confidence
                   corners. The lower a corner's visibility, the more
                   margin is added (pushed outward from quad center).
        adaptive_margin_thresh: Visibility threshold below which corners
                   start receiving extra margin (default 0.5). A corner
                   at visibility 0 gets adaptive_margin_max extra fraction.
        adaptive_margin_max: Maximum extra margin as a fraction of the
                   photo diagonal for a corner with visibility 0
                   (default 0.03 = 3%). Margin scales linearly:
                   extra_frac = max * (1 - vis/thresh) for vis < thresh.
        warp_fallback_thresh: If any corner has visibility below this
                   threshold, fall back to simple crop instead of warp.
                   Emits a warning to stderr. Set to 0 to disable
                   fallback (default 0.3).
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

        kps = res.get("keypoints", [])
        kp_by_name = {kp["name"]: kp for kp in kps} if kps else {}

        # Compute photo diagonal from detection bbox for resolution-independent
        # margin conversion.  diagonal = sqrt(w² + h²) gives a natural scale
        # that works across different image resolutions.
        box = res["detection"]["box"]
        bw = box["x2"] - box["x1"]
        bh = box["y2"] - box["y1"]
        diag = (bw ** 2 + bh ** 2) ** 0.5

        # Convert fraction-of-diagonal margins to absolute pixels
        px_margin = margin * diag

        # Compute adaptive margins for low-confidence corners (in pixels)
        # Uses original NN visibility (nn_vis) when available -- the CV
        # refinement boosts visibility artificially, but the original
        # confidence tells us whether we should trust the corner.
        adaptive_margins = None
        if adaptive_margin and kps:
            adaptive_margins = {}
            for name in ["LL", "UL", "UR", "LR"]:
                kp = kp_by_name.get(name, {})
                # Use original NN visibility if CV refinement saved it,
                # otherwise fall back to the current (possibly boosted) value
                vis = kp.get("nn_vis", kp.get("visibility", 0))
                if vis < adaptive_margin_thresh:
                    # Linearly scale: vis=0 -> max fraction, vis=thresh -> 0
                    extra_frac = adaptive_margin_max * (1 - vis / adaptive_margin_thresh)
                    adaptive_margins[name] = px_margin + extra_frac * diag
                else:
                    adaptive_margins[name] = px_margin

        # Check for warp fallback: if any corner has very low CURRENT
        # visibility, fall back to simple crop -- a perspective warp with
        # uncertain corners can produce badly distorted results.
        # Note: we check current visibility, not nn_vis, because corner
        # refinement can recover invisible corners to vis=1.0. Those
        # recovered positions are reliable enough for warping.
        #
        # Before falling back, attempt CV edge rescue on just this photo.
        # If rescue recovers the low-vis corners above the threshold,
        # we can still do a proper warp crop instead of a degraded simple crop.
        do_warp = crop_mode.startswith("warp")
        if do_warp and warp_fallback_thresh > 0 and kps:
            low_vis_corners = []
            for kp in kps:
                vis = kp["visibility"]
                if vis < warp_fallback_thresh:
                    low_vis_corners.append(f"{kp['name']}={vis:.2f}")
            if low_vis_corners:
                # Try CV rescue before giving up on warp
                rescued = rescue_low_vis_corners(
                    image, [res],
                    vis_threshold=warp_fallback_thresh,
                    min_visible_corners=4,
                )
                # Check if rescue fixed the problem
                kps = rescued[0].get("keypoints", [])
                kp_by_name = {kp["name"]: kp for kp in kps} if kps else {}
                still_low = [f"{kp['name']}={kp['visibility']:.2f}"
                             for kp in kps
                             if kp["visibility"] < warp_fallback_thresh]
                if still_low:
                    _log(f"    Photo #{photo_id}: WARNING low-confidence corners "
                          f"[{', '.join(still_low)}] after rescue, "
                          f"falling back to simple crop")
                    do_warp = False
                else:
                    _log(f"    Photo #{photo_id}: CV rescue recovered corners -- proceeding with warp")
                    # Update res with rescued keypoints for the warp
                    res["keypoints"] = kps
                    res["center"] = rescued[0].get("center", res.get("center"))

        # Determine the effective crop tag for the filename
        if do_warp:
            crop_tag = "warp"
        else:
            if crop_mode.startswith("warp"):
                crop_tag = "crop"      # warp was requested but fell back
            elif crop_mode == "simple-corners":
                crop_tag = "crop"
            else:
                crop_tag = "box"

        out_name = f"{stem}_{crop_tag}_{photo_id}{ext}"
        out_path = out_dir / out_name

        if do_warp:
            wm = "outward" if crop_mode == "warp-stretch" else "inward"
            cropped = crop_perspective_warp(
                image, res, warp_mode=wm,
                border_fill=border_fill, transparent=transparent,
                margin=px_margin,
                adaptive_margins=adaptive_margins,
            )
            if cropped is None:
                # Fallback to simple crop if warp fails (insufficient keypoints)
                _log(f"    Photo #{photo_id}: warp failed (insufficient keypoints), falling back to simple crop")
                cropped = crop_simple(image, res, transparent=transparent, margin=px_margin)
        else:
            use_corners = (crop_mode == "simple-corners")
            effective_margin = px_margin
            if adaptive_margins:
                # For simple crops, use the max adaptive margin (uniform expansion)
                effective_margin = max(adaptive_margins.values())
            cropped = crop_simple(image, res, transparent=transparent,
                                  use_corners=use_corners, margin=effective_margin)

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
        dim_info = f"{cropped.size[0]}x{cropped.size[1]}"
        mode_info = cropped.mode
        _log(f"    Photo #{photo_id} -> {out_path}  "
              f"({dim_info} {mode_info}  "
              f"box=({box['x1']:.0f},{box['y1']:.0f})->({box['x2']:.0f},{box['y2']:.0f})  "
              f"visible_corners={vis_corners}/4)")


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def infer_single(detection_session, pose_session, image_path: str,
                 output_path: str = None, det_conf: float = 0.5,
                 pose_conf: float = 0.5, iou_threshold: float = 0.45,
                 pose_crop_expand: float = POSE_CROP_EXPAND,
                 pose_refine: bool = False,
                 pose_refine_expand: float = POSE_REFINE_EXPAND,
                 img_size: int = DEFAULT_IMG_SIZE,
                 dedup_min_dist_ratio: float = DEDUP_MIN_DIST_RATIO,
                 crop_mode: str = None, crop_dir: str = None,
                 transparent: bool = False,
                 border_fill: tuple = (114, 114, 114),
                 margin: float = 0,
                 do_pose_sweep: bool = False,
                 sweep_crop_expands: list = None,
                 sweep_refine_expands: list = None,
                 do_pose_sweep_xy: bool = False,
                 sweep_xy_expands: list = None,
                 center_bias: bool = False,
                 cv_refine: bool = False,
                 cv_refine_radius: int = 40,
                 coords: str = None,
                 debug: bool = False,
                 no_image: bool = False,
                 adaptive_margin: bool = False,
                 adaptive_margin_thresh: float = 0.5,
                 adaptive_margin_max: float = 0.03,
                 warp_fallback_thresh: float = 0.3,
                 corner_refine: bool = False,
                 corner_refine_iterations: int = 2,
                 corner_refine_conf: float = 0.3,
                 corner_refine_model: str = "regression",
                 corner_regression_session=None):
    """Run the full pipeline on a single image."""

    # Open image and capture EXIF before converting (convert creates
    # a new image that loses the .info dict including exif)
    _raw_image = Image.open(image_path)
    _source_exif = _raw_image.info.get("exif")
    image = _raw_image.convert("RGB")
    image.load()  # Force pixel data into memory for thread-safe cropping
    orig_w, orig_h = image.size

    sweep_params = None  # filled in if sweep is used

    if do_pose_sweep:
        results, sweep_params = pose_sweep(
            detection_session, pose_session, image,
            det_conf, pose_conf, iou_threshold,
            sweep_crop_expands, sweep_refine_expands,
            img_size, dedup_min_dist_ratio,
        )
    elif do_pose_sweep_xy:
        xy_ex = sweep_xy_expands or list(SWEEP_XY_EXPANDS)
        xy_re = sweep_refine_expands or list(SWEEP_REFINE_EXPANDS)
        results, sweep_params = pose_sweep_xy(
            detection_session, pose_session, image,
            det_conf, pose_conf, iou_threshold,
            x_expands=xy_ex, y_expands=xy_ex,
            refine_expands=xy_re,
            img_size=img_size,
            dedup_min_dist_ratio=dedup_min_dist_ratio,
        )
    else:
        results = pipeline(
            detection_session, pose_session, image,
            det_conf, pose_conf, iou_threshold, pose_crop_expand,
            pose_refine, pose_refine_expand,
            img_size, dedup_min_dist_ratio,
            center_bias=center_bias,
            cv_refine=cv_refine,
            cv_refine_radius=cv_refine_radius,
        )

    # ── Post-sweep CV refinement ──
    # When sweep is used, cv_refine doesn't run inside pipeline() --
    # apply it here on the sweep results instead.
    if (do_pose_sweep_xy or do_pose_sweep) and cv_refine:
        # Optimization: If all corners are already confident, skip
        _CV_SKIP_VIS = 0.5
        all_corners_good = all(
            kp.get("visibility", 0) >= _CV_SKIP_VIS
            for res in results
            for kp in res.get("keypoints", [])
        )
        if all_corners_good:
            _log(f"  Skipping cv-refine: all corners vis >= {_CV_SKIP_VIS:.2f} after sweep")
        else:
            results = refine_corners_cv(image, results,
                                         search_radius=cv_refine_radius)
            _log(f"  Post-sweep cv-refine applied")

    # ── Post-sweep rescue of low-visibility corners ──
    # Always on -- same as the rescue in pipeline(), but applied here
    # for sweep results that bypass pipeline().
    if do_pose_sweep_xy or do_pose_sweep:
        results = rescue_low_vis_corners(image, results)

    # ── Corner refinement ──
    # After pose (+ optional CV refinement), refine corner positions using
    # the corner regression model: crop around each approximate corner,
    # run the regression model, and pick the detection closest to the
    # expected position.
    if corner_refine and results:
        if corner_regression_session is None:
            _log("  Warning: corner regression model not loaded, "
                 "skipping corner refinement")
        else:
            _log(f"  Corner refinement: {corner_refine_iterations} iteration(s) "
                 f"(model: regression)")
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(refine_corners_regression, image, res,
                                corner_regression_session=corner_regression_session,
                                iterations=corner_refine_iterations,
                                conf_threshold=corner_refine_conf)
                    for res in results
                ]
                for f in futures:
                    f.result()  # Propagate any exceptions
            _log(f"  Corner refinement complete")

    # Print results
    mode = "2-stage" if detection_session else "1-stage (pose only)"
    if do_pose_sweep:
        mode += " + sweep"
    elif do_pose_sweep_xy:
        mode += " + sweep-xy"
    elif pose_refine:
        mode += " + refine"
    if center_bias:
        mode += " + center-bias"
    if cv_refine:
        mode += " + cv-refine"
    if corner_refine:
        mode += " + corner-refine"
    _log(f"Mode: {mode}")
    _log(f"\n{'=' * 60}")
    _log(f"Image: {image_path}")
    _log(f"Size: {orig_w}x{orig_h}")
    _log(f"Mode: {mode}")
    _log(f"Photos found: {len(results)}")
    if sweep_params:
        _log(f"Best params per photo:")
        for i, sp in enumerate(sweep_params):
            if "expand_x" in sp:
                # X/Y sweep format
                rx_str = f"{sp['refine_x']:.2f}" if sp["refine_x"] is not None else "--"
                ry_str = f"{sp['refine_y']:.2f}" if sp["refine_y"] is not None else "--"
                refine_info = f"refine=({rx_str},{ry_str})" if sp["refine_x"] is not None else "no refine"
                _log(f"  Photo #{i+1}:  expand=({sp['expand_x']:.2f},{sp['expand_y']:.2f})  {refine_info}")
            else:
                # Uniform sweep format
                re_str = f"refine={sp['refine_expand']:.2f}" if sp["refine_expand"] is not None else "no refine"
                _log(f"  Photo #{i+1}:  crop_expand={sp['crop_expand']:.2f}  {re_str}")
    _log(f"{'=' * 60}")

    for i, res in enumerate(results):
        box = res["detection"]["box"]
        det_c = res["detection"]["confidence"]
        pose_c = res.get("pose_confidence", 0)
        kps = res.get("keypoints", [])

        line = f"\n  Photo #{i+1}:  detection_conf={det_c:.4f}  pose_conf={pose_c:.4f}"
        if sweep_params:
            sp = sweep_params[i] if i < len(sweep_params) else None
            if sp:
                if "expand_x" in sp:
                    rx_str = f"{sp['refine_x']:.2f}" if sp["refine_x"] is not None else "--"
                    ry_str = f"{sp['refine_y']:.2f}" if sp["refine_y"] is not None else "--"
                    refine_info = f"refine=({rx_str},{ry_str})" if sp["refine_x"] is not None else "no refine"
                    line += f"  [expand=({sp['expand_x']:.2f},{sp['expand_y']:.2f}) {refine_info}]"
                else:
                    re_str = f"refine={sp['refine_expand']:.2f}" if sp["refine_expand"] is not None else "no refine"
                    line += f"  [{sp['crop_expand']:.2f} / {re_str}]"
        _log(line)
        _log(f"    Box: ({box['x1']:.1f}, {box['y1']:.1f}) -> ({box['x2']:.1f}, {box['y2']:.1f})")
        if kps:
            for kp in kps:
                full = KEYPOINT_FULL_NAMES[kp["name"]]
                _log(f"    {kp['name']} ({full}): "
                      f"({kp['x']:.1f}, {kp['y']:.1f})  vis={kp['visibility']:.3f}")
        else:
            _log("    No pose detection in this crop")

    # Output coordinates if requested
    if coords:
        print(format_coords(results, fmt=coords))

    # Draw and save annotated image (only with --debug)
    if debug:
        vis = image.copy()
        draw_results(vis, results)

        if output_path is None:
            base = Path(image_path).stem
            parent = Path(image_path).parent
            output_path = str(parent / f"{base}_detected.jpg")
        else:
            out = Path(output_path)
            # If --output is a directory (no extension), place the annotated
            # image inside it with the default naming convention.
            if out.suffix == "" or out.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                output_path = str(out / f"{Path(image_path).stem}_detected.jpg")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        vis.save(output_path, quality=95)
        _log(f"\n  Saved: {output_path}")

    # Save crops if requested (skip with --no-image)
    if crop_mode and not no_image:
        _log(f"\n  Cropping ({crop_mode}):")
        wm = "outward" if crop_mode == "warp-stretch" else ("inward" if crop_mode.startswith("warp") else None)
        save_crops(image, results, image_path, crop_mode, crop_dir,
                   transparent=transparent, warp_mode=wm,
                   border_fill=border_fill,
                   source_exif=_source_exif,
                   margin=margin,
                   adaptive_margin=adaptive_margin,
                   adaptive_margin_thresh=adaptive_margin_thresh,
                   adaptive_margin_max=adaptive_margin_max,
                   warp_fallback_thresh=warp_fallback_thresh)

    return results


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def infer_batch(detection_session, pose_session, image_dir: str,
               output_dir: str = None, det_conf: float = 0.5,
               pose_conf: float = 0.5, iou_threshold: float = 0.45,
               pose_crop_expand: float = POSE_CROP_EXPAND,
               pose_refine: bool = False,
               pose_refine_expand: float = POSE_REFINE_EXPAND,
               img_size: int = DEFAULT_IMG_SIZE,
               dedup_min_dist_ratio: float = DEDUP_MIN_DIST_RATIO,
               limit: int = 0,
               crop_mode: str = None, crop_dir: str = None,
               transparent: bool = False,
               border_fill: tuple = (114, 114, 114),
               margin: float = 0,
               do_pose_sweep: bool = False,
               sweep_crop_expands: list = None,
               sweep_refine_expands: list = None,
               do_pose_sweep_xy: bool = False,
               sweep_xy_expands: list = None,
               center_bias: bool = False,
               cv_refine: bool = False,
               cv_refine_radius: int = 40,
               coords: str = None,
               debug: bool = False,
               no_image: bool = False,
               adaptive_margin: bool = False,
               adaptive_margin_thresh: float = 0.5,
               adaptive_margin_max: float = 0.03,
               warp_fallback_thresh: float = 0.3,
               corner_refine: bool = False,
               corner_refine_iterations: int = 2,
               corner_refine_conf: float = 0.3,
               corner_refine_model: str = "regression",
               corner_regression_session=None):

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
        out_path = image_dir / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    _log(f"Processing {len(images)} images")
    _log(f"  Input:  {image_dir}")
    _log(f"  Output: {out_path}")

    summary = []
    for img_path in images:
        out_file = out_path / f"{img_path.stem}_detected.jpg"
        results = infer_single(
            detection_session, pose_session, str(img_path), str(out_file),
            det_conf, pose_conf, iou_threshold, pose_crop_expand,
            pose_refine, pose_refine_expand,
            img_size, dedup_min_dist_ratio,
            crop_mode, crop_dir,
            transparent, border_fill,
            margin,
            do_pose_sweep=do_pose_sweep,
            sweep_crop_expands=sweep_crop_expands,
            sweep_refine_expands=sweep_refine_expands,
            do_pose_sweep_xy=do_pose_sweep_xy,
            sweep_xy_expands=sweep_xy_expands,
            center_bias=center_bias,
            cv_refine=cv_refine,
            cv_refine_radius=cv_refine_radius,
            coords=coords,
            debug=debug,
            no_image=no_image,
            adaptive_margin=adaptive_margin,
            adaptive_margin_thresh=adaptive_margin_thresh,
            adaptive_margin_max=adaptive_margin_max,
            warp_fallback_thresh=warp_fallback_thresh,
            corner_refine=corner_refine,
            corner_refine_iterations=corner_refine_iterations,
            corner_refine_conf=corner_refine_conf,
            corner_refine_model=corner_refine_model,
            corner_regression_session=corner_regression_session,
        )
        summary.append({
            "image": img_path.name,
            "photos": len(results),
            "with_pose": pose_ok,
        })

    # Summary
    _log(f"\n{'=' * 60}")
    _log("BATCH SUMMARY")
    _log(f"{'=' * 60}")
    _log(f"{'Image':<30} {'Photos':>8} {'W/Pose':>8}")
    _log(f"{'-' * 30} {'-' * 8} {'-' * 8}")
    for e in summary:
        _log(f"{e['image']:<30} {e['photos']:>8} {e['with_pose']:>8}")
    total_photos = sum(e["photos"] for e in summary)
    total_pose = sum(e["with_pose"] for e in summary)
    _log(f"{'-' * 30} {'-' * 8} {'-' * 8}")
    _log(f"{'Total':<30} {total_photos:>8} {total_pose:>8}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Presets -- detection/refinement tiers (crop method is separate)
# ---------------------------------------------------------------------------
# Presets control ONLY how corners are detected and refined.
# The --crop argument controls the output crop method independently.
# Valid combinations are checked by _validate_preset_crop() which
# warns (not errors) on questionable pairings.

_PRESETS = {
    "quick": {
        "description": "Fast detection + pose, no refinement",
        "args": {},
    },
    "standard": {
        "description": "Pose-refine + adaptive margin (good balance of speed and accuracy)",
        "args": {
            "pose_refine": True,
            "adaptive_margin": True,
        },
    },
    "thorough": {
        "description": "Full refinement: pose-refine -> corner-refine (regression) -> cv-refine -> adaptive margin "
                       "(best quality, recovers invisible corners)",
        "args": {
            "pose_refine": True,
            "corner_refine": True,
            "corner_refine_iterations": 2,
            "corner_refine_conf": 0.3,
            "corner_refine_model": "regression",
            "cv_refine": True,
            "adaptive_margin": True,
        },
    },
}

# Crop defaults that pair well with each preset (used when --crop is
# specified alongside a preset but individual crop settings like
# --crop-margin are left at defaults).
_PRESET_CROP_DEFAULTS = {
    "quick": {
        "crop_margin": 0.02,
    },
    "standard": {
        "crop_margin": 0.02,
    },
    "thorough": {
        "crop_margin": 0.02,
        "border_fill": "white",
    },
}


def _validate_preset_crop(preset_name: str, crop_mode: str | None):
    """Check for questionable preset + crop method combinations.

    Returns (errors, warnings) where:
      errors:   list of error messages (combine is invalid, should abort)
      warnings: list of warning messages (combine is questionable but allowed)
    """
    errors = []
    warnings = []

    if preset_name and crop_mode:
        # thorough + simple: refinement effort wasted on a loose crop
        if preset_name == "thorough" and crop_mode == "simple":
            warnings.append(
                f"Preset 'thorough' with --crop simple: thorough refinement "
                f"is wasted on a detection-bbox crop that doesn't use corner "
                f"positions. Consider --crop simple-corners or --crop warp-stretch."
            )
        # quick + warp variants: imprecise corners + perspective warp
        if preset_name == "quick" and crop_mode.startswith("warp"):
            warnings.append(
                f"Preset 'quick' with --crop {crop_mode}: no corner refinement, "
                f"so warp accuracy depends on raw pose keypoints. Consider "
                f"--preset standard or --preset thorough for better corners."
            )

    if crop_mode and preset_name is None:
        # --crop specified with no preset -- valid use case
        pass

    if preset_name and crop_mode is None:
        # Preset specified but no --crop: bad usage -- no output will be produced
        errors.append(
            f"Preset '{preset_name}' requires --crop. "
            f"Specify a crop method: --crop simple-corners or --crop warp-stretch."
        )

    return errors, warnings


def _apply_preset(parser, args):
    """Apply a named preset, then let explicit CLI args override it.

    Parses sys.argv to find which args the user explicitly set, applies
    the preset defaults, then re-applies the user's explicit args on top.
    This lets presets provide sensible defaults while still allowing any
    individual arg to be overridden.

    Returns the modified args namespace.
    """
    preset_name = getattr(args, "preset", None)
    if not preset_name:
        return args

    preset = _PRESETS.get(preset_name)
    if preset is None:
        parser.error(f"Unknown preset: {preset_name!r} "
                     f"(choose from: {', '.join(_PRESETS.keys())})")

    # Determine which args the user explicitly set on the CLI
    # by parsing with a dummy namespace that tracks what was provided
    user_defaults = {}
    for action in parser._actions:
        if action.dest != "preset" and action.dest != "help":
            user_defaults[action.dest] = action.default

    # Re-parse with same defaults to find what the user explicitly set
    # We check each arg: if it differs from the parser default, the user set it
    user_explicit = {}
    for action in parser._actions:
        if action.dest in ("preset", "help",):
            continue
        cli_val = getattr(args, action.dest)
        default_val = action.default
        if cli_val != default_val:
            user_explicit[action.dest] = cli_val

    # Apply preset values (detection/refinement only, no crop settings)
    preset_args = preset["args"]
    for key, val in preset_args.items():
        # Only apply preset value if the user didn't explicitly override it
        if key not in user_explicit:
            setattr(args, key, val)

    # Apply preset-crop defaults when --crop is used alongside a preset.
    # These provide sensible margins/fills for the crop method without
    # hardcoding them in the detection-only preset.
    crop_mode = getattr(args, "crop", None)
    crop_defaults = _PRESET_CROP_DEFAULTS.get(preset_name, {})
    if crop_mode and crop_defaults:
        for key, val in crop_defaults.items():
            if key not in user_explicit:
                setattr(args, key, val)

    # Validate preset + crop combination
    errors, warnings = _validate_preset_crop(preset_name, crop_mode)
    for w in warnings:
        _log(f"Warning: {w}")
    for e in errors:
        parser.error(e)

    return args


def main():
    parser = argparse.ArgumentParser(
        prog="photocrop",
        description="photocrop -- Detect & Extract Photos from Multi-Photo Scans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Basic Usage:
  # Single image (models auto-detected)
  photocrop --image scan.jpg

  # Folder of images -> folder of results
  photocrop --image ./scans/ --output ./crops/

Presets + Crop Method:
  Presets control detection/refinement SPEED and ACCURACY only.
  The --crop argument controls the OUTPUT CROP METHOD separately.
  Use them together for the best results.

  # Quick -- detect + pose, no refinement (~5s)
  photocrop --image scan.jpg --preset quick --crop simple-corners

  # Standard -- pose-refine + adaptive margin (~7s)
  photocrop --image scan.jpg --preset standard --crop simple-corners
  photocrop --image scan.jpg --preset standard --crop warp-stretch --border-fill white

  # Thorough -- full pipeline, best corner accuracy (~15s)
  photocrop --image scan.jpg --preset thorough --crop warp-stretch --border-fill white

  # Quick + warp -- works but corners may be less accurate (warning issued)
  photocrop --image scan.jpg --preset quick --crop warp-stretch

  # Override preset defaults
  photocrop --image scan.jpg --preset thorough --crop warp-stretch --crop-margin 0.03

  # Detection only -- preset without --crop (no files saved, warning issued)
  photocrop --image scan.jpg --preset standard

Recommended for photos close together:
  When photos are close together on the scanner bed, the pose model can
  latch onto an adjacent photo if the crop is too large. --pose-sweep-xy
  tries multiple crop-expand sizes per photo and picks the best, preventing
  this issue. It adds ~3x processing time but significantly improves
  corner detection on tight layouts:

  # Add sweep to any preset + crop combination
  photocrop --image scan.jpg --preset standard --crop warp-stretch --pose-sweep-xy

  # Thorough + sweep for the most challenging layouts
  photocrop --image scan.jpg --preset thorough --crop warp-stretch --pose-sweep-xy

CV Refinement (manual):
  # Edge detection + line intersection (Enhancements 1+2)
  photocrop --image scan.jpg --cv-refine

  # With perspective warp cropping
  photocrop --image scan.jpg --cv-refine \\
    --crop warp-stretch --crop-margin 0.02 --border-fill white

Detailed Crop Commands:
  # Best simple crop -- keypoint bbox + margin so edges aren't clipped
  photocrop --image scan.jpg --crop simple-corners --crop-margin 0.02

  # Best warp crop -- outward warp + margin + white fill for clean edges
  photocrop --image scan.jpg --crop warp-stretch \\
    --crop-margin 0.02 --border-fill white

  # Best transparent crop -- corner-based with margin, for compositing
  photocrop --image scan.jpg --crop simple-corners \\
    --crop-margin 0.02 --crop-transparent

  # Crop a whole folder
  photocrop --image ./scans/ --output ./crops/ \\
    --crop simple-corners --crop-margin 0.02

Coordinates Only (scripting-friendly):
  # Output corner coordinates as JSON, no file output
  photocrop --image scan.jpg --coords json --no-image

  # Coordinates as line-delimited text (bash-friendly)
  photocrop --image scan.jpg --coords text --no-image

  # Debug: save annotated detection image with boxes and keypoints
  photocrop --image scan.jpg --preset standard --crop warp-stretch --debug

  # Coordinates + debug image
  photocrop --image scan.jpg --coords json --debug
""",
    )

    parser.add_argument(
        "--detection-model", "-d", type=str, default=str(DEFAULT_DETECTION_MODEL),
        help="Path to detection ONNX model (default: ../models/detection_ep47.onnx)",
    )
    parser.add_argument(
        "--pose-model", "-p", type=str, default=str(DEFAULT_POSE_MODEL),
        help="Path to pose ONNX model (default: ../models/pose_single_ep42.onnx)",
    )
    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="Path to image file or directory of images to process",
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        choices=list(_PRESETS.keys()),
        help="Detection/refinement preset. Controls how corners are found, "
             "NOT the output crop method (use --crop for that). "
             "quick=detect+pose only, standard=+pose-refine+adaptive-margin, "
             "thorough=+pose-refine+corner-refine+cv-refine+adaptive-margin. "
             "Note: low-visibility corner rescue (CV edge detection) is always "
             "on -- it runs automatically when corners are invisible. "
             "Always pair with --crop to save output. "
             "(default: none)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for annotated image(s). For a single input image, "
             "defaults to {stem}_detected.jpg next to the input. For a "
             "directory of images, defaults to an 'output' subdirectory.",
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
        "--pose-crop-expand", type=float, default=POSE_CROP_EXPAND,
        help="Proportion of the larger detection box dimension to expand "
             "the crop by before passing to the pose model. A larger value "
             "gives the pose model more surrounding context, helping it find "
             "corners near the edges. (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--pose-refine", action="store_true",
        help="Run a second pose pass: after the first pose finds corners, "
             "derive a tighter bounding box from those keypoints and re-run "
             "pose with a smaller crop. This can improve accuracy when the "
             "first pass has low-visibility corners due to a loose detection "
             "box or misalignment.",
    )
    parser.add_argument(
        "--pose-refine-expand", type=float, default=POSE_REFINE_EXPAND,
        help="Proportion to expand the refined crop by (from keypoint bbox) "
             "before the second pose pass. Only used with --pose-refine. "
             "(default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--pose-sweep", action="store_true",
        help="Search for the best pose-crop-expand and pose-refine-expand "
             "values for each detected photo. Tries a grid of values, scores "
             "each by corner visibility (min visibility across 4 corners, "
             "then total visible corners), and picks the best per-photo. "
             "Overrides --pose-crop-expand, --pose-refine, and "
             "--pose-refine-expand.",
    )
    parser.add_argument(
        "--sweep-crop-expands", type=str, default=None,
        help="Comma-separated list of crop-expand values to try in sweep "
             "(default: 0.05,0.10,0.15,0.20)",
    )
    parser.add_argument(
        "--sweep-refine-expands", type=str, default=None,
        help="Comma-separated list of refine-expand values to try in sweep "
             "(default: 0.03,0.05,0.10,0.15)",
    )
    parser.add_argument(
        "--pose-sweep-xy", action="store_true",
        help="Search for the best per-axis (X/E-W and Y/N-S) expansion "
             "values for each detected photo. Tries a grid of "
             "(expand_x, expand_y) pairs with optional refine, scoring "
             "by corner visibility. Overrides --pose-sweep, "
             "--pose-crop-expand, and --pose-refine.",
    )
    parser.add_argument(
        "--sweep-xy-expands", type=str, default=None,
        help="Comma-separated list of per-axis expand values to try in "
             "X/Y sweep (applied to both X and Y axes). "
             "(default: 0.05,0.10,0.15,0.20,0.25)",
    )
    parser.add_argument(
        "--center-bias", action="store_true",
        help="Bias crop expansion toward the image center instead of "
             "expanding symmetrically. For photos near the edge of a "
             "multi-photo scan, this shifts more expansion budget toward "
             "the center where there's useful context and less toward "
             "the image edge where there's just background.",
    )
    parser.add_argument(
        "--cv-refine", action="store_true",
        help="Refine corner positions using classical computer vision "
             "(edge detection + line intersection). After the neural "
             "network finds approximate corner positions, this step "
             "finds the two dominant edge lines near each corner and "
             "computes their intersection as the refined position. "
             "Especially useful for low-visibility corners where the "
             "NN estimate is approximate but edges are still visible.",
    )
    parser.add_argument(
        "--cv-refine-radius", type=int, default=40,
        help="Search radius (pixels) around each NN-predicted corner "
             "for CV refinement. Larger values handle corners that are "
             "further from the true position. (default: 40)",
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
        "--limit", "-n", type=int, default=0,
        help="When processing a directory, limit the number of images "
             "to process (0 = all)",
    )
    parser.add_argument(
        "--crop", type=str, default=None,
        choices=["simple", "simple-corners", "warp", "warp-stretch"],
        help="Crop detected photos: "
             "'simple' (detection bbox crop, no transforms), "
             "'simple-corners' (tighter bbox from detected corner keypoints, "
             "more accurate), "
             "'warp' (perspective warp, inward -- average edge dims), or "
             "'warp-stretch' (perspective warp, outward -- max edge dims, "
             "preserves all photo content). "
             "Output saved as {stem}_{tag}_{N}.{ext} where tag is "
             "'warp', 'crop', or 'box' based on actual crop mode used.",
    )
    parser.add_argument(
        "--crop-dir", type=str, default=None,
        help="Output directory for cropped photos (default: 'crops' subdirectory "
             "next to input image)",
    )
    parser.add_argument(
        "--crop-margin", type=float, default=0,
        help="Margin as a fraction of the detected photo's diagonal. E.g. "
             "0.02 adds ~2%% of the diagonal (~20px on a 1000px photo). "
             "For simple crops, expands the bounding box. For warp crops, "
             "pushes corner keypoints outward from the quad center. "
             "Resolution-independent -- works across different image sizes. "
             "(default: 0)",
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
    parser.add_argument(
        "--coords", type=str, default=None,
        choices=["json", "text"],
        help="Output corner coordinates to stdout. 'json' prints a JSON array "
             "[[[x1,y1],[x2,y2],...], ...] with corners in LL,UL,UR,LR order. "
             "'text' prints one corner per line: 'PHOTO LL x y vis'. "
             "Useful for scripting -- combine with --no-image to skip "
             "file output and get coordinates only.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save the annotated detection image (boxes, keypoints, corners "
             "drawn on the original). Not saved by default.",
    )
    parser.add_argument(
        "--no-image", action="store_true",
        help="Don't save cropped photo files. Useful with --coords when you "
             "only need coordinates without extracting image files.",
    )
    parser.add_argument(
        "--adaptive-margin", action="store_true",
        help="Expand crop margin for low-confidence corners. The lower a "
             "corner's visibility score, the more margin is added outward from "
             "the quadrilateral center, ensuring content near uncertain corners "
             "is included. Useful for photos with partially-detected edges.",
    )
    parser.add_argument(
        "--adaptive-margin-thresh", type=float, default=0.5,
        help="Visibility threshold below which corners receive extra margin "
             "(default: 0.5). A corner at visibility 0 gets "
             "--adaptive-margin-max extra fraction; one at the threshold gets 0.",
    )
    parser.add_argument(
        "--adaptive-margin-max", type=float, default=0.03,
        help="Maximum extra margin as a fraction of the photo diagonal for a "
             "corner with visibility 0 (default: 0.03 = 3%% of diagonal). "
             "Scales linearly between 0 (at threshold) and this fraction "
             "(at visibility 0). Resolution-independent across image sizes.",
    )
    parser.add_argument(
        "--warp-fallback-thresh", type=float, default=0.3,
        help="If any corner has visibility below this threshold, fall back "
             "from perspective warp to simple crop and emit a warning. "
             "Set to 0 to disable warp fallback (default: 0.3).",
    )
    parser.add_argument(
        "--corner-refine", action="store_true", default=False,
        help="Enable corner refinement after pose detection. Crops around "
             "each approximate corner and runs pose or detection model to "
             "find a more precise position. Pose model uses named keypoints; "
             "detection model uses bbox corners with geometric classification. "
             "Crop size is auto-computed from the photo's bbox. Default: disabled.",
    )
    parser.add_argument(
        "--corner-refine-iterations", type=int, default=2,
        help="Number of corner refinement iterations. Each iteration re-crops "
             "around the detected corner position for higher precision. "
             "(default: 2)",
    )
    parser.add_argument(
        "--corner-refine-conf", type=float, default=0.3,
        help="Confidence threshold for corner regression model. "
             "(default: 0.3, lower than pose model since corner regression "
             "works on tight crops where the corner is expected to be present)",
    )
    parser.add_argument(
        "--corner-refine-model", type=str, default="regression",
        choices=["regression"],
        help="Model to use for corner refinement. Only 'regression' is supported -- "
             "the corner regression model detects tight bounding boxes around corner "
             "points in 320x320 crops, then picks the detection closest to the "
             "expected position based on the pose model's approximate corner.",
    )
    parser.add_argument(
        "--corner-regression-model", type=str, default=str(DEFAULT_CORNER_REGRESSION_MODEL),
        help="Path to corner regression ONNX model "
             "(default: ../models/corner-regression-v2.onnx)",
    )

    args = parser.parse_args()

    # Apply preset (lets individual CLI args override preset values)
    if args.preset:
        args = _apply_preset(parser, args)

    # Validate models
    detection_model_path = Path(args.detection_model).resolve()
    pose_model_path = Path(args.pose_model).resolve()

    if not detection_model_path.exists():
        _log(f"Error: Detection model not found: {detection_model_path}")
        _log(f"  (default: {DEFAULT_DETECTION_MODEL.resolve()})")
        sys.exit(1)
    if not pose_model_path.exists():
        _log(f"Error: Pose model not found: {pose_model_path}")
        _log(f"  (default: {DEFAULT_POSE_MODEL.resolve()})")
        sys.exit(1)

    # Load detection model
    _log(f"Loading detection model: {detection_model_path}")
    detection_session = load_onnx_model(str(detection_model_path))

    # Load pose model
    _log(f"Loading pose model: {pose_model_path}")
    pose_session = load_onnx_model(str(pose_model_path))

    # Load corner regression model (only if needed for corner refinement)
    corner_regression_session = None
    if args.corner_refine:
        corner_regression_model_path = Path(args.corner_regression_model).resolve()
        if corner_regression_model_path.exists():
            _log(f"Loading corner regression model: {corner_regression_model_path}")
            corner_regression_session = load_onnx_model(str(corner_regression_model_path))
        else:
            _log(f"Error: Corner regression model not found: {corner_regression_model_path}")
            _log(f"  (default: {DEFAULT_CORNER_REGRESSION_MODEL.resolve()})")
            sys.exit(1)

    # Validate image path and auto-detect file vs directory mode
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image path not found: {args.image}")
        sys.exit(1)

    is_dir = image_path.is_dir()

    # Parse border fill color
    border_fill = parse_border_fill(args.border_fill)

    # Parse sweep parameters
    do_sweep = args.pose_sweep
    do_sweep_xy = args.pose_sweep_xy
    if args.sweep_crop_expands:
        sweep_ce = [float(x.strip()) for x in args.sweep_crop_expands.split(",")]
    else:
        sweep_ce = list(SWEEP_CROP_EXPANDS)
    if args.sweep_refine_expands:
        sweep_re = [float(x.strip()) for x in args.sweep_refine_expands.split(",")]
    else:
        sweep_re = list(SWEEP_REFINE_EXPANDS)
    if args.sweep_xy_expands:
        sweep_xy = [float(x.strip()) for x in args.sweep_xy_expands.split(",")]
    else:
        sweep_xy = list(SWEEP_XY_EXPANDS)

    # Run inference -- auto-detect file vs directory
    if is_dir:
        infer_batch(
            detection_session, pose_session, args.image, args.output,
            det_conf=args.det_conf,
            pose_conf=args.pose_conf,
            iou_threshold=args.iou,
            pose_crop_expand=args.pose_crop_expand,
            pose_refine=args.pose_refine,
            pose_refine_expand=args.pose_refine_expand,
            img_size=args.imgsz,
            dedup_min_dist_ratio=args.dedup_dist,
            limit=args.limit,
            crop_mode=args.crop,
            crop_dir=args.crop_dir,
            transparent=args.crop_transparent,
            border_fill=border_fill,
            margin=args.crop_margin,
            do_pose_sweep=do_sweep,
            sweep_crop_expands=sweep_ce,
            sweep_refine_expands=sweep_re,
            do_pose_sweep_xy=do_sweep_xy,
            sweep_xy_expands=sweep_xy,
            center_bias=args.center_bias,
            cv_refine=args.cv_refine,
            cv_refine_radius=args.cv_refine_radius,
            coords=args.coords,
            debug=args.debug,
            no_image=args.no_image,
            adaptive_margin=args.adaptive_margin,
            adaptive_margin_thresh=args.adaptive_margin_thresh,
            adaptive_margin_max=args.adaptive_margin_max,
            warp_fallback_thresh=args.warp_fallback_thresh,
            corner_refine=args.corner_refine,
            corner_refine_iterations=args.corner_refine_iterations,
            corner_refine_conf=args.corner_refine_conf,
            corner_refine_model=args.corner_refine_model,
            corner_regression_session=corner_regression_session,
        )
    else:
        infer_single(
            detection_session, pose_session, args.image, args.output,
            det_conf=args.det_conf,
            pose_conf=args.pose_conf,
            iou_threshold=args.iou,
            pose_crop_expand=args.pose_crop_expand,
            pose_refine=args.pose_refine,
            pose_refine_expand=args.pose_refine_expand,
            img_size=args.imgsz,
            dedup_min_dist_ratio=args.dedup_dist,
            crop_mode=args.crop,
            crop_dir=args.crop_dir,
            transparent=args.crop_transparent,
            border_fill=border_fill,
            margin=args.crop_margin,
            do_pose_sweep=do_sweep,
            sweep_crop_expands=sweep_ce,
            sweep_refine_expands=sweep_re,
            do_pose_sweep_xy=do_sweep_xy,
            sweep_xy_expands=sweep_xy,
            center_bias=args.center_bias,
            cv_refine=args.cv_refine,
            cv_refine_radius=args.cv_refine_radius,
            coords=args.coords,
            debug=args.debug,
            no_image=args.no_image,
            adaptive_margin=args.adaptive_margin,
            adaptive_margin_thresh=args.adaptive_margin_thresh,
            adaptive_margin_max=args.adaptive_margin_max,
            warp_fallback_thresh=args.warp_fallback_thresh,
            corner_refine=args.corner_refine,
            corner_refine_iterations=args.corner_refine_iterations,
            corner_refine_conf=args.corner_refine_conf,
            corner_refine_model=args.corner_refine_model,
            corner_regression_session=corner_regression_session,
        )


if __name__ == "__main__":
    main()