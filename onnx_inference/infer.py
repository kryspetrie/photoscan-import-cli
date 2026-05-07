#!/usr/bin/env python3
"""
Photo Pose Detector — Two/Three-Stage ONNX Inference CLI
=========================================================

Two-stage pipeline for detecting photo corners in multi-photo images:

  Stage 1 (Detection):  Detect photo bounding boxes in the full image
  Stage 2 (Pose):       Crop each detected box → detect 4 corner keypoints
                        Map keypoints back to original image coordinates
  Stage 3 (Dedup):      Greedy deduplication by keypoint-center proximity

Optional Stage 2b (Refine): --pose-refine
  After the first pose pass, derive a tighter bounding box from the detected
  keypoints and re-run the pose model with a smaller crop. This helps when
  the detection box is loose or misaligned — the second pass gets a crop that's
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

Recommended Commands
-------------------
    # Best simple crop — keypoint-based bbox + 10px margin so edges aren't clipped
    python3 infer.py --image scan.jpg --crop simple-corners --crop-margin 10

    # Best warp crop — outward warp + margin + white fill for clean edges
    python3 infer.py --image scan.jpg --crop warp-stretch \
                     --crop-margin 10 --border-fill white

    # Best transparent crop — corner-based with margin, for compositing
    python3 infer.py --image scan.jpg --crop simple-corners \
                     --crop-margin 10 --crop-transparent

    # Crop a whole folder of images
    python3 infer.py --image ./scans/ --output ./crops/ \
                     --crop simple-corners --crop-margin 10

    # Best accuracy — 3-stage refine for improved corner detection
    python3 infer.py --image scan.jpg --pose-refine \
                     --crop warp-stretch --crop-margin 10 --border-fill white

Usage
-----
    # Single image (models auto-detected)
    python3 infer.py --image scan.jpg

    # Folder of images
    python3 infer.py --image ./scans/ --output ./crops/

    # Override model paths
    python3 infer.py --detection-model /path/to/det.onnx \
                     --pose-model /path/to/pose.onnx \
                     --image scan.jpg

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
POSE_CROP_EXPAND = 0.15    # Expand detection box by 15% of larger dim for pose crop
POSE_REFINE_EXPAND = 0.05  # After first pose, re-crop from keypoints + 5% for refine pass
SWEEP_CROP_EXPANDS = [0.05, 0.10, 0.15, 0.20]   # values to try for --pose-sweep
SWEEP_REFINE_EXPANDS = [0.03, 0.05, 0.10, 0.15]  # values to try for --pose-sweep refine
DEFAULT_IMG_SIZE = 640
DEDUP_MIN_DIST_RATIO = 0.12  # 12% of image min-dimension
_VIS_THRESH_DEDUP = 0.25      # visibility threshold for center-based dedup
_VIS_THRESH_SWEEP = 0.30      # visibility threshold for sweep scoring

# Default model paths (relative to this script's location)
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DETECTION_MODEL = _SCRIPT_DIR / ".." / "models" / "detection_model.onnx"
DEFAULT_POSE_MODEL = _SCRIPT_DIR / ".." / "models" / "pose_model_v2.onnx"


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

def _run_pose_on_crop(pose_session, image: Image.Image, box: dict,
                     pose_conf: float, img_size: int,
                     expand_ratio: float) -> dict | None:
    """
    Run the pose model on a single detection box, expanding the crop
    by ``expand_ratio`` of the box's larger dimension before cropping.

    Returns a result dict with keypoints mapped to original image coords,
    or None if no pose detection above threshold.
    """
    orig_w, orig_h = image.size
    x1, y1 = box["x1"], box["y1"]
    x2, y2 = box["x2"], box["y2"]

    box_w = x2 - x1
    box_h = y2 - y1
    expand_px = int(max(box_w, box_h) * expand_ratio)

    crop_x1 = max(0, int(x1) - expand_px)
    crop_y1 = max(0, int(y1) - expand_px)
    crop_x2 = min(orig_w, int(x2) + expand_px)
    crop_y2 = min(orig_h, int(y2) + expand_px)

    crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    pose_dets = run_pose(pose_session, crop, pose_conf, img_size)
    if not pose_dets:
        return None

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


def _keypoints_to_bbox(keypoints: list, margin: float = 0) -> dict:
    """
    Derive a bounding box from detected corner keypoints.

    If at least 2 visible corners are available, uses their positions.
    Falls back to the detection bbox corners for invisible keypoints.
    Adds ``margin`` pixels on each side.
    """
    kp_by_name = {kp["name"]: kp for kp in keypoints}
    corners = {}
    for name in ["LL", "UL", "UR", "LR"]:
        kp = kp_by_name.get(name)
        if kp and kp["visibility"] >= _VIS_THRESH_DEDUP:
            corners[name] = (kp["x"], kp["y"])
        # Invisible corners are omitted — the bbox is computed from
        # whichever corners we have

    if len(corners) < 2:
        return None

    xs = [c[0] for c in corners.values()]
    ys = [c[1] for c in corners.values()]
    x1 = min(xs) - margin
    y1 = min(ys) - margin
    x2 = max(xs) + margin
    y2 = max(ys) + margin

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def pipeline(detection_session, pose_session, image: Image.Image,
            det_conf: float = 0.5, pose_conf: float = 0.5,
            iou_threshold: float = 0.45,
            pose_crop_expand: float = POSE_CROP_EXPAND,
            pose_refine: bool = False,
            pose_refine_expand: float = POSE_REFINE_EXPAND,
            img_size: int = DEFAULT_IMG_SIZE,
            dedup_min_dist_ratio: float = DEDUP_MIN_DIST_RATIO):
    """
    Two- or three-stage pipeline: detect → pose → [refine →] dedup.

    Two-stage (default):
        Detection model finds bounding boxes → pose model finds corners.

    Three-stage refine (``pose_refine=True``):
        Detection → pose (coarse, with expanded crop) → derive a tighter
        bounding box from the detected keypoints → pose again (refined,
        with a smaller crop centered on the actual photo). This helps
        when the first pose pass has low-visibility corners because the
        detection box is too loose or misaligned — the second pass gets
        a crop that's tightly centered on the actual photo.

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

    # Stage 2: Find corners for each detected photo
    pose_results = []
    for det in detections:
        result = _run_pose_on_crop(
            pose_session, image, det["box"],
            pose_conf, img_size, pose_crop_expand,
        )
        if result is None:
            continue

        # Stage 2b (optional): Refine by re-running pose on a tighter crop
        # derived from the first pass keypoints.
        if pose_refine:
            refined_box = _keypoints_to_bbox(result["keypoints"])
            if refined_box is not None:
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

    return pose_results


# ---------------------------------------------------------------------------
# Parameter sweep — search for best pose-crop-expand / pose-refine-expand
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

    # Build list of (crop_expand, refine_expand_or_None) combos
    combos = []
    for ce in crop_expands:
        combos.append((ce, None))  # two-stage (no refine)
        for re in refine_expands:
            combos.append((ce, re))  # three-stage refine

    # For each detection, run all combos and find the best
    best_results = []
    best_params = []

    print(f"\n{'=' * 80}")
    print(f"POSE PARAMETER SWEEP  —  {len(detections)} detection(s), "
          f"{len(combos)} combo(s) per detection")
    print(f"{'=' * 80}")
    print(f"  crop_expands:   {crop_expands}")
    print(f"  refine_expands: {refine_expands}")
    print(f"  (also trying 2-stage [no refine] for each crop_expand)\n")

    for det_idx, det in enumerate(detections):
        box = det["box"]
        print(f"  Photo #{det_idx+1}:  box=({box['x1']:.0f},{box['y1']:.0f})"
              f"→({box['x2']:.0f},{box['y2']:.0f})  det_conf={det['confidence']:.3f}")
        print(f"  {'crop_exp':>9}  {'refn_exp':>9}  "
              f"{'pose_conf':>9}  "
              f"{'vis_corners':>11}  "
              f"{'min_vis':>7}  "
              f"{'LL':>6}  {'UL':>6}  {'UR':>6}  {'LR':>6}")
        print(f"  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*11}  {'-'*7}  "
              f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

        best_score = None
        best_result = None
        best_combo = None

        for ce, re in combos:
            # Run pose with this crop_expand
            result = _run_pose_on_crop(
                pose_session, image, box, pose_conf, img_size, ce
            )
            if result is None:
                label = f"{'—':>6}" if re is None else f"{re:.2f}"
                print(f"  {ce:9.2f}  {label:>9}  {'—':>9}  {'—':>11}  {'—':>7}  "
                      f"{'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}")
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

            re_label = f"{re:.2f}" if re is not None else "  —"
            ll = vis_values.get("LL", 0)
            ul = vis_values.get("UL", 0)
            ur = vis_values.get("UR", 0)
            lr = vis_values.get("LR", 0)
            print(f"  {ce:9.2f}  {re_label:>9}  {pose_con:9.3f}  "
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
            print(f"  >>> BEST: crop_expand={ce_best:.2f}  {re_str}  "
                  f"vis={vis_count}/4  min_vis={min_vis:.3f}\n")

            best_result["detection"] = det
            best_results.append(best_result)
            best_params.append({
                "crop_expand": ce_best,
                "refine_expand": re_best,
            })
        else:
            print(f"  >>> NO valid pose result for this photo\n")

    # Dedup the final results
    if detection_session is not None and len(best_results) > 1:
        min_dist = min(orig_w, orig_h) * dedup_min_dist_ratio
        best_results = dedup_pose_results(best_results, min_dist)

    # Summary
    print(f"\n{'=' * 80}")
    print("SWEEP SUMMARY — best params per photo:")
    print(f"{'=' * 80}")
    for i, (res, params) in enumerate(zip(best_results, best_params)):
        kps = res["keypoints"]
        vis_values = {kp["name"]: kp["visibility"] for kp in kps}
        vis_count = sum(1 for v in vis_values.values() if v >= _VIS_THRESH_SWEEP)
        re_str = (f"refine={params['refine_expand']:.2f}"
                  if params["refine_expand"] is not None else "no refine")
        print(f"  Photo #{i+1}:  crop_expand={params['crop_expand']:.2f}  {re_str}  "
              f"vis={vis_count}/4  min_vis={min(vis_values.values()):.3f}  "
              f"pose_conf={res['pose_confidence']:.3f}")

    return best_results, best_params

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
                 sweep_refine_expands: list = None):
    """Run the full pipeline on a single image."""

    # Open image and capture EXIF before converting (convert creates
    # a new image that loses the .info dict including exif)
    _raw_image = Image.open(image_path)
    _source_exif = _raw_image.info.get("exif")
    image = _raw_image.convert("RGB")
    orig_w, orig_h = image.size

    sweep_params = None  # filled in if sweep is used

    if do_pose_sweep:
        results, sweep_params = pose_sweep(
            detection_session, pose_session, image,
            det_conf, pose_conf, iou_threshold,
            sweep_crop_expands, sweep_refine_expands,
            img_size, dedup_min_dist_ratio,
        )
    else:
        results = pipeline(
            detection_session, pose_session, image,
            det_conf, pose_conf, iou_threshold, pose_crop_expand,
            pose_refine, pose_refine_expand,
            img_size, dedup_min_dist_ratio,
        )

    # Print results
    mode = "2-stage" if detection_session else "1-stage (pose only)"
    if do_pose_sweep:
        mode += " + sweep"
    elif pose_refine:
        mode += " + refine"
    print(f"Mode: {mode}")
    print(f"\n{'=' * 60}")
    print(f"Image: {image_path}")
    print(f"Size: {orig_w}×{orig_h}")
    print(f"Mode: {mode}")
    print(f"Photos found: {len(results)}")
    if sweep_params:
        print(f"Best params per photo:")
        for i, sp in enumerate(sweep_params):
            re_str = f"refine={sp['refine_expand']:.2f}" if sp["refine_expand"] is not None else "no refine"
            print(f"  Photo #{i+1}:  crop_expand={sp['crop_expand']:.2f}  {re_str}")
    print(f"{'=' * 60}")

    for i, res in enumerate(results):
        box = res["detection"]["box"]
        det_c = res["detection"]["confidence"]
        pose_c = res.get("pose_confidence", 0)
        kps = res.get("keypoints", [])

        line = f"\n  Photo #{i+1}:  detection_conf={det_c:.4f}  pose_conf={pose_c:.4f}"
        if sweep_params:
            sp = sweep_params[i] if i < len(sweep_params) else None
            if sp:
                re_str = f"refine={sp['refine_expand']:.2f}" if sp["refine_expand"] is not None else "no refine"
                line += f"  [{sp['crop_expand']:.2f} / {re_str}]"
        print(line)
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
               sweep_refine_expands: list = None):
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
        out_path = image_dir / "output"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(images)} images")
    print(f"  Input:  {image_dir}")
    print(f"  Output: {out_path}")

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
Basic Usage:
  # Single image (models auto-detected)
  python3 infer.py --image scan.jpg

  # Folder of images → folder of results
  python3 infer.py --image ./scans/ --output ./crops/

Recommended Crop Commands:
  # Best simple crop — keypoint bbox + margin so edges aren't clipped
  python3 infer.py --image scan.jpg --crop simple-corners --crop-margin 10

  # Best warp crop — outward warp + margin + white fill for clean edges
  python3 infer.py --image scan.jpg --crop warp-stretch \\
    --crop-margin 10 --border-fill white

  # Best transparent crop — corner-based with margin, for compositing
  python3 infer.py --image scan.jpg --crop simple-corners \\
    --crop-margin 10 --crop-transparent

  # Best accuracy — 3-stage refine for improved corner detection
  python3 infer.py --image scan.jpg --pose-refine \\
    --crop warp-stretch --crop-margin 10 --border-fill white

  # Crop a whole folder
  python3 infer.py --image ./scans/ --output ./crops/ \\
    --crop simple-corners --crop-margin 10
""",
    )

    parser.add_argument(
        "--detection-model", "-d", type=str, default=str(DEFAULT_DETECTION_MODEL),
        help="Path to detection ONNX model (default: ../models/detection_model.onnx)",
    )
    parser.add_argument(
        "--pose-model", "-p", type=str, default=str(DEFAULT_POSE_MODEL),
        help="Path to pose ONNX model (default: ../models/pose_model_v2.onnx)",
    )
    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="Path to image file or directory of images to process",
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

    # Validate models
    detection_model_path = Path(args.detection_model).resolve()
    pose_model_path = Path(args.pose_model).resolve()

    if not detection_model_path.exists():
        print(f"Error: Detection model not found: {detection_model_path}")
        print(f"  (default: {DEFAULT_DETECTION_MODEL.resolve()})")
        sys.exit(1)
    if not pose_model_path.exists():
        print(f"Error: Pose model not found: {pose_model_path}")
        print(f"  (default: {DEFAULT_POSE_MODEL.resolve()})")
        sys.exit(1)

    # Load detection model
    print(f"Loading detection model: {detection_model_path}")
    detection_session = load_onnx_model(str(detection_model_path))

    # Load pose model
    print(f"Loading pose model: {pose_model_path}")
    pose_session = load_onnx_model(str(pose_model_path))

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
    if args.sweep_crop_expands:
        sweep_ce = [float(x.strip()) for x in args.sweep_crop_expands.split(",")]
    else:
        sweep_ce = list(SWEEP_CROP_EXPANDS)
    if args.sweep_refine_expands:
        sweep_re = [float(x.strip()) for x in args.sweep_refine_expands.split(",")]
    else:
        sweep_re = list(SWEEP_REFINE_EXPANDS)

    # Run inference — auto-detect file vs directory
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
        )


if __name__ == "__main__":
    main()