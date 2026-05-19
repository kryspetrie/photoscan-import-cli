#!/usr/bin/env python3
"""Benchmark photocrop preset+crop combinations against ground truth.

Runs each combination on val images, measures timing and corner accuracy.
Then scales images to larger resolutions and repeats to capture timing scaling.

Usage:
    python3 benchmark_presets.py [--images N] [--skip-scaling]
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onnx_inference.photocrop import (
    load_onnx_model,
    infer_single,
    _PRESETS,
    _PRESET_CROP_DEFAULTS,
    KEYPOINT_NAMES,
)

# ---------------------------------------------------------------------------
# Ground truth parsing
# ---------------------------------------------------------------------------

# Keypoint order in YOLO labels matches our convention:
# kp0=LL, kp1=UL, kp2=UR, kp3=LR
LABEL_KP_NAMES = ["LL", "UL", "UR", "LR"]


def parse_label_file(label_path: str, img_w: int, img_h: int):
    """Parse a YOLO pose label file into a list of ground-truth photos.

    Each photo is a dict with 'keypoints' list of {name, x, y, visibility}.
    Coordinates are converted from normalized to absolute pixels.
    """
    photos = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:  # class cx cy w h + 4 keypoints * 3 values = 5+12=17
                continue
            # Box (not needed for comparison, but skip zero-size)
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Keypoints: 4 corners × (x, y, visibility) starting at index 5
            keypoints = []
            for kp_idx in range(4):
                base = 5 + kp_idx * 3
                kx = float(parts[base]) * img_w
                ky = float(parts[base + 1]) * img_h
                vis = int(float(parts[base + 2]))
                keypoints.append({
                    "name": LABEL_KP_NAMES[kp_idx],
                    "x": kx,
                    "y": ky,
                    "visibility": vis,
                })
            photos.append({"keypoints": keypoints})
    return photos


# ---------------------------------------------------------------------------
# Match detections to ground truth
# ---------------------------------------------------------------------------

def match_detections_to_gt(results: list, gt_photos: list, dist_threshold: float = 50):
    """Match each detected photo to the nearest ground truth photo by center distance.

    Returns a list of (det_result, gt_photo) tuples for matched pairs.
    Unmatched detections or GT photos are counted as misses.
    """
    if not results or not gt_photos:
        return [], len(results), len(gt_photos)

    def center(photo):
        kps = photo.get("keypoints", [])
        if not kps:
            return (0, 0)
        xs = [kp["x"] for kp in kps if kp.get("visibility", 0) > 0]
        ys = [kp["y"] for kp in kps if kp.get("visibility", 0) > 0]
        if not xs:
            xs = [kp["x"] for kp in kps]
            ys = [kp["y"] for kp in kps]
        return (np.mean(xs), np.mean(ys))

    # Greedy matching by distance
    det_used = [False] * len(results)
    gt_used = [False] * len(gt_photos)
    matches = []

    # Compute all distances
    pairs = []
    for di, det in enumerate(results):
        dc = center(det)
        for gi, gt in enumerate(gt_photos):
            gc = center(gt)
            dist = np.sqrt((dc[0] - gc[0])**2 + (dc[1] - gc[1])**2)
            pairs.append((dist, di, gi))
    pairs.sort()

    for dist, di, gi in pairs:
        if det_used[di] or gt_used[gi]:
            continue
        if dist > dist_threshold:
            break
        matches.append((results[di], gt_photos[gi]))
        det_used[di] = True
        gt_used[gi] = True

    unmatched_det = sum(1 for d in det_used if not d)
    unmatched_gt = sum(1 for g in gt_used if not g)
    return matches, unmatched_det, unmatched_gt


def compute_corner_errors(det, gt):
    """Compute per-corner pixel distance between detection and ground truth."""
    det_kps = {kp["name"]: kp for kp in det.get("keypoints", [])}
    gt_kps = {kp["name"]: kp for kp in gt.get("keypoints", [])}

    errors = {}
    for name in LABEL_KP_NAMES:
        dk = det_kps.get(name)
        gk = gt_kps.get(name)
        if dk and gk and gk.get("visibility", 0) > 0:
            dist = np.sqrt((dk["x"] - gk["x"])**2 + (dk["y"] - gk["y"])**2)
            errors[name] = dist
        else:
            errors[name] = None  # Missing corner
    return errors


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

# Preset × crop combinations to test
COMBINATIONS = [
    ("quick", "simple-corners"),
    ("quick", "warp-stretch"),
    ("standard", "simple"),
    ("standard", "simple-corners"),
    ("standard", "warp"),
    ("standard", "warp-stretch"),
    ("thorough", "simple-corners"),
    ("thorough", "warp-stretch"),
    (None, "simple-corners"),  # No preset, manual crop only
]

# Resolution scaling targets (megapixels)
SCALE_RESOLUTIONS = [0.4, 2, 5, 10, 20]  # 0.4MP = 640×640 (native)


def scale_image(img: Image.Image, target_mp: float) -> Image.Image:
    """Scale image to approximately target_mp megapixels, preserving aspect ratio."""
    w, h = img.size
    current_mp = w * h / 1e6
    if abs(current_mp - target_mp) < 0.01:
        return img
    scale = np.sqrt(target_mp / current_mp)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Make divisible by 2 for cleaner processing
    new_w = max(2, new_w // 2 * 2)
    new_h = max(2, new_h // 2 * 2)
    return img.resize((new_w, new_h), Image.LANCZOS)


def run_benchmark(
    detection_session,
    pose_session,
    val_img_dir: str,
    val_label_dir: str,
    preset_name: str | None,
    crop_mode: str,
    max_images: int = 0,
    target_mp: float = 0.4,
):
    """Run pipeline on val images and collect timing + accuracy."""

    img_dir = Path(val_img_dir)
    label_dir = Path(val_label_dir)

    # Collect image files
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    img_files = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in extensions and not p.name.endswith("_detected.jpg")
    ])

    if max_images > 0:
        img_files = img_files[:max_images]

    # Map preset name to kwargs
    preset_kwargs = {}
    if preset_name and preset_name in _PRESETS:
        preset_kwargs = dict(_PRESETS[preset_name]["args"])
        # Apply crop defaults if crop mode specified
        crop_defaults = _PRESET_CROP_DEFAULTS.get(preset_name, {})

    all_corner_errors = []
    total_time = 0.0
    n_images = 0
    n_detected = 0
    n_gt_total = 0
    n_false_positives = 0
    n_missed = 0
    crop_errors = []  # Track warp failures if any

    for img_path in img_files:
        # Find corresponding label
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        # Load and potentially scale image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        if target_mp < 0.39 or target_mp > 0.41:  # Not native 640×640
            img = scale_image(img, target_mp)

        cur_w, cur_h = img.size
        scale_x = cur_w / orig_w
        scale_y = cur_h / orig_h

        # Parse GT — scale to match the image we're actually processing
        gt_photos = parse_label_file(str(label_path), cur_w, cur_h)

        # Save scaled image to temp file for infer_single
        import tempfile, shutil
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name, quality=95)
            tmp_path = tmp.name

        tmp_crop_dir = tempfile.mkdtemp(prefix="photocrop_bench_")

        try:
            t0 = time.perf_counter()
            results = infer_single(
                detection_session, pose_session, tmp_path,
                det_conf=0.5, pose_conf=0.3,
                crop_mode=crop_mode, crop_dir=tmp_crop_dir,
                **preset_kwargs,
            )
            elapsed = time.perf_counter() - t0
        finally:
            os.unlink(tmp_path)
            shutil.rmtree(tmp_crop_dir, ignore_errors=True)

        total_time += elapsed
        n_images += 1

        # Match detections to GT
        matches, un_det, un_gt = match_detections_to_gt(results, gt_photos)
        n_detected += len(results)
        n_gt_total += len(gt_photos)
        n_false_positives += un_det
        n_missed += un_gt

        for det, gt in matches:
            errors = compute_corner_errors(det, gt)
            for name, err in errors.items():
                if err is not None:
                    all_corner_errors.append(err)

    # Aggregate
    if all_corner_errors:
        arr = np.array(all_corner_errors)
        stats = {
            "mean_px": float(np.mean(arr)),
            "median_px": float(np.median(arr)),
            "p90_px": float(np.percentile(arr, 90)),
            "p95_px": float(np.percentile(arr, 95)),
            "max_px": float(np.max(arr)),
            "n_corners": len(arr),
        }
    else:
        stats = {
            "mean_px": None,
            "median_px": None,
            "p90_px": None,
            "p95_px": None,
            "max_px": None,
            "n_corners": 0,
        }

    return {
        "preset": preset_name or "(none)",
        "crop": crop_mode,
        "target_mp": target_mp,
        "resolution": f"{cur_w}×{cur_h}",
        "n_images": n_images,
        "n_detected": n_detected,
        "n_gt_total": n_gt_total,
        "n_false_positives": n_false_positives,
        "n_missed": n_missed,
        "avg_time_s": total_time / max(1, n_images),
        "total_time_s": total_time,
        "accuracy": stats,
    }


def print_results(results: list):
    """Print a formatted results table."""
    print("\n" + "=" * 100)
    print(f"{'Preset':<12} {'Crop':<16} {'Res':<12} {'Avg(s)':<8} {'Mean(px)':<10} "
          f"{'Med(px)':<10} {'P90(px)':<10} {'P95(px)':<10} {'Det':<5} {'GT':<5} "
          f"{'FP':<4} {'Miss':<5}")
    print("-" * 100)
    for r in results:
        a = r["accuracy"]
        det = r["n_detected"]
        gt = r["n_gt_total"]
        print(f"{r['preset']:<12} {r['crop']:<16} {r['resolution']:<12} "
              f"{r['avg_time_s']:<8.2f} "
              f"{a['mean_px'] or '-':<10} "
              f"{a['median_px'] or '-':<10} "
              f"{a['p90_px'] or '-':<10} "
              f"{a['p95_px'] or '-':<10} "
              f"{det:<5} {gt:<5} {r['n_false_positives']:<4} {r['n_missed']:<5}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark photocrop presets")
    parser.add_argument("--images", type=int, default=50,
                        help="Max images to process per combination (default: 50)")
    parser.add_argument("--skip-scaling", action="store_true",
                        help="Skip scaled resolution benchmarks (native 640×640 only)")
    parser.add_argument("--val-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "data" / "images" / "val"),
                        help="Path to val images directory")
    parser.add_argument("--label-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "data" / "pose" / "labels" / "val"),
                        help="Path to val label files directory")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    label_dir = Path(args.label_dir)

    if not val_dir.exists():
        print(f"Error: val images not found at {val_dir}")
        sys.exit(1)
    if not label_dir.exists():
        print(f"Error: val labels not found at {label_dir}")
        sys.exit(1)

    # Count available images
    extensions = {".jpg", ".jpeg", ".png"}
    n_available = sum(1 for p in val_dir.iterdir() if p.suffix.lower() in extensions)
    n_use = min(args.images, n_available)
    print(f"Benchmarking {n_use} images from {val_dir}")
    print(f"Ground truth labels: {label_dir}")

    # Load models
    script_dir = Path(__file__).resolve().parent.parent / "models"
    det_model = script_dir / "detection_ep47.onnx"
    pose_model = script_dir / "pose_single_ep42.onnx"

    if not det_model.exists():
        print(f"Error: Detection model not found at {det_model}")
        sys.exit(1)
    if not pose_model.exists():
        print(f"Error: Pose model not found at {pose_model}")
        sys.exit(1)

    print(f"\nLoading detection model: {det_model}")
    det_session = load_onnx_model(str(det_model))
    print(f"Loading pose model: {pose_model}")
    pose_session = load_onnx_model(str(pose_model))

    all_results = []

    # Native resolution first
    resolutions = [0.4]  # 640×640 ≈ 0.4MP
    if not args.skip_scaling:
        resolutions = SCALE_RESOLUTIONS

    for target_mp in resolutions:
        mp_label = f"~{target_mp}MP"
        print(f"\n{'=' * 60}")
        print(f"  Resolution: {mp_label}")
        print(f"{'=' * 60}")

        for preset_name, crop_mode in COMBINATIONS:
            label = f"{preset_name or '(none)'}+{crop_mode}"
            print(f"\n  Running {label} at {mp_label}...", end="", flush=True)

            result = run_benchmark(
                det_session, pose_session,
                str(val_dir), str(label_dir),
                preset_name, crop_mode,
                max_images=n_use,
                target_mp=target_mp,
            )

            a = result["accuracy"]
            print(f" {result['avg_time_s']:.2f}s/img, "
                  f"mean_err={a['mean_px'] or 'N/A'}px, "
                  f"median={a['median_px'] or 'N/A'}px")
            all_results.append(result)

    # Print final table
    print_results(all_results)

    # Save raw results as JSON
    results_file = Path(__file__).resolve().parent.parent / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {results_file}")


if __name__ == "__main__":
    main()