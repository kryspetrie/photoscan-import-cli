#!/usr/bin/env python3
"""
Download and Prepare DTD Textures for Synthetic Data Generation
===============================================================

Downloads specific texture images from the Describable Textures Dataset (DTD),
then processes them for use as background textures in photo-pose-detector's
synthetic data generator.

Processing pipeline:
  1. Download dtd-r1.0.1.tar.gz from Oxford VGG
  2. Extract the archive
  3. Copy the 85 selected textures to a working directory
  4. Resize to 1200×1200 using Lanczos scaling
  5. Convert to greyscale
  6. Normalize average brightness to medium-grey (128)
  7. Copy processed textures to the textures/ directory

Usage:
    python download_textures.py              # Download, process, and install
    python download_textures.py --skip-download   # Process only (already downloaded)
    python download_textures.py --keep-temp       # Keep temporary files after processing

DTD Dataset:
    https://www.robots.ox.ac.uk/~vgg/data/dtd/
    Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
"""

import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
# SHA-256 of the DTD archive (set to empty string to skip verification)
DTD_SHA256 = ""

# Base directory for this project
PROJECT_ROOT = Path(__file__).resolve().parent
TEXTURES_DIR = PROJECT_ROOT / "textures"

# Temporary directories
DOWNLOAD_DIR = PROJECT_ROOT / ".tmp_textures"
EXTRACT_DIR = DOWNLOAD_DIR / "dtd"

# The 85 specific texture images to extract (organized by category)
SELECTED_TEXTURES = [
    "blotchy_0009.jpg",
    "blotchy_0015.jpg",
    "blotchy_0017.jpg",
    "blotchy_0019.jpg",
    "blotchy_0022.jpg",
    "blotchy_0028.jpg",
    "blotchy_0032.jpg",
    "blotchy_0033.jpg",
    "blotchy_0047.jpg",
    "blotchy_0048.jpg",
    "blotchy_0049.jpg",
    "blotchy_0053.jpg",
    "blotchy_0054.jpg",
    "blotchy_0059.jpg",
    "blotchy_0067.jpg",
    "blotchy_0089.jpg",
    "blotchy_0099.jpg",
    "blotchy_0100.jpg",
    "cracked_0046.jpg",
    "cracked_0062.jpg",
    "cracked_0111.jpg",
    "cracked_0118.jpg",
    "cracked_0158.jpg",
    "crosshatched_0050.jpg",
    "crosshatched_0071.jpg",
    "crosshatched_0092.jpg",
    "crosshatched_0093.jpg",
    "crosshatched_0099.jpg",
    "crosshatched_0116.jpg",
    "fibrous_0089.jpg",
    "fibrous_0137.jpg",
    "fibrous_0162.jpg",
    "gauzy_0096.jpg",
    "gauzy_0125.jpg",
    "gauzy_0143.jpg",
    "knitted_0079.jpg",
    "knitted_0091.jpg",
    "knitted_0118.jpg",
    "marbled_0076.jpg",
    "marbled_0080.jpg",
    "marbled_0082.jpg",
    "marbled_0177.jpg",
    "marbled_0184.jpg",
    "marbled_0194.jpg",
    "smeared_0042.jpg",
    "smeared_0052.jpg",
    "smeared_0066.jpg",
    "smeared_0093.jpg",
    "smeared_0116.jpg",
    "sprinkled_0033.jpg",
    "sprinkled_0052.jpg",
    "stained_0033.jpg",
    "stained_0038.jpg",
    "stained_0047.jpg",
    "stained_0050.jpg",
    "stained_0052.jpg",
    "stained_0051.jpg",
    "stained_0057.jpg",
    "stained_0066.jpg",
    "stained_0067.jpg",
    "stained_0070.jpg",
    "stained_0075.jpg",
    "stained_0077.jpg",
    "stained_0080.jpg",
    "stained_0097.jpg",
    "stained_0105.jpg",
    "stained_0108.jpg",
    "stained_0133.jpg",
    "veined_0088.jpg",
    "veined_0090.jpg",
    "woven_0009.jpg",
    "woven_0016.jpg",
    "woven_0022.jpg",
    "wrinkled_0013.jpg",
    "wrinkled_0021.jpg",
    "wrinkled_0025.jpg",
    "wrinkled_0027.jpg",
    "wrinkled_0039.jpg",
    "wrinkled_0043.jpg",
    "wrinkled_0046.jpg",
    "wrinkled_0065.jpg",
    "wrinkled_0072.jpg",
    "wrinkled_0085.jpg",
    "wrinkled_0090.jpg",
    "wrinkled_0100.jpg",
    "wrinkled_0115.jpg",
    "wrinkled_0129.jpg",
    "wrinkled_0134.jpg",
]

# Processing parameters
TARGET_SIZE = 1200
TARGET_BRIGHTNESS = 128  # Medium grey


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dtd():
    """Download the DTD archive if not already present."""
    import urllib.request

    archive_path = DOWNLOAD_DIR / "dtd-r1.0.1.tar.gz"

    if archive_path.exists():
        print(f"Archive already exists at {archive_path}")
        if _verify_sha256(archive_path):
            print("  SHA-256 verified ✓")
            return archive_path
        else:
            print("  SHA-256 mismatch — re-downloading...")
            archive_path.unlink()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DTD dataset from:\n  {DTD_URL}")
    print(f"Saving to: {archive_path}")
    print("This may take a few minutes (~600 MB)...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
        sys.stdout.write(f"\r  {pct:3d}% ({mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(DTD_URL, str(archive_path), _progress)
    print()  # newline after progress

    if not _verify_sha256(archive_path):
        print("ERROR: SHA-256 verification failed! The download may be corrupt.")
        print("Delete the archive and try again.")
        sys.exit(1)
    print("SHA-256 verified ✓")
    return archive_path


def _verify_sha256(path: Path) -> bool:
    """Verify file SHA-256 hash. Returns True if hash matches or is not configured."""
    if not DTD_SHA256:
        print("  SHA-256 not configured — skipping verification")
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest() == DTD_SHA256


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_dtd(archive_path: Path) -> Path:
    """Extract the DTD archive."""
    if (EXTRACT_DIR / "dtd").exists():
        print(f"Already extracted at {EXTRACT_DIR}")
        return EXTRACT_DIR

    print(f"Extracting {archive_path}...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=str(EXTRACT_DIR))
    print("  Done.")
    return EXTRACT_DIR


# ---------------------------------------------------------------------------
# Select and copy
# ---------------------------------------------------------------------------

def find_texture_in_dtd(extract_dir: Path, filename: str) -> Path | None:
    """Find a texture image in the DTD directory structure.

    DTD organizes images as: dtd/images/{category}/{filename}
    Some images also appear in: dtd/images/{category}/{category}_{nnn}.jpg
    """
    # The category is the prefix before the underscore and number
    category = filename.rsplit("_", 1)[0]

    # Try the expected category directory
    candidate = extract_dir / "dtd" / "images" / category / filename
    if candidate.exists():
        return candidate

    # Search more broadly
    for img_path in (extract_dir / "dtd" / "images").rglob(filename):
        return img_path

    return None


def copy_selected_textures(extract_dir: Path) -> Path:
    """Copy selected texture images to a working directory."""
    work_dir = DOWNLOAD_DIR / "selected_textures"
    work_dir.mkdir(parents=True, exist_ok=True)

    found = 0
    missing = []

    for filename in SELECTED_TEXTURES:
        src = find_texture_in_dtd(extract_dir, filename)
        if src is None:
            missing.append(filename)
            continue
        dst = work_dir / filename
        shutil.copy2(src, dst)
        found += 1

    print(f"Copied {found}/{len(SELECTED_TEXTURES)} selected textures to {work_dir}")
    if missing:
        print(f"WARNING: {len(missing)} textures not found in archive:")
        for m in missing:
            print(f"  {m}")

    return work_dir


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_texture(img: np.ndarray) -> np.ndarray:
    """Process a texture image:
    1. Resize to 1200×1200 with Lanczos scaling
    2. Convert to greyscale
    3. Normalize average brightness to medium-grey (128)
    """
    # 1. Resize with Lanczos (INTER_AREA for downscaling, INTER_CUBIC for upscaling)
    h, w = img.shape[:2]
    if (h, w) != (TARGET_SIZE, TARGET_SIZE):
        if h > TARGET_SIZE and w > TARGET_SIZE:
            # Downscaling — INTER_AREA gives best quality
            img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        else:
            # Upscaling or mixed — Lanczos4
            img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)

    # 2. Convert to greyscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Normalize average brightness to medium-grey (128)
    current_mean = img.mean()
    if current_mean > 0:
        # Scale so that the mean becomes 128; clip to valid range
        scale = TARGET_BRIGHTNESS / current_mean
        img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    return img


def process_all_textures(input_dir: Path) -> Path:
    """Process all selected textures and save to output directory."""
    output_dir = DOWNLOAD_DIR / "processed_textures"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")))
    print(f"\nProcessing {len(image_files)} textures...")

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Could not read {img_path}")
            continue

        processed = process_texture(img)
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), processed, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"  Processed {len(image_files)} textures → {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def install_textures(processed_dir: Path) -> None:
    """Copy processed textures to the project textures/ directory."""
    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old numbered texture files
    for old in TEXTURES_DIR.glob("texture_*.jpg"):
        old.unlink()
    for old in TEXTURES_DIR.glob("texture_*.png"):
        old.unlink()

    # Copy new textures with sequential naming
    image_files = sorted(list(processed_dir.glob("*.jpg")))
    for i, img_path in enumerate(image_files, 1):
        dst = TEXTURES_DIR / f"texture_{i:02d}.jpg"
        shutil.copy2(img_path, dst)

    # Also copy with original names for reference
    for img_path in image_files:
        dst = TEXTURES_DIR / img_path.name
        shutil.copy2(img_path, dst)

    print(f"\nInstalled {len(image_files)} textures to {TEXTURES_DIR}/")
    print(f"  Sequential names: texture_01.jpg … texture_{len(image_files):02d}.jpg")
    print(f"  Original names also preserved for reference")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup(keep_temp: bool = False) -> None:
    """Remove temporary files."""
    if keep_temp:
        print(f"\nKeeping temporary files at {DOWNLOAD_DIR}/")
        print("  Re-run with --skip-download to reuse them")
        return

    print(f"\nCleaning up temporary files...")
    shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
    print("  Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and prepare DTD textures for photo-pose-detector"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading (use previously downloaded archive)"
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep temporary files after processing"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DTD Texture Download & Processing")
    print("=" * 60)
    print(f"Target textures: {len(SELECTED_TEXTURES)}")
    print(f"Output size: {TARGET_SIZE}×{TARGET_SIZE} greyscale, brightness → {TARGET_BRIGHTNESS}")
    print(f"Output dir: {TEXTURES_DIR}")
    print("=" * 60)

    # Step 1: Download
    if args.skip_download:
        archive_path = DOWNLOAD_DIR / "dtd-r1.0.1.tar.gz"
        if not archive_path.exists():
            print(f"ERROR: No archive found at {archive_path}")
            print("Run without --skip-download first.")
            sys.exit(1)
        print(f"Using existing archive: {archive_path}")
        if _verify_sha256(archive_path):
            print("SHA-256 verified ✓")
        else:
            print("WARNING: SHA-256 verification failed. Archive may be corrupt.")
    else:
        archive_path = download_dtd()

    # Step 2: Extract
    extract_dir = extract_dtd(archive_path)

    # Step 3: Copy selected textures
    selected_dir = copy_selected_textures(extract_dir)

    # Step 4: Process (resize, greyscale, normalize brightness)
    processed_dir = process_all_textures(selected_dir)

    # Step 5: Install to textures/
    install_textures(processed_dir)

    # Step 6: Cleanup
    cleanup(keep_temp=args.keep_temp)

    # Final summary
    texture_count = len(list(TEXTURES_DIR.glob("*.jpg")))
    print(f"\n✅ Done! {texture_count} texture files ready in {TEXTURES_DIR}/")
    print("\nAttribution:")
    print("  Textures from the Describable Textures Dataset (DTD)")
    print("  https://www.robots.ox.ac.uk/~vgg/data/dtd/")
    print("  Licensed under CC BY-NC-SA 4.0")


if __name__ == "__main__":
    main()