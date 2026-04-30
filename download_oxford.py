#!/usr/bin/env python3
"""
Download Oxford Buildings Dataset for Synthetic Data Generation
===============================================================

Downloads the Oxford Buildings Dataset (Oxford5k) and extracts the images
into the data_generator/images/ directory for use as source photos in the
synthetic data generator.

The dataset contains 5,062 images of Oxford landmarks and architecture,
created by the Visual Geometry Group at the University of Oxford.

Usage:
    python download_oxford.py              # Download and extract
    python download_oxford.py --verify-only   # Just verify existing images

Oxford Buildings Dataset:
    https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
    Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0

Expected result:
    data_generator/images/
    ├── all_souls_000000.jpg
    ├── all_souls_000001.jpg
    ├── ...
    └── windmill_000897.jpg   (5,062 total files)
"""

import hashlib
import os
import sys
import tarfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "data_generator" / "images"

OXFORD_URLS = [
    "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz",
]

# Known hashes for verification (optional — set to empty string to skip)
OXFORD_SHA256 = ""  # Archive is large; we verify file count instead

EXPECTED_IMAGE_COUNT = 5062


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_oxford():
    """Download the Oxford Buildings archive."""
    import urllib.request

    download_dir = PROJECT_ROOT / ".tmp_oxford"
    download_dir.mkdir(parents=True, exist_ok=True)

    archive_path = download_dir / "oxbuild_images.tgz"

    if archive_path.exists():
        print(f"Archive already exists at {archive_path}")
        print("Skipping download. Delete the archive to re-download.")
        return archive_path

    url = OXFORD_URLS[0]
    print(f"Downloading Oxford Buildings Dataset from:")
    print(f"  {url}")
    print(f"Saving to: {archive_path}")
    print("This may take several minutes (~1.5 GB)...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            sys.stdout.write(f"\r  {pct:3d}% ({mb:.1f}/{total_mb:.1f} MB)")
        else:
            sys.stdout.write(f"\r  {mb:.1f} MB downloaded")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(archive_path), _progress)
    print()  # newline after progress
    print("Download complete.")
    return archive_path


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_oxford(archive_path: Path) -> None:
    """Extract the Oxford Buildings archive into data_generator/images/.

    The archive contains images at the top level (no subdirectory), or
    possibly under an 'images/' subdirectory. We handle both cases.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting images to {IMAGES_DIR}/...")

    with tarfile.open(archive_path, "r:gz") as tar:
        # Check archive structure
        members = tar.getmembers()

        # Find image files in the archive
        image_members = [
            m for m in members
            if m.name.lower().endswith(('.jpg', '.jpeg', '.png'))
               and not m.name.startswith('.')
        ]

        if not image_members:
            print("ERROR: No image files found in archive!")
            sys.exit(1)

        print(f"  Found {len(image_members)} images in archive")

        # Extract images, flattening directory structure
        extracted = 0
        for member in image_members:
            # Get just the filename, stripping any directory prefix
            filename = Path(member.name).name

            # Skip if already exists
            dest = IMAGES_DIR / filename
            if dest.exists():
                extracted += 1
                continue

            # Extract the file content
            f = tar.extractfile(member)
            if f is not None:
                with open(dest, "wb") as out:
                    out.write(f.read())
                extracted += 1

        print(f"  Extracted {extracted} images")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_images() -> bool:
    """Verify that the Oxford Buildings images are in place."""
    if not IMAGES_DIR.exists():
        print(f"ERROR: Images directory not found: {IMAGES_DIR}")
        return False

    image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg"))
    count = len(image_files)

    print(f"Found {count} images in {IMAGES_DIR}/")

    if count >= EXPECTED_IMAGE_COUNT:
        print(f"  ✓ Expected {EXPECTED_IMAGE_COUNT}+ images, found {count}")
        return True
    elif count > 0:
        print(f"  ⚠ Expected {EXPECTED_IMAGE_COUNT} images, found {count}")
        print(f"  Some images may be missing. Re-run download to complete.")
        return True
    else:
        print(f"  ✗ No images found. Run download first.")
        return False


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup() -> None:
    """Remove temporary download files."""
    download_dir = PROJECT_ROOT / ".tmp_oxford"
    if download_dir.exists():
        import shutil
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(download_dir, ignore_errors=True)
        print("  Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Oxford Buildings Dataset for photo-pose-detector"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Just verify existing images, don't download"
    )
    parser.add_argument(
        "--keep-archive", action="store_true",
        help="Keep the downloaded archive after extraction"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Oxford Buildings Dataset Download")
    print("=" * 60)
    print(f"Target directory: {IMAGES_DIR}/")
    print(f"Expected images:  {EXPECTED_IMAGE_COUNT}")
    print("=" * 60)

    if args.verify_only:
        ok = verify_images()
        sys.exit(0 if ok else 1)

    # Verify if already present
    if verify_images():
        print("\nImages already present. Use --verify-only to just check.")
        print("Delete data_generator/images/ to re-download.")

    # Step 1: Download
    archive_path = download_oxford()

    # Step 2: Extract
    extract_oxford(archive_path)

    # Step 3: Verify
    print("\nFinal verification:")
    ok = verify_images()

    # Step 4: Cleanup
    if not args.keep_archive:
        cleanup()
    else:
        archive_dir = PROJECT_ROOT / ".tmp_oxford"
        print(f"\nKeeping archive at {archive_dir}/")

    if ok:
        print(f"\n✅ Done! {len(list(IMAGES_DIR.glob('*.jpg')))} source images ready in {IMAGES_DIR}/")
        print("\nAttribution:")
        print("  Images from the Oxford Buildings Dataset (Oxford5k)")
        print("  https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/")
        print("  Created by the Visual Geometry Group, University of Oxford")
        print("  Licensed under CC BY-NC-SA 4.0")
    else:
        print("\n❌ Verification failed. Check the download and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()