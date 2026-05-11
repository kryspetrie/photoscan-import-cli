#!/bin/bash
# Auto-resume fiducial training after epoch 50 completes
# 1. Backs up best.pt and last.pt
# 2. Exports best.pt to ONNX
# 3. Launches resume training to epoch 100
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/runs/detect/runs/fiducial/fiducial-corner/weights"
BACKUP_DIR="$WEIGHTS_DIR/backups_50ep"
LOG_DIR="$SCRIPT_DIR"

echo "============================================"
echo "FIDUCIAL AUTO-RESUME SCRIPT"
echo "============================================"
echo "Started: $(date)"
echo ""

# --- Wait for original training to finish ---
echo "Waiting for original fiducial training to complete..."

while true; do
    if pgrep -f "train_fiducial" > /dev/null 2>&1; then
        echo "  [$(date '+%H:%M:%S')] Training still running... checking again in 60s"
        sleep 60
    else
        echo "  [$(date '+%H:%M:%S')] Training process no longer found. Proceeding."
        sleep 10  # Brief pause to ensure final writes are flushed
        break
    fi
done

echo ""
echo "Original training completed."
echo ""

# --- Step 1: Backup best.pt and last.pt ---
echo "Step 1: Backing up weights..."

mkdir -p "$BACKUP_DIR"

if [ -f "$WEIGHTS_DIR/best.pt" ]; then
    cp "$WEIGHTS_DIR/best.pt" "$BACKUP_DIR/best.pt"
    SIZE=$(stat -f%z "$BACKUP_DIR/best.pt" 2>/dev/null || stat -c%s "$BACKUP_DIR/best.pt" 2>/dev/null || echo "unknown")
    echo "  ✓ best.pt backed up ($SIZE bytes)"
else
    echo "  ✗ WARNING: best.pt not found!"
fi

if [ -f "$WEIGHTS_DIR/last.pt" ]; then
    cp "$WEIGHTS_DIR/last.pt" "$BACKUP_DIR/last.pt"
    SIZE=$(stat -f%z "$BACKUP_DIR/last.pt" 2>/dev/null || stat -c%s "$BACKUP_DIR/last.pt" 2>/dev/null || echo "unknown")
    echo "  ✓ last.pt backed up ($SIZE bytes)"
else
    echo "  ✗ WARNING: last.pt not found!"
fi

echo "  Backups stored in: $BACKUP_DIR"
echo ""

# --- Step 2: Export best.pt to ONNX ---
echo "Step 2: Exporting best.pt to ONNX..."

/usr/local/bin/python3 "$SCRIPT_DIR/export_fiducial_onnx.py"

echo ""

# --- Step 3: Verify backups and ONNX before resuming ---
echo "Step 3: Verifying backups and ONNX export..."

ALL_OK=true

if [ ! -f "$BACKUP_DIR/best.pt" ]; then
    echo "  ✗ best.pt backup missing!"
    ALL_OK=false
else
    echo "  ✓ best.pt backup exists"
fi

if [ ! -f "$BACKUP_DIR/last.pt" ]; then
    echo "  ✗ last.pt backup missing!"
    ALL_OK=false
else
    echo "  ✓ last.pt backup exists"
fi

if [ -f "$WEIGHTS_DIR/best.onnx" ]; then
    ONNX_SIZE=$(stat -f%z "$WEIGHTS_DIR/best.onnx" 2>/dev/null || stat -c%s "$WEIGHTS_DIR/best.onnx" 2>/dev/null || echo "unknown")
    echo "  ✓ best.onnx exists ($ONNX_SIZE bytes)"
else
    echo "  ✗ best.onnx missing!"
    ALL_OK=false
fi

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "ERROR: Verification failed. NOT launching resume training."
    echo "Investigate manually before proceeding."
    exit 1
fi

echo ""
echo "All verifications passed!"
echo ""

# --- Step 4: Launch resume training ---
echo "Step 4: Launching resume training to epoch 100..."

nohup /usr/local/bin/python3 "$SCRIPT_DIR/resume_fiducial.py" > "$LOG_DIR/resume_fiducial.log" 2>&1 &
RESUME_PID=$!

echo "  Resume training launched as PID $RESUME_PID"
echo "  Log file: $LOG_DIR/resume_fiducial.log"
echo ""
echo "============================================"
echo "AUTO-RESUME COMPLETE"
echo "============================================"
echo "Finished: $(date)"