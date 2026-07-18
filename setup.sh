#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PhotoScan Import CLI — Environment Setup
# ---------------------------------------------------------------------------
# Usage:
#   ./setup.sh              # create .venv, install deps (default CPU torch)
#   ./setup.sh --gpu        # create .venv, install deps with CUDA 11.8 torch
#   ./setup.sh --cuda12     # create .venv, install deps with CUDA 12.1 torch
#   ./setup.sh --existing   # install deps into the currently-active venv
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJ_DIR}/.venv"
REQUIREMENTS="${PROJ_DIR}/requirements.txt"

# ── Colours (disabled when not a tty) ──────────────────────────────────────
if [ -t 1 ]; then
    GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; NC=''
fi

info()  { printf "${GREEN}[INFO]${NC}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
error() { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; exit 1; }

# ── Parse arguments ─────────────────────────────────────────────────────────
TORCH_VARIANT="cpu"

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)     TORCH_VARIANT="cu118";  shift ;;
        --cuda12)  TORCH_VARIANT="cu121";  shift ;;
        --existing)
            if [ -z "${VIRTUAL_ENV:-}" ]; then
                error "--existing requires an already-active virtual environment"
            fi
            info "Using existing venv: $VIRTUAL_ENV"
            VENV_DIR=""
            shift ;;
        -h|--help)
            echo "Usage: $0 [--gpu | --cuda12 | --existing]"
            echo ""
            echo "  (no flag)   Install CPU-only PyTorch (default)"
            echo "  --gpu       Install PyTorch with CUDA 11.8"
            echo "  --cuda12    Install PyTorch with CUDA 12.1"
            echo "  --existing  Install into the currently-active venv"
            exit 0 ;;
        *) error "Unknown option: $1" ;;
    esac
done

# ── Create venv if needed ───────────────────────────────────────────────────
if [ -n "$VENV_DIR" ]; then
    if [ -d "$VENV_DIR" ]; then
        warn ".venv already exists at $VENV_DIR — reusing it"
    else
        info "Creating virtual environment at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    info "Activating venv"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# ── Upgrade pip ─────────────────────────────────────────────────────────────
info "Upgrading pip"
pip install --upgrade pip --quiet

# ── Install PyTorch with the correct index URL ─────────────────────────────
case "$TORCH_VARIANT" in
    cpu)
        info "Installing PyTorch 2.2.2 (CPU)"
        pip install torch==2.2.2 torchvision==0.17.2
        ;;
    cu118)
        info "Installing PyTorch 2.2.2 (CUDA 11.8)"
        pip install torch==2.2.2 torchvision==0.17.2 \
            --index-url https://download.pytorch.org/whl/cu118
        ;;
    cu121)
        info "Installing PyTorch 2.2.2 (CUDA 12.1)"
        pip install torch==2.2.2 torchvision==0.17.2 \
            --index-url https://download.pytorch.org/whl/cu121
        ;;
esac

# ── Install remaining requirements ──────────────────────────────────────────
# PyTorch is already installed; pip will skip it since the version satisfies
# the constraint in requirements.txt.
info "Installing remaining dependencies from requirements.txt"
pip install -r "$REQUIREMENTS"

# ── Verify ──────────────────────────────────────────────────────────────────
info "Verifying installation"
python -c "
import ultralytics, torch, torchvision, cv2, numpy, PIL, onnx, onnxruntime
print(f'  ultralytics   {ultralytics.__version__}')
print(f'  torch         {torch.__version__}')
print(f'  torchvision   {torchvision.__version__}')
print(f'  opencv-python {cv2.__version__}')
print(f'  numpy         {numpy.__version__}')
print(f'  Pillow        {PIL.__version__}')
print(f'  onnx          {onnx.__version__}')
print(f'  onnxruntime   {onnxruntime.__version__}')
print(f'  CUDA available {torch.cuda.is_available()}')
"

# ── Done ────────────────────────────────────────────────────────────────────
echo ""
info "✅ Setup complete!"
if [ -n "$VENV_DIR" ]; then
    info "Activate with:  source $VENV_DIR/bin/activate"
fi