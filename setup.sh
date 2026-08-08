#!/bin/bash
# CHIRA - Setup Script (Bash)
# Jalankan script ini untuk setup dan menjalankan CHIRA

set -e

CHIRA_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_DIR="$CHIRA_DIR/web"
MODEL_PATH="$CHIRA_DIR/models/chira_disease_detection/weights/best.pt"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   CHIRA - Chicken Health Identification and Recommendation   ║"
echo "║                    Assistant Setup Script                     ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check Python
echo "[1/6] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "    Python version: $PYTHON_VERSION"

# Step 2: Install dependencies
echo ""
echo "[2/6] Installing dependencies..."
cd "$CHIRA_DIR"
pip install -r requirements.txt
echo "    Dependencies installed"

# Step 3: Check GPU
echo ""
echo "[3/6] Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'    GPU: {torch.cuda.get_device_name(0)}')
    print(f'    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('    No GPU detected')
"

# Step 4: Train model if not exists
echo ""
if [ ! -f "$MODEL_PATH" ]; then
    echo "[4/6] Model not found. Training required..."
    echo "    This will take 2-4 hours. Continue? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        cd "$CHIRA_DIR"
        python3 train_disease.py
    else
        echo "    Skipping training. Web app may not work without model."
    fi
else
    echo "[4/6] Model found: $MODEL_PATH"
fi

# Step 5: Setup Django
echo ""
echo "[5/6] Setting up Django..."
cd "$WEB_DIR"
mkdir -p media/temp
python3 manage.py makemigrations
python3 manage.py migrate
echo "    Django setup complete"

# Step 6: Run server
echo ""
echo "[6/6] Starting CHIRA server..."
echo "    🌐 Open browser: http://localhost:8000"
echo "    ⏹️  Press Ctrl+C to stop"
echo ""
python3 manage.py runserver 0.0.0.0:8000
