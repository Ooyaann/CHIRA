#!/bin/bash
# CHIRA - Setup Script untuk Linux/macOS/WSL
# Script otomatis dengan virtual environment

set -e

CHIRA_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$CHIRA_DIR/venv"
WEB_DIR="$CHIRA_DIR/web"
MODEL_PATH="$CHIRA_DIR/models/chira_disease_detection/weights/best.pt"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   CHIRA - Chicken Health Identification and Recommendation   ║"
echo "║              Linux/macOS/WSL Setup Script                     ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check Python
echo "[1/7] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 tidak ditemukan! Install Python 3.10+ terlebih dahulu."
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "    ✅ Python $PYTHON_VERSION ditemukan"

# Step 2: Create Virtual Environment
echo ""
echo "[2/7] Membuat virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "    ℹ️  Virtual environment sudah ada, menggunakan yang ada..."
else
    python3 -m venv "$VENV_DIR"
    echo "    ✅ Virtual environment dibuat di $VENV_DIR"
fi

# Step 3: Activate venv & Install dependencies
echo ""
echo "[3/7] Install dependencies ke virtual environment..."
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
pip install -r "$CHIRA_DIR/requirements.txt"
echo "    ✅ Dependencies terinstall di venv"

# Step 4: Check GPU
echo ""
echo "[4/7] Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'    ✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'    ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('    ⚠️  Tidak ada GPU terdeteksi. Training akan sangat lambat.')
"

# Step 5: Training model if not exists
echo ""
if [ -f "$MODEL_PATH" ]; then
    echo "[5/7] ✅ Model sudah ada: $MODEL_PATH"
else
    echo "[5/7] Model belum ditemukan di $MODEL_PATH"
    echo ""
    read -p "🚀 Jalankan training sekarang? Butuh 2-4 jam. (y/n): " TRAIN
    if [ "$TRAIN" = "y" ] || [ "$TRAIN" = "Y" ]; then
        echo "    ⏳ Training dimulai..."
        cd "$CHIRA_DIR"
        python3 train_disease.py
    else
        echo "    ⚠️  Training dilewati. Web tidak akan bisa deteksi tanpa model."
        echo "    Jalankan 'python3 train_disease.py' nanti untuk training."
    fi
fi

# Step 6: Setup Django
echo ""
echo "[6/7] Setup Django..."
cd "$WEB_DIR"
mkdir -p media/temp
python3 manage.py makemigrations
python3 manage.py migrate
echo "    ✅ Django siap"

# Step 7: Run server
echo ""
echo "[7/7] Menjalankan CHIRA Server..."
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  🌐 Buka browser: http://localhost:8000                       ║"
echo "║  ⏹️  Tekan Ctrl+C untuk menghentikan server                   ║"
echo "║  📁 Virtual env aktif: source venv/bin/activate               ║"
echo "║  📁 Untuk nonaktif: deactivate                                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

python3 manage.py runserver 0.0.0.0:8000

echo ""
echo "👋 Server dihentikan. Terima kasih telah menggunakan CHIRA!"
