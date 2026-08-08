#!/usr/bin/env python3
"""
CHIRA - Setup Script
Script otomatis untuk setup dan menjalankan CHIRA
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

CHIRA_DIR = Path(__file__).parent.resolve()
WEB_DIR = CHIRA_DIR / "web"
MODEL_DIR = CHIRA_DIR / "models" / "chira_disease_detection" / "weights"

def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   CHIRA - Chicken Health Identification and Recommendation   ║
║                    Assistant Setup Script                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

def check_python():
    """Check Python version"""
    print("[1/6] Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.10+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required packages"""
    print("\n[2/6] Installing dependencies...")
    req_file = CHIRA_DIR / "requirements.txt"
    if not req_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], 
                      check=True, capture_output=False)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n[3/6] Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
            if vram >= 12:
                print("   🎯 Recommended model: yolov11m.pt or yolov11l.pt")
            elif vram >= 8:
                print("   🎯 Recommended model: yolov11m.pt")
            elif vram >= 6:
                print("   🎯 Recommended model: yolov11s.pt")
            else:
                print("   ⚠️  Limited VRAM. Recommended: yolov11n.pt")
            return True
        else:
            print("⚠️  No GPU detected. Training will be very slow on CPU.")
            print("   💡 Recommendation: Use Google Colab or Kaggle for training")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def train_model():
    """Train YOLOv11 model"""
    print("\n[4/6] Training YOLOv11 model...")
    print("   This will take 2-4 hours depending on your GPU.")
    
    train_script = CHIRA_DIR / "train_disease.py"
    if not train_script.exists():
        print("❌ train_disease.py not found!")
        return False
    
    try:
        subprocess.run([sys.executable, str(train_script)], check=True)
        print("✅ Model training completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

def setup_django():
    """Setup Django web application"""
    print("\n[5/6] Setting up Django web application...")
    
    os.chdir(WEB_DIR)
    
    # Create media directory
    media_dir = WEB_DIR / "media"
    media_dir.mkdir(exist_ok=True)
    (media_dir / "temp").mkdir(exist_ok=True)
    
    try:
        # Makemigrations
        subprocess.run([sys.executable, "manage.py", "makemigrations"], check=True, capture_output=True)
        print("   ✅ Migrations created")
        
        # Migrate
        subprocess.run([sys.executable, "manage.py", "migrate"], check=True, capture_output=True)
        print("   ✅ Database migrated")
        
        # Collect static
        subprocess.run([sys.executable, "manage.py", "collectstatic", "--noinput"], 
                      check=False, capture_output=True)
        
        print("✅ Django setup completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Django setup failed: {e}")
        return False

def run_server():
    """Run Django development server"""
    print("\n[6/6] Starting CHIRA web server...")
    print("   🌐 Open your browser and go to: http://localhost:8000")
    print("   ⏹️  Press Ctrl+C to stop the server\n")
    
    os.chdir(WEB_DIR)
    try:
        subprocess.run([sys.executable, "manage.py", "runserver", "0.0.0.0:8000"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Thank you for using CHIRA!")

def main():
    print_banner()
    
    # Check if model already exists
    model_exists = (MODEL_DIR / "best.pt").exists() if MODEL_DIR.exists() else False
    
    if not check_python():
        sys.exit(1)
    
    if not install_dependencies():
        print("\n⚠️  Continuing without all dependencies...")
    
    has_gpu = check_gpu()
    
    # Ask for training if model doesn't exist
    if not model_exists:
        print(f"\n📦 Model not found at {MODEL_DIR / 'best.pt'}")
        if has_gpu:
            response = input("\n🚀 Do you want to train the model now? (y/n): ").lower()
            if response == 'y':
                if not train_model():
                    print("\n⚠️  Training failed. You can run it later with: python train_disease.py")
            else:
                print("\n⚠️  Skipping training. The web app will not work without a trained model.")
                print("   Run 'python train_disease.py' later to train the model.")
        else:
            print("\n⚠️  No GPU detected. Training on CPU is not recommended.")
            print("   Options:")
            print("   1. Use Google Colab (free T4 GPU)")
            print("   2. Download a pre-trained model")
            print("   3. Train on CPU (very slow)")
    else:
        print(f"\n✅ Model found at {MODEL_DIR / 'best.pt'}")
    
    # Setup Django
    if not setup_django():
        print("\n⚠️  Django setup incomplete. Please check the errors above.")
        sys.exit(1)
    
    # Run server
    run_server()

if __name__ == "__main__":
    main()
