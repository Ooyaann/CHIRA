"""
CHIRA - Chicken Health Identification and Recommendation Assistant
Unified AI Model Training Pipeline (YOLOv11)

Purpose: Train, evaluate, and export YOLOv11 model for poultry disease detection.
Supports: Cross-platform execution (Windows & Linux).
"""

import os
import sys
import yaml
import glob
import shutil
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Dynamic Base Directories
BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 15 Standard Disease & Health Classes
FINAL_CLASSES = [
    'healthy', 'fowl_pox', 'coryza', 'coccidiosis', 'newcastle_disease',
    'salmonellosis', 'avian_influenza', 'gumboro', 'mareks_disease',
    'infectious_bronchitis', 'aspergillosis', 'botulism',
    'external_parasites', 'egg_peritonitis', 'sick_unspecified'
]

# Class Mapping Harmonization
CLASS_MAPPING = {
    'healthy': 'healthy', 'sehat': 'healthy', 'normal': 'healthy', 'Healthy': 'healthy',
    'Fowlpox': 'fowl_pox', 'fowlpox': 'fowl_pox', 'fowl_pox': 'fowl_pox', 'FowlPox': 'fowl_pox',
    'Coryza': 'coryza', 'coryza': 'coryza', 'coryza_infectious': 'coryza',
    'Coccidiosis': 'coccidiosis', 'coccidiosis': 'coccidiosis', 'coccidia': 'coccidiosis',
    'Newcastle': 'newcastle_disease', 'newcastle': 'newcastle_disease', 
    'NewcastleDisease': 'newcastle_disease', 'newcastle_disease': 'newcastle_disease', 'tetelo': 'newcastle_disease',
    'Salmonella': 'salmonellosis', 'salmonella': 'salmonellosis', 'salmonellosis': 'salmonellosis',
    'AvianInfluenza': 'avian_influenza', 'avian_influenza': 'avian_influenza', 
    'bird_flu': 'avian_influenza', 'flu_burung': 'avian_influenza',
    'Gumboro': 'gumboro', 'gumboro': 'gumboro', 'IBD': 'gumboro', 'infectious_bursal_disease': 'gumboro',
    'Mareks': 'mareks_disease', 'mareks': 'mareks_disease', 'mareks_disease': 'mareks_disease',
    'InfectiousBronchitis': 'infectious_bronchitis', 'infectious_bronchitis': 'infectious_bronchitis', 'bronchitis': 'infectious_bronchitis',
    'Aspergillosis': 'aspergillosis', 'aspergillosis': 'aspergillosis',
    'Botulism': 'botulism', 'botulism': 'botulism',
    'ExternalParasites': 'external_parasites', 'external_parasites': 'external_parasites', 
    'parasites': 'external_parasites', 'mite': 'external_parasites', 'lice': 'external_parasites',
    'EggPeritonitis': 'egg_peritonitis', 'egg_peritonitis': 'egg_peritonitis', 'peritonitis': 'egg_peritonitis',
    'sick': 'sick_unspecified', 'Sick': 'sick_unspecified', 'sakit': 'sick_unspecified', 
    'diseased': 'sick_unspecified', 'unhealthy': 'sick_unspecified',
}

def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def check_environment():
    """Verify Python & GPU acceleration environment"""
    print_section("ENVIRONMENT CHECK")
    print(f"Base Directory: {BASE_DIR}")
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Detected: {gpu_name} ({vram:.1f} GB VRAM)")
        return True, vram
    else:
        print("WARNING: No GPU detected. Training will run on CPU.")
        return False, 0

def prepare_merged_dataset():
    """Locate and harmonize local YOLO datasets into a unified dataset"""
    print_section("DATASET HARMONIZATION")
    merged_dir = DATASET_DIR / "merged_disease_dataset"
    data_yaml_path = merged_dir / "data.yaml"

    if data_yaml_path.exists():
        print(f"Using existing merged dataset: {data_yaml_path}")
        return str(data_yaml_path)

    all_samples = []
    for item in os.listdir(DATASET_DIR):
        item_path = DATASET_DIR / item
        if not item_path.is_dir():
            continue
        yaml_path = item_path / "data.yaml"
        if not yaml_path.exists():
            continue
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        orig_classes = config.get('names', [])
        
        for split in ['train', 'valid', 'test']:
            img_dir = item_path / split / 'images'
            lbl_dir = item_path / split / 'labels'
            if not lbl_dir.exists():
                continue
            for lf in glob.glob(str(lbl_dir / '*.txt')):
                with open(lf, 'r') as f:
                    lines = f.readlines()
                if not lines:
                    continue
                new_lines = []
                valid = True
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    orig_id = int(parts[0])
                    if orig_id >= len(orig_classes):
                        valid = False
                        break
                    orig_name = orig_classes[orig_id]
                    mapped = CLASS_MAPPING.get(orig_name) or CLASS_MAPPING.get(orig_name.lower())
                    if not mapped:
                        valid = False
                        break
                    new_id = FINAL_CLASSES.index(mapped)
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                
                if valid and new_lines:
                    base = os.path.splitext(os.path.basename(lf))[0]
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        ip = img_dir / (base + ext)
                        if ip.exists():
                            all_samples.append((str(ip), new_lines))
                            break

    if not all_samples:
        print("No local samples found in datasets directory.")
        return None

    print(f"Total valid samples collected: {len(all_samples)}")

    # Create directory structure
    for s in ['train', 'valid', 'test']:
        os.makedirs(merged_dir / s / 'images', exist_ok=True)
        os.makedirs(merged_dir / s / 'labels', exist_ok=True)

    # Stratified split
    train_s, val_s, test_s = [], [], []
    by_class = defaultdict(list)
    for ip, labels in all_samples:
        cid = int(labels[0].strip().split()[0])
        by_class[FINAL_CLASSES[cid]].append((ip, labels))

    for cls, items in by_class.items():
        if len(items) < 5:
            train_s.extend(items)
            continue
        tv, test = train_test_split(items, test_size=0.1, random_state=42)
        train, val = train_test_split(tv, test_size=0.222, random_state=42)
        train_s.extend(train)
        val_s.extend(val)
        test_s.extend(test)

    def copy_samples(sample_list, split_name):
        for ip, labels in sample_list:
            base = os.path.splitext(os.path.basename(ip))[0]
            ext = os.path.splitext(ip)[1]
            shutil.copy2(ip, merged_dir / split_name / 'images' / (base + ext))
            with open(merged_dir / split_name / 'labels' / (base + '.txt'), 'w') as f:
                f.writelines(labels)

    copy_samples(train_s, 'train')
    copy_samples(val_s, 'valid')
    copy_samples(test_s, 'test')

    yaml_data = {
        'path': str(merged_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(FINAL_CLASSES),
        'names': FINAL_CLASSES
    }
    with open(data_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"Merged dataset created at: {data_yaml_path}")
    return str(data_yaml_path)

def train_yolo(data_yaml, has_gpu, vram):
    """Train YOLOv11 model"""
    print_section("YOLOv11 MODEL TRAINING")
    
    # Model scale selection based on hardware
    if has_gpu:
        if vram >= 12:
            model_name, batch = 'yolov11m.pt', 16
        elif vram >= 8:
            model_name, batch = 'yolov11s.pt', 16
        else:
            model_name, batch = 'yolov11n.pt', 8
    else:
        model_name, batch = 'yolov11n.pt', 4

    print(f"Selected Base Model: {model_name} (Batch Size: {batch})")

    model = YOLO(model_name)
    save_dir = MODEL_DIR / "chira_disease_detection"

    train_config = {
        'data': data_yaml,
        'epochs': 100,
        'batch': batch,
        'imgsz': 640,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'device': 0 if has_gpu else 'cpu',
        'workers': 4 if has_gpu else 0,
        'project': str(MODEL_DIR),
        'name': 'chira_disease_detection',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True
    }

    print("Starting training process...")
    results = model.train(**train_config)
    print("Training finished successfully.")

    # Export ONNX
    try:
        print("Exporting model to ONNX format...")
        model.export(format='onnx', imgsz=640, dynamic=True)
        print("Export completed.")
    except Exception as e:
        print(f"ONNX export skipped: {e}")

    return model

def main():
    print("=" * 65)
    print("  CHIRA - Unified Disease Detection Training Pipeline")
    print("=" * 65)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    has_gpu, vram = check_environment()
    data_yaml = prepare_merged_dataset()

    if data_yaml and os.path.exists(data_yaml):
        train_yolo(data_yaml, has_gpu, vram)
    else:
        print("\nDataset configuration unavailable. Please ensure Roboflow datasets exist in datasets/ directory.")

    print("\n" + "=" * 65)
    print(f"Pipeline Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

if __name__ == "__main__":
    main()
