"""
CHIRA - Chicken Health Identification and Recommendation Assistant
Training Script for Disease Detection using YOLOv11

Author: CHIRA Development Team
Date: 2026
Purpose: Train AI model to detect chicken diseases from body images

Datasets Used:
1. chicken-diseases-with-roboflow-model-c58n3 (Roboflow)
2. chicken-disease-detection-ncfow (Roboflow)
3. chicken-desease (Roboflow)
4. sick-and-healthy-chicken (Roboflow)

Target Metrics:
- mAP50: > 85%
- mAP50-95: > 70%
- Precision: > 80%
- Recall: > 80%
- F1-Score: > 80%
"""

import os
import sys
import yaml
import glob
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from ultralytics import YOLO
from roboflow import Roboflow

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration
API_KEY = "dsblUxwiX6gU5Oww0Ufc"
BASE_DIR = "/mnt/agents/output/chira"
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Class mapping for harmonization
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

FINAL_CLASSES = [
    'healthy', 'fowl_pox', 'coryza', 'coccidiosis', 'newcastle_disease',
    'salmonellosis', 'avian_influenza', 'gumboro', 'mareks_disease',
    'infectious_bronchitis', 'aspergillosis', 'botulism', 
    'external_parasites', 'egg_peritonitis', 'sick_unspecified'
]

DATASETS = {
    "dataset1_diseases": {"workspace": "datasets-oiryj", "project": "chicken-diseases-with-roboflow-model-c58n3", "version": 4},
    "dataset2_disease_detection": {"workspace": "coding-hysjm", "project": "chicken-disease-detection-ncfow", "version": 2},
    "dataset3_desease": {"workspace": "coding-hysjm", "project": "chicken-desease", "version": 2},
    "dataset4_sick_healthy": {"workspace": "chicken-amvkm", "project": "sick-and-healthy-chicken", "version": 2},
}


def print_section(title):
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def download_datasets():
    """Download all datasets from Roboflow"""
    print_section("DOWNLOADING DATASETS FROM ROBOFLOW")
    
    rf = Roboflow(api_key=API_KEY)
    downloaded = {}
    
    for name, config in DATASETS.items():
        print(f"\nDownloading: {name}")
        try:
            project = rf.workspace(config['workspace']).project(config['project'])
            dataset = project.version(config['version']).download("yolov11", location=os.path.join(DATASET_DIR, name))
            downloaded[name] = dataset
            print(f"  Success: {os.path.join(DATASET_DIR, name)}")
        except Exception as e:
            print(f"  Error: {e}")
            downloaded[name] = None
    
    return downloaded


def analyze_dataset(dataset_path):
    """Analyze YOLO dataset structure"""
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(yaml_path):
        return None
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    classes = config.get('names', [])
    split_stats = {}
    class_counts = {cls: 0 for cls in classes}
    
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(labels_dir):
            files = glob.glob(os.path.join(labels_dir, '*.txt'))
            split_stats[split] = len(files)
            for lf in files:
                with open(lf, 'r') as f:
                    for line in f:
                        cid = int(line.strip().split()[0])
                        if cid < len(classes):
                            class_counts[classes[cid]] += 1
        else:
            split_stats[split] = 0
    
    return {
        'classes': classes,
        'split_stats': split_stats,
        'class_counts': class_counts,
        'total': sum(split_stats.values())
    }


def merge_datasets():
    """Merge and harmonize all datasets"""
    print_section("MERGING & HARMONIZING DATASETS")
    
    all_samples = []
    
    for name, config in DATASETS.items():
        path = os.path.join(DATASET_DIR, name)
        if not os.path.exists(path):
            continue
        
        analysis = analyze_dataset(path)
        if not analysis:
            continue
        
        orig_classes = analysis['classes']
        print(f"\nProcessing: {name} ({analysis['total']} images)")
        print(f"  Original classes: {orig_classes}")
        
        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(path, split, 'images')
            lbl_dir = os.path.join(path, split, 'labels')
            
            if not os.path.exists(lbl_dir):
                continue
            
            for lf in glob.glob(os.path.join(lbl_dir, '*.txt')):
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
                        print(f"    Unmapped: {orig_name}")
                        valid = False
                        break
                    
                    new_id = FINAL_CLASSES.index(mapped)
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                
                if valid and new_lines:
                    base = os.path.splitext(os.path.basename(lf))[0]
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        ip = os.path.join(img_dir, base + ext)
                        if os.path.exists(ip):
                            all_samples.append((ip, new_lines, split))
                            break
    
    print(f"\nTotal valid samples: {len(all_samples)}")
    return all_samples


def create_merged_dataset(samples):
    """Create stratified merged dataset"""
    print_section("CREATING STRATIFIED MERGED DATASET")
    
    merged_dir = os.path.join(DATASET_DIR, "merged_disease_dataset")
    for s in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(merged_dir, s, 'images'), exist_ok=True)
        os.makedirs(os.path.join(merged_dir, s, 'labels'), exist_ok=True)
    
    # Group by class
    by_class = defaultdict(list)
    for ip, labels, _ in samples:
        cid = int(labels[0].strip().split()[0])
        by_class[FINAL_CLASSES[cid]].append((ip, labels))
    
    train_s, val_s, test_s = [], [], []
    
    for cls, items in by_class.items():
        if len(items) < 5:
            train_s.extend(items)
            continue
        
        tv, test = train_test_split(items, test_size=0.1, random_state=42)
        train, val = train_test_split(tv, test_size=0.222, random_state=42)
        train_s.extend(train)
        val_s.extend(val)
        test_s.extend(test)
        print(f"  {cls:25s}: train={len(train):3d}, val={len(val):3d}, test={len(test):3d}")
    
    # Copy files
    def copy(sample_list, split_name):
        for ip, labels in sample_list:
            base = os.path.splitext(os.path.basename(ip))[0]
            ext = os.path.splitext(ip)[1]
            shutil.copy2(ip, os.path.join(merged_dir, split_name, 'images', base + ext))
            with open(os.path.join(merged_dir, split_name, 'labels', base + '.txt'), 'w') as f:
                f.writelines(labels)
    
    copy(train_s, 'train')
    copy(val_s, 'valid')
    copy(test_s, 'test')
    
    # Create data.yaml
    yaml_data = {
        'path': merged_dir,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(FINAL_CLASSES),
        'names': FINAL_CLASSES
    }
    with open(os.path.join(merged_dir, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    print(f"\nMerged dataset created:")
    print(f"  Train: {len(train_s)}")
    print(f"  Valid: {len(val_s)}")
    print(f"  Test:  {len(test_s)}")
    
    return merged_dir


def train_model(data_yaml):
    """Train YOLOv11 model"""
    print_section("TRAINING YOLOv11 MODEL")
    
    # Select model based on VRAM
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU VRAM: {vram:.1f} GB")
        
        if vram >= 16:
            model_size, batch = 'yolov11x.pt', 16
        elif vram >= 12:
            model_size, batch = 'yolov11l.pt', 16
        elif vram >= 8:
            model_size, batch = 'yolov11m.pt', 16
        elif vram >= 6:
            model_size, batch = 'yolov11s.pt', 16
        else:
            model_size, batch = 'yolov11n.pt', 8
    else:
        print("WARNING: No GPU detected. Using CPU (very slow)")
        model_size, batch = 'yolov11n.pt', 4
    
    print(f"Model: {model_size}, Batch: {batch}")
    
    # Auto-detect available model
    import glob
    available_models = glob.glob('yolov11*.pt') + glob.glob('yolov8*.pt')
    model_file = available_models[0] if available_models else model_size
    print(f"Using model: {model_file}")
    model = YOLO(model_file)
    
    config = {
        'data': data_yaml,
        'epochs': 100,
        'batch': batch,
        'imgsz': 640,
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'project': MODEL_DIR,
        'name': 'chira_disease_detection',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True
    }
    
    print("\nStarting training...")
    results = model.train(**config)
    
    return results, model


def evaluate_model(model, data_yaml):
    """Evaluate trained model"""
    print_section("EVALUATING MODEL")
    
    metrics = model.val(data=data_yaml, split='test')
    
    print(f"\nFinal Metrics:")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    
    return metrics


def export_model(model):
    """Export model to ONNX format"""
    print_section("EXPORTING MODEL")
    
    onnx_path = os.path.join(MODEL_DIR, 'chira_disease.onnx')
    model.export(format='onnx', imgsz=640, dynamic=True)
    print(f"Model exported to: {onnx_path}")
    
    return onnx_path


def plot_training_results():
    """Plot training curves"""
    results_csv = os.path.join(MODEL_DIR, 'chira_disease_detection', 'results.csv')
    
    if not os.path.exists(results_csv):
        print("Results CSV not found")
        return
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    # Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0,0].plot(df['epoch'], df['train/box_loss'], label='Train', color='#3498db')
    axes[0,0].plot(df['epoch'], df['val/box_loss'], label='Val', color='#e74c3c')
    axes[0,0].set_title('Box Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(df['epoch'], df['train/cls_loss'], label='Train', color='#3498db')
    axes[0,1].plot(df['epoch'], df['val/cls_loss'], label='Val', color='#e74c3c')
    axes[0,1].set_title('Classification Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(df['epoch'], df['metrics/mAP50(B)'], color='#2ecc71', linewidth=2)
    axes[1,0].set_title('mAP@50')
    axes[1,0].set_ylim([0, 1])
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(df['epoch'], df['metrics/mAP50-95(B)'], color='#e67e22', linewidth=2)
    axes[1,1].set_title('mAP@50-95')
    axes[1,1].set_ylim([0, 1])
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('CHIRA Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Training curves saved to: {os.path.join(RESULTS_DIR, 'training_curves.png')}")


def main():
    print("=" * 60)
    print(" CHIRA - Disease Detection Training Pipeline")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    
    # Step 1: Download datasets
    download_datasets()
    
    # Step 2: Merge datasets
    samples = merge_datasets()
    
    # Step 3: Create merged dataset
    data_yaml = os.path.join(create_merged_dataset(samples), 'data.yaml')
    
    # Step 4: Train model
    results, model = train_model(data_yaml)
    
    # Step 5: Evaluate
    metrics = evaluate_model(model, data_yaml)
    
    # Step 6: Export
    export_model(model)
    
    # Step 7: Plot results
    plot_training_results()
    
    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE!")
    print(f" End time: {datetime.now()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
