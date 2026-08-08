"""
CHIRA - Complete Training Pipeline
Integrasi semua dataset + Training YOLOv11
"""

import os
import sys
import glob
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from roboflow import Roboflow

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Config
API_KEY = "dsblUxwiX6gU5Oww0Ufc"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Class mapping
CLASS_MAPPING = {
    # === HEALTHY ===
    'healthy': 'healthy', 'sehat': 'healthy', 'normal': 'healthy', 'Healthy': 'healthy',
    'HEALTHY': 'healthy', 'normal_chicken': 'healthy', 'Normal': 'healthy',
    'chicken': 'healthy',  # dari dataset4 yang cuma label 'chicken'

    # === FOWL POX ===
    'Fowlpox': 'fowl_pox', 'fowlpox': 'fowl_pox', 'fowl_pox': 'fowl_pox', 
    'FowlPox': 'fowl_pox', 'fowl pox': 'fowl_pox', 'Fowl_Pox': 'fowl_pox', 
    'pox': 'fowl_pox', 'Fowl Pox Disease': 'fowl_pox', 'Fowl Poxx Disease': 'fowl_pox',

    # === CORYZA ===
    'Coryza': 'coryza', 'coryza': 'coryza', 'coryza_infectious': 'coryza',
    'Infectious_Coryza': 'coryza', 'coryza_': 'coryza', 'Infectious coryza': 'coryza',

    # === COCCIDIOSIS ===
    'Coccidiosis': 'coccidiosis', 'coccidiosis': 'coccidiosis', 
    'coccidia': 'coccidiosis', 'COCCIDIOSIS': 'coccidiosis', 
    'coccidiosi': 'coccidiosis',

    # === NEWCASTLE DISEASE ===
    'Newcastle': 'newcastle_disease', 'newcastle': 'newcastle_disease',
    'NewcastleDisease': 'newcastle_disease', 'newcastle_disease': 'newcastle_disease',
    'tetelo': 'newcastle_disease', 'NEWCASTLE': 'newcastle_disease',
    'Newcastle_Disease': 'newcastle_disease', 'Newcastle Disease': 'newcastle_disease',

    # === SALMONELLOSIS ===
    'Salmonella': 'salmonellosis', 'salmonella': 'salmonellosis', 
    'salmonellosis': 'salmonellosis', 'salmonela': 'salmonellosis', 
    'SALMONELLA': 'salmonellosis',

    # === AVIAN INFLUENZA ===
    'AvianInfluenza': 'avian_influenza', 'avian_influenza': 'avian_influenza',
    'bird_flu': 'avian_influenza', 'flu_burung': 'avian_influenza',
    'Avian_Influenza': 'avian_influenza', 'avian_flu': 'avian_influenza',
    'Avian Influenza': 'avian_influenza', 'Anemia Virus': 'avian_influenza',

    # === GUMBORO (IBD) ===
    'Gumboro': 'gumboro', 'gumboro': 'gumboro', 'IBD': 'gumboro',
    'infectious_bursal_disease': 'gumboro', 'GUMBORO': 'gumboro',

    # === MAREK'S DISEASE ===
    'Mareks': 'mareks_disease', 'mareks': 'mareks_disease',
    'mareks_disease': 'mareks_disease', "Marek's": 'mareks_disease',
    'Marek': 'mareks_disease', 'Mareks Disease': 'mareks_disease',

    # === INFECTIOUS BRONCHITIS ===
    'InfectiousBronchitis': 'infectious_bronchitis', 
    'infectious_bronchitis': 'infectious_bronchitis',
    'bronchitis': 'infectious_bronchitis', 'IB': 'infectious_bronchitis',
    'Chronic Respiratory disease': 'infectious_bronchitis',

    # === ASPERGILLOSIS ===
    'Aspergillosis': 'aspergillosis', 'aspergillosis': 'aspergillosis',
    'aspergillosi': 'aspergillosis',

    # === BOTULISM ===
    'Botulism': 'botulism', 'botulism': 'botulism', 'BOTULISM': 'botulism',
    'Botulism Disease': 'botulism',

    # === EXTERNAL PARASITES ===
    'ExternalParasites': 'external_parasites', 'external_parasites': 'external_parasites',
    'parasites': 'external_parasites', 'mite': 'external_parasites', 
    'lice': 'external_parasites', 'external_parasite': 'external_parasites', 
    'Parasites': 'external_parasites', 'Bumble Foot': 'external_parasites',
    'Vent Gleet': 'external_parasites', 'Thrush Disease': 'external_parasites',

    # === EGG PERITONITIS ===
    'EggPeritonitis': 'egg_peritonitis', 'egg_peritonitis': 'egg_peritonitis',
    'peritonitis': 'egg_peritonitis', 'Egg_Peritonitis': 'egg_peritonitis',

    # === SICK (UNSPECIFIED) ===
    'sick': 'sick_unspecified', 'Sick': 'sick_unspecified', 
    'sakit': 'sick_unspecified', 'diseased': 'sick_unspecified', 
    'unhealthy': 'sick_unspecified', 'SICK': 'sick_unspecified', 
    'disease': 'sick_unspecified', 'Fowl Cholera': 'sick_unspecified',
}

FINAL_CLASSES = [
    'healthy', 'fowl_pox', 'coryza', 'coccidiosis', 'newcastle_disease',
    'salmonellosis', 'avian_influenza', 'gumboro', 'mareks_disease',
    'infectious_bronchitis', 'aspergillosis', 'botulism',
    'external_parasites', 'egg_peritonitis', 'sick_unspecified'
]

def normalize_class_name(name):
    """Normalisasi nama class dari berbagai format dataset"""
    name = name.strip()
    
    # Coba exact match dulu
    if name in CLASS_MAPPING:
        return CLASS_MAPPING[name]
    
    # Coba lowercase
    lower = name.lower()
    if lower in CLASS_MAPPING:
        return CLASS_MAPPING[lower]
    
    # Normalisasi: spasi → underscore, hapus karakter aneh
    normalized = lower.replace(' ', '_').replace('-', '_')
    if normalized in CLASS_MAPPING:
        return CLASS_MAPPING[normalized]
    
    # Coba tanpa kata "disease" / "virus"
    simplified = normalized.replace('_disease', '').replace('_virus', '')
    if simplified in CLASS_MAPPING:
        return CLASS_MAPPING[simplified]
    
    return None

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def find_model():
    """Cari model YOLO yang tersedia"""
    for model_name in ['yolov11m.pt', 'yolov11s.pt', 'yolov11n.pt', 'yolov8m.pt', 'yolov8s.pt', 'yolov8n.pt']:
        if os.path.exists(model_name):
            return model_name
    return None

def download_datasets():
    """Download semua dataset yang tersedia"""
    print_section("STEP 1: DOWNLOADING DATASETS")
    
    rf = Roboflow(api_key=API_KEY)
    downloaded = {}
    
    datasets_to_try = [
        ("dataset1", "datasets-oiryj", "chicken-diseases-with-roboflow-model-c58n3", [4, 3, 2, 1]),
        ("dataset2", "coding-hysjm", "chicken-disease-detection-ncfow", [3, 2, 1, 4, 5]),
        ("dataset3", "coding-hysjm", "chicken-desease", [3, 2, 1, 4, 5]),
        ("dataset4", "chicken-amvkm", "sick-and-healthy-chicken", [3, 2, 1, 4, 5]),
    ]
    
    for name, workspace, project, versions in datasets_to_try:
        print(f"\n📁 Trying: {name} ({workspace}/{project})")
        
        for version in versions:
            try:
                proj = rf.workspace(workspace).project(project)
                location = os.path.join(DATASET_DIR, f"{name}_v{version}")
                
                if os.path.exists(location):
                    shutil.rmtree(location)
                
                dataset = proj.version(version).download("yolov11", location=location)
                
                # Verify
                yaml_path = os.path.join(location, "data.yaml")
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        config = yaml.safe_load(f)
                    classes = config.get('names', [])
                    
                    # Count images
                    total = 0
                    for split in ['train', 'valid', 'test']:
                        img_dir = os.path.join(location, split, 'images')
                        if os.path.exists(img_dir):
                            total += len(glob.glob(os.path.join(img_dir, '*.*')))
                    
                    print(f"   ✅ SUCCESS! Version {version}")
                    print(f"      Classes: {classes}")
                    print(f"      Images: {total}")
                    downloaded[name] = location
                    break
                    
            except Exception as e:
                error = str(e)
                if "not found" in error.lower():
                    print(f"   ⚠️  Version {version} not found")
                elif "invalid format" in error.lower():
                    print(f"   ⚠️  Invalid format for version {version}")
                else:
                    print(f"   ⚠️  Error: {error[:80]}")
                continue
    
    print(f"\n✅ Downloaded {len(downloaded)}/{len(datasets_to_try)} datasets")
    return downloaded

def merge_datasets():
    """Merge semua dataset yang didownload"""
    print_section("STEP 2: MERGING & HARMONIZING DATASETS")
    
    all_samples = []
    class_dist = {cls: 0 for cls in FINAL_CLASSES}
    unmapped = set()
    
    # Find all downloaded datasets
    for item in os.listdir(DATASET_DIR):
        item_path = os.path.join(DATASET_DIR, item)
        if not os.path.isdir(item_path) or not item.startswith('dataset'):
            continue
        
        yaml_path = os.path.join(item_path, "data.yaml")
        if not os.path.exists(yaml_path):
            continue
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        orig_classes = config.get('names', [])
        print(f"\n🔄 Processing: {item}")
        print(f"   Classes: {orig_classes}")
        
        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(item_path, split, 'images')
            lbl_dir = os.path.join(item_path, split, 'labels')
            
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
                    mapped = normalize_class_name(orig_name)
                    
                    if not mapped:
                        unmapped.add(orig_name)
                        valid = False
                        break
                    
                    new_id = FINAL_CLASSES.index(mapped)
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                    class_dist[mapped] += 1
                
                if valid and new_lines:
                    base = os.path.splitext(os.path.basename(lf))[0]
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        ip = os.path.join(img_dir, base + ext)
                        if os.path.exists(ip):
                            all_samples.append((ip, new_lines, split))
                            break
    
    print(f"\n📊 Total samples: {len(all_samples)}")
    print(f"📊 Class distribution:")
    for cls, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cls:25s}: {count:5d}")
    
    if unmapped:
        print(f"⚠️  Unmapped (skipped): {sorted(unmapped)}")
    
    return all_samples
    

def create_dataset(samples):
    """Create stratified dataset"""
    print_section("STEP 3: CREATING STRATIFIED DATASET")
    
    merged_dir = os.path.join(DATASET_DIR, "merged_disease_dataset")
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    
    for s in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(merged_dir, s, 'images'), exist_ok=True)
        os.makedirs(os.path.join(merged_dir, s, 'labels'), exist_ok=True)
    
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
        print(f"   {cls:25s}: train={len(train):4d}, val={len(val):4d}, test={len(test):4d}")
    
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
    
    print(f"\n✅ Dataset: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")
    return os.path.join(merged_dir, 'data.yaml')

def train(data_yaml):
    """Train YOLOv11"""
    print_section("STEP 4: TRAINING YOLOv11")
    
    # Find model
    model_file = find_model()
    if not model_file:
        print("❌ No YOLO model found!")
        print("💡 Download from: https://github.com/ultralytics/assets/releases")
        return
    
    print(f"✅ Model: {model_file}")
    
    # Check GPU
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
        if vram >= 12:
            batch = 16
        elif vram >= 8:
            batch = 16
        elif vram >= 6:
            batch = 16
        else:
            batch = 8
    else:
        print("⚠️  No GPU, using CPU (very slow)")
        batch = 4
    
    print(f"📦 Batch: {batch}")
    
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
        'workers': 4,
        'project': MODEL_DIR,
        'name': 'chira_disease_detection',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True
    }
    
    print("\n⏳ Starting training... (Press Ctrl+C to stop)")
    print("=" * 70)
    
    try:
        results = model.train(**config)
        print("\n✅ Training complete!")
        
        if hasattr(results, 'results_dict'):
            print("\n📊 Results:")
            for k, v in results.results_dict.items():
                print(f"   {k}: {v:.4f}")
        
        # Export
        print("\n🔄 Exporting to ONNX...")
        model.export(format='onnx', imgsz=640, dynamic=True)
        print("✅ Export complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")

def main():
    print("=" * 70)
    print(" CHIRA - Complete Training Pipeline")
    print("=" * 70)
    print(f" Start: {datetime.now()}")
    
    # Step 1: Download
    download_datasets()
    
    # Step 2: Merge
    samples = merge_datasets()
    
    if not samples:
        print("\n❌ No samples available!")
        return
    
    # Step 3: Create dataset
    data_yaml = create_dataset(samples)
    
    # Step 4: Train
    train(data_yaml)
    
    print("\n" + "=" * 70)
    print(" COMPLETE!")
    print(f" End: {datetime.now()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
