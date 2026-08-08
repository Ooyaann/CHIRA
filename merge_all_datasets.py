"""
CHIRA - Merge All Downloaded Datasets
Integrasi semua dataset yang berhasil didownload
"""

import os
import sys
import yaml
import glob
import shutil
import random
from collections import defaultdict, Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

# Class mapping untuk harmonisasi
CLASS_MAPPING = {
    'healthy': 'healthy', 'sehat': 'healthy', 'normal': 'healthy', 'Healthy': 'healthy',
    'HEALTHY': 'healthy', 'normal_chicken': 'healthy', 'Normal': 'healthy',
    
    'Fowlpox': 'fowl_pox', 'fowlpox': 'fowl_pox', 'fowl_pox': 'fowl_pox', 'FowlPox': 'fowl_pox',
    'fowl pox': 'fowl_pox', 'Fowl_Pox': 'fowl_pox', 'pox': 'fowl_pox',
    
    'Coryza': 'coryza', 'coryza': 'coryza', 'coryza_infectious': 'coryza',
    'Infectious_Coryza': 'coryza', 'coryza_': 'coryza',
    
    'Coccidiosis': 'coccidiosis', 'coccidiosis': 'coccidiosis', 'coccidia': 'coccidiosis',
    'COCCIDIOSIS': 'coccidiosis', 'coccidiosi': 'coccidiosis',
    
    'Newcastle': 'newcastle_disease', 'newcastle': 'newcastle_disease',
    'NewcastleDisease': 'newcastle_disease', 'newcastle_disease': 'newcastle_disease',
    'tetelo': 'newcastle_disease', 'NEWCASTLE': 'newcastle_disease',
    'Newcastle_Disease': 'newcastle_disease',
    
    'Salmonella': 'salmonellosis', 'salmonella': 'salmonellosis', 'salmonellosis': 'salmonellosis',
    'salmonela': 'salmonellosis', 'SALMONELLA': 'salmonellosis',
    
    'AvianInfluenza': 'avian_influenza', 'avian_influenza': 'avian_influenza',
    'bird_flu': 'avian_influenza', 'flu_burung': 'avian_influenza',
    'Avian_Influenza': 'avian_influenza', 'avian_flu': 'avian_influenza',
    
    'Gumboro': 'gumboro', 'gumboro': 'gumboro', 'IBD': 'gumboro',
    'infectious_bursal_disease': 'gumboro', 'GUMBORO': 'gumboro',
    
    'Mareks': 'mareks_disease', 'mareks': 'mareks_disease',
    'mareks_disease': 'mareks_disease', "Marek's": 'mareks_disease',
    'Marek': 'mareks_disease',
    
    'InfectiousBronchitis': 'infectious_bronchitis', 'infectious_bronchitis': 'infectious_bronchitis',
    'bronchitis': 'infectious_bronchitis', 'IB': 'infectious_bronchitis',
    
    'Aspergillosis': 'aspergillosis', 'aspergillosis': 'aspergillosis',
    'aspergillosi': 'aspergillosis',
    
    'Botulism': 'botulism', 'botulism': 'botulism', 'BOTULISM': 'botulism',
    
    'ExternalParasites': 'external_parasites', 'external_parasites': 'external_parasites',
    'parasites': 'external_parasites', 'mite': 'external_parasites', 'lice': 'external_parasites',
    'external_parasite': 'external_parasites', 'Parasites': 'external_parasites',
    
    'EggPeritonitis': 'egg_peritonitis', 'egg_peritonitis': 'egg_peritonitis',
    'peritonitis': 'egg_peritonitis', 'Egg_Peritonitis': 'egg_peritonitis',
    
    'sick': 'sick_unspecified', 'Sick': 'sick_unspecified', 'sakit': 'sick_unspecified',
    'diseased': 'sick_unspecified', 'unhealthy': 'sick_unspecified',
    'SICK': 'sick_unspecified', 'disease': 'sick_unspecified',
}

FINAL_CLASSES = [
    'healthy', 'fowl_pox', 'coryza', 'coccidiosis', 'newcastle_disease',
    'salmonellosis', 'avian_influenza', 'gumboro', 'mareks_disease',
    'infectious_bronchitis', 'aspergillosis', 'botulism',
    'external_parasites', 'egg_peritonitis', 'sick_unspecified'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def find_all_datasets():
    """Cari semua folder dataset yang tersedia"""
    datasets = {}
    
    for item in os.listdir(DATASET_DIR):
        item_path = os.path.join(DATASET_DIR, item)
        if not os.path.isdir(item_path):
            continue
        
        yaml_path = os.path.join(item_path, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            classes = config.get('names', [])
            datasets[item] = {
                'path': item_path,
                'classes': classes,
                'yaml': yaml_path
            }
    
    return datasets

def analyze_dataset(dataset_path):
    """Analisis struktur dataset YOLO"""
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if not os.path.exists(yaml_path):
        return None
    
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    classes = data_config.get('names', [])
    num_classes = len(classes)
    
    split_stats = {}
    class_counts = {cls: 0 for cls in classes}
    total_images = 0
    
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        images_dir = os.path.join(dataset_path, split, 'images')
        
        if os.path.exists(labels_dir):
            label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
            split_stats[split] = len(label_files)
            total_images += len(label_files)
            
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id < len(classes):
                            class_counts[classes[class_id]] += 1
        else:
            split_stats[split] = 0
    
    return {
        'classes': classes,
        'num_classes': num_classes,
        'split_stats': split_stats,
        'class_counts': class_counts,
        'total_images': total_images
    }

def merge_all_datasets():
    """Merge semua dataset yang tersedia"""
    print_section("FINDING ALL DATASETS")
    
    all_datasets = find_all_datasets()
    
    if not all_datasets:
        print("❌ No datasets found!")
        return None
    
    print(f"\n📁 Found {len(all_datasets)} dataset(s):")
    for name, info in all_datasets.items():
        print(f"   • {name}: {info['classes']}")
    
    print_section("ANALYZING DATASETS")
    
    dataset_analyses = {}
    for name, info in all_datasets.items():
        analysis = analyze_dataset(info['path'])
        if analysis:
            dataset_analyses[name] = analysis
            print(f"\n📊 {name}:")
            print(f"   Classes: {analysis['classes']}")
            print(f"   Total images: {analysis['total_images']}")
            print(f"   Class distribution: {dict(analysis['class_counts'])}")
    
    print_section("MERGING & HARMONIZING ALL DATASETS")
    
    all_samples = []
    class_distribution = {cls: 0 for cls in FINAL_CLASSES}
    unmapped_classes = set()
    
    for dataset_name, analysis in dataset_analyses.items():
        dataset_path = analysis  # We need the path, fix this
        break
    
    # Get paths from all_datasets
    for dataset_name, info in all_datasets.items():
        if dataset_name not in dataset_analyses:
            continue
        
        dataset_path = info['path']
        orig_classes = info['classes']
        
        print(f"\n🔄 Processing: {dataset_name}")
        print(f"   Original classes: {orig_classes}")
        
        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(dataset_path, split, 'images')
            lbl_dir = os.path.join(dataset_path, split, 'labels')
            
            if not os.path.exists(lbl_dir):
                continue
            
            label_files = glob.glob(os.path.join(lbl_dir, '*.txt'))
            
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    continue
                
                new_lines = []
                valid = True
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    orig_class_id = int(parts[0])
                    if orig_class_id >= len(orig_classes):
                        valid = False
                        break
                    
                    orig_class_name = orig_classes[orig_class_id]
                    
                    # Map ke kelas baru
                    mapped = CLASS_MAPPING.get(orig_class_name)
                    if not mapped:
                        mapped = CLASS_MAPPING.get(orig_class_name.lower())
                    
                    if not mapped:
                        unmapped_classes.add(orig_class_name)
                        valid = False
                        break
                    
                    new_class_id = FINAL_CLASSES.index(mapped)
                    new_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
                    class_distribution[mapped] += 1
                
                if valid and new_lines:
                    base = os.path.splitext(os.path.basename(label_file))[0]
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        ip = os.path.join(img_dir, base + ext)
                        if os.path.exists(ip):
                            all_samples.append((ip, new_lines, split))
                            break
    
    print(f"\n📊 Total valid samples: {len(all_samples)}")
    print(f"\n📊 Class distribution:")
    for cls, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cls:25s}: {count:5d} instances")
    
    if unmapped_classes:
        print(f"\n⚠️  Unmapped classes (skipped): {sorted(unmapped_classes)}")
    
    return all_samples, class_distribution

def create_stratified_split(samples, class_dist):
    """Buat stratified split dari semua sampel"""
    print_section("CREATING STRATIFIED SPLIT")
    
    merged_dir = os.path.join(DATASET_DIR, "merged_disease_dataset")
    
    # Bersihkan direktori lama
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    
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
            print(f"   ⚠️  {cls}: only {len(items)} samples, all go to train")
            train_s.extend(items)
            continue
        
        # Stratified split: 70% train, 20% val, 10% test
        tv, test = train_test_split(items, test_size=0.1, random_state=42)
        train, val = train_test_split(tv, test_size=0.222, random_state=42)
        
        train_s.extend(train)
        val_s.extend(val)
        test_s.extend(test)
        print(f"   ✅ {cls:25s}: train={len(train):4d}, val={len(val):4d}, test={len(test):4d}")
    
    # Copy files
    def copy_samples(sample_list, split_name):
        for ip, labels in sample_list:
            base = os.path.splitext(os.path.basename(ip))[0]
            ext = os.path.splitext(ip)[1]
            shutil.copy2(ip, os.path.join(merged_dir, split_name, 'images', base + ext))
            with open(os.path.join(merged_dir, split_name, 'labels', base + '.txt'), 'w') as f:
                f.writelines(labels)
    
    copy_samples(train_s, 'train')
    copy_samples(val_s, 'valid')
    copy_samples(test_s, 'test')
    
    # Create data.yaml
    yaml_data = {
        'path': merged_dir,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(FINAL_CLASSES),
        'names': FINAL_CLASSES
    }
    
    yaml_path = os.path.join(merged_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    print(f"\n✅ Merged dataset created:")
    print(f"   Train: {len(train_s)}")
    print(f"   Valid: {len(val_s)}")
    print(f"   Test:  {len(test_s)}")
    print(f"   Total: {len(train_s) + len(val_s) + len(test_s)}")
    print(f"\n📄 data.yaml: {yaml_path}")
    
    return merged_dir

def main():
    print("=" * 70)
    print(" CHIRA - Complete Dataset Integration")
    print("=" * 70)
    
    samples, class_dist = merge_all_datasets()
    
    if not samples:
        print("\n❌ No samples to merge!")
        return
    
    merged_dir = create_stratified_split(samples, class_dist)
    
    print("\n" + "=" * 70)
    print(" ✅ DATASET INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"\n🚀 Ready for training!")
    print(f"   python train_disease.py")

if __name__ == "__main__":
    main()
