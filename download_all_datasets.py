"""
CHIRA - Download All Datasets with Version Detection
Script ini akan mencoba semua versi dataset yang tersedia
"""

import os
import sys
import yaml
from roboflow import Roboflow

API_KEY = "dsblUxwiX6gU5Oww0Ufc"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

os.makedirs(DATASET_DIR, exist_ok=True)

# Daftar dataset dengan multiple versi yang akan dicoba
DATASETS_CONFIG = {
    "dataset1_diseases": {
        "workspace": "datasets-oiryj",
        "project": "chicken-diseases-with-roboflow-model-c58n3",
        "versions": [4, 3, 2, 1],  # Coba dari versi terbaru
        "format": "yolov11",
        "description": "CHICKEN DISEASES - With Roboflow Model"
    },
    "dataset2_disease_detection": {
        "workspace": "coding-hysjm",
        "project": "chicken-disease-detection-ncfow",
        "versions": [3, 2, 1, 4, 5],  # Coba semua versi
        "format": "yolov11",
        "description": "Chicken Disease Detection"
    },
    "dataset3_desease": {
        "workspace": "coding-hysjm",
        "project": "chicken-desease",
        "versions": [3, 2, 1, 4, 5],  # Coba semua versi
        "format": "yolov11",  # Coba yolov11 dulu
        "fallback_formats": ["yolov8", "yolov5"],
        "description": "Chicken Desease"
    },
    "dataset4_sick_healthy": {
        "workspace": "chicken-amvkm",
        "project": "sick-and-healthy-chicken",
        "versions": [3, 2, 1, 4, 5],  # Coba semua versi
        "format": "yolov11",
        "description": "Sick and Healthy Chicken"
    },
    # Dataset tambahan untuk perilaku
    "dataset5_behavior": {
        "workspace": "chicken-amvkm",
        "project": "sick-and-healthy-chicken",
        "versions": [2, 1, 3],
        "format": "yolov11",
        "description": "Behavior Dataset"
    }
}

# Mapping kelas untuk harmonisasi
CLASS_MAPPING = {
    # healthy variants
    'healthy': 'healthy', 'sehat': 'healthy', 'normal': 'healthy', 'Healthy': 'healthy',
    'HEALTHY': 'healthy', 'normal_chicken': 'healthy', 'Normal': 'healthy',
    
    # Fowl pox variants
    'Fowlpox': 'fowl_pox', 'fowlpox': 'fowl_pox', 'fowl_pox': 'fowl_pox', 'FowlPox': 'fowl_pox',
    'fowl pox': 'fowl_pox', 'Fowl_Pox': 'fowl_pox', 'pox': 'fowl_pox',
    
    # Coryza variants
    'Coryza': 'coryza', 'coryza': 'coryza', 'coryza_infectious': 'coryza',
    'Infectious_Coryza': 'coryza', 'coryza_': 'coryza',
    
    # Coccidiosis variants
    'Coccidiosis': 'coccidiosis', 'coccidiosis': 'coccidiosis', 'coccidia': 'coccidiosis',
    'COCCIDIOSIS': 'coccidiosis', 'coccidiosi': 'coccidiosis',
    
    # Newcastle disease variants
    'Newcastle': 'newcastle_disease', 'newcastle': 'newcastle_disease', 
    'NewcastleDisease': 'newcastle_disease', 'newcastle_disease': 'newcastle_disease',
    'tetelo': 'newcastle_disease', 'NEWCASTLE': 'newcastle_disease',
    'Newcastle_Disease': 'newcastle_disease',
    
    # Salmonella variants
    'Salmonella': 'salmonellosis', 'salmonella': 'salmonellosis', 'salmonellosis': 'salmonellosis',
    'salmonela': 'salmonellosis', 'SALMONELLA': 'salmonellosis',
    
    # Avian Influenza variants
    'AvianInfluenza': 'avian_influenza', 'avian_influenza': 'avian_influenza',
    'bird_flu': 'avian_influenza', 'flu_burung': 'avian_influenza',
    'Avian_Influenza': 'avian_influenza', 'avian_flu': 'avian_influenza',
    
    # Gumboro variants
    'Gumboro': 'gumboro', 'gumboro': 'gumboro', 'IBD': 'gumboro',
    'infectious_bursal_disease': 'gumboro', 'GUMBORO': 'gumboro',
    
    # Marek's disease variants
    'Mareks': 'mareks_disease', 'mareks': 'mareks_disease',
    'mareks_disease': 'mareks_disease', "Marek's": 'mareks_disease',
    'Marek': 'mareks_disease',
    
    # Infectious Bronchitis variants
    'InfectiousBronchitis': 'infectious_bronchitis', 'infectious_bronchitis': 'infectious_bronchitis',
    'bronchitis': 'infectious_bronchitis', 'IB': 'infectious_bronchitis',
    
    # Aspergillosis variants
    'Aspergillosis': 'aspergillosis', 'aspergillosis': 'aspergillosis',
    'aspergillosi': 'aspergillosis',
    
    # Botulism variants
    'Botulism': 'botulism', 'botulism': 'botulism', 'BOTULISM': 'botulism',
    
    # External parasites variants
    'ExternalParasites': 'external_parasites', 'external_parasites': 'external_parasites',
    'parasites': 'external_parasites', 'mite': 'external_parasites', 'lice': 'external_parasites',
    'external_parasite': 'external_parasites', 'Parasites': 'external_parasites',
    
    # Egg peritonitis variants
    'EggPeritonitis': 'egg_peritonitis', 'egg_peritonitis': 'egg_peritonitis',
    'peritonitis': 'egg_peritonitis', 'Egg_Peritonitis': 'egg_peritonitis',
    
    # Unspecified sick
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

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def try_download_dataset(rf, name, config):
    """Coba download dataset dengan berbagai versi dan format"""
    print(f"\n📁 Dataset: {name}")
    print(f"   Project: {config['workspace']}/{config['project']}")
    
    try:
        project = rf.workspace(config['workspace']).project(config['project'])
        
        # Cek versi yang tersedia
        versions = project.versions()
        available_versions = [int(v.version) for v in versions]
        print(f"   Available versions: {available_versions}")
        
        # Coba setiap versi yang dikonfigurasi
        for version_num in config['versions']:
            if version_num not in available_versions:
                continue
                
            print(f"\n   🔄 Trying version {version_num}...")
            
            # Coba format utama
            formats_to_try = [config['format']]
            if 'fallback_formats' in config:
                formats_to_try.extend(config['fallback_formats'])
            
            for fmt in formats_to_try:
                try:
                    location = os.path.join(DATASET_DIR, f"{name}_v{version_num}")
                    
                    # Hapus folder lama kalau ada
                    if os.path.exists(location):
                        import shutil
                        shutil.rmtree(location)
                    
                    dataset = project.version(version_num).download(fmt, location=location)
                    
                    # Cek apakah dataset valid
                    yaml_path = os.path.join(location, "data.yaml")
                    if os.path.exists(yaml_path):
                        with open(yaml_path, 'r') as f:
                            data_config = yaml.safe_load(f)
                        
                        classes = data_config.get('names', [])
                        num_images = sum(
                            len(os.listdir(os.path.join(location, split, 'images'))) 
                            if os.path.exists(os.path.join(location, split, 'images')) else 0
                            for split in ['train', 'valid', 'test']
                        )
                        
                        print(f"   ✅ SUCCESS! Format: {fmt}")
                        print(f"      Classes: {classes}")
                        print(f"      Images: ~{num_images}")
                        return location, classes
                        
                except Exception as e:
                    error_msg = str(e)
                    if "invalid format" in error_msg.lower():
                        print(f"      ⚠️  Format {fmt} not supported, trying next...")
                    elif "not found" in error_msg.lower():
                        print(f"      ⚠️  Version {version_num} not found")
                        break
                    else:
                        print(f"      ⚠️  Error with {fmt}: {error_msg[:100]}")
                    continue
        
        print(f"   ❌ All versions/formats failed for {name}")
        return None, []
        
    except Exception as e:
        print(f"   ❌ Failed to access project: {e}")
        return None, []

def download_all_datasets():
    """Download semua dataset yang tersedia"""
    print_section("DOWNLOADING ALL DATASETS FROM ROBOFLOW")
    
    rf = Roboflow(api_key=API_KEY)
    downloaded = {}
    all_classes_found = set()
    
    for name, config in DATASETS_CONFIG.items():
        location, classes = try_download_dataset(rf, name, config)
        if location:
            downloaded[name] = {
                'location': location,
                'classes': classes
            }
            all_classes_found.update(classes)
    
    # Summary
    print_section("DOWNLOAD SUMMARY")
    print(f"\n✅ Successfully downloaded: {len(downloaded)}/{len(DATASETS_CONFIG)} datasets")
    
    if downloaded:
        print("\n📊 Downloaded datasets:")
        for name, info in downloaded.items():
            print(f"   • {name}: {info['classes']}")
        
        print(f"\n📋 All unique classes found: {sorted(all_classes_found)}")
        print(f"📋 Target classes: {FINAL_CLASSES}")
        
        # Cek kelas yang belum ter-cover
        mapped_classes = set()
        for cls in all_classes_found:
            mapped = CLASS_MAPPING.get(cls) or CLASS_MAPPING.get(cls.lower())
            if mapped:
                mapped_classes.add(mapped)
        
        missing = set(FINAL_CLASSES) - mapped_classes
        if missing:
            print(f"\n⚠️  Classes not yet covered: {sorted(missing)}")
            print("   These will be added when more datasets are available.")
    
    return downloaded

def main():
    print("=" * 70)
    print(" CHIRA - Complete Dataset Downloader")
    print("=" * 70)
    print("\n🎯 Target: Download and integrate ALL available datasets")
    print(f"🎯 Target classes: {len(FINAL_CLASSES)} classes")
    
    downloaded = download_all_datasets()
    
    print("\n" + "=" * 70)
    print(" NEXT STEPS")
    print("=" * 70)
    
    if downloaded:
        print("\n✅ Dataset download complete!")
        print("\n🚀 Run training with:")
        print("   python train_disease.py")
    else:
        print("\n❌ No datasets downloaded.")
        print("\n💡 Please check:")
        print("   1. Internet connection")
        print("   2. Roboflow API key")
        print("   3. Dataset availability")

if __name__ == "__main__":
    main()
