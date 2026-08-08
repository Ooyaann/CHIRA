# CHIRA - Chicken Health Identification and Recommendation Assistant

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv11-Deep%20Learning-green?style=for-the-badge" alt="YOLOv11">
  <img src="https://img.shields.io/badge/Django-Web%20Framework-blue?style=for-the-badge" alt="Django">
  <img src="https://img.shields.io/badge/Gemini-AI%20Recommendation-orange?style=for-the-badge" alt="Gemini">
  <img src="https://img.shields.io/badge/GEMASTIK-2026-red?style=for-the-badge" alt="GEMASTIK 2026">
</p>

## Deskripsi

**CHIRA** adalah sistem berbasis **Kecerdasan Buatan (AI)** yang dirancang untuk membantu peternak ayam dalam **mendeteksi penyakit ayam secara non-invasif** melalui analisis citra tubuh ayam menggunakan kamera HP atau CCTV. Sistem ini menggabungkan **YOLOv11** untuk deteksi objek dan **Gemini API** untuk memberikan rekomendasi penanganan yang komprehensif.

## Fitur Utama

### 1. Deteksi Penyakit Berbasis AI (YOLOv11)
- **15 Kelas Deteksi**: healthy, fowl_pox, coryza, coccidiosis, newcastle_disease, salmonellosis, avian_influenza, gumboro, mareks_disease, infectious_bronchitis, aspergillosis, botulism, external_parasites, egg_peritonitis, sick_unspecified
- **Akurasi Tinggi**: Target mAP50 > 85%, mAP50-95 > 70%
- **Anti-Halusinasi**: Sistem menolak objek non-ayam (manusia, hewan lain)

### 2. Analisis Perilaku Ayam
- Deteksi perilaku mencurigakan: lesu, isolasi, nafsu makan turun, bulu kusam, sayap terkulai
- Motion tracking untuk analisis pergerakan

### 3. Platform Web Interaktif (Django)
- **Dashboard Monitoring**: Statistik real-time, grafik distribusi penyakit
- **Kamera Real-time**: WebRTC streaming dengan bounding box overlay
- **Upload Manual**: Drag & drop gambar untuk analisis instan
- **Decision Support System**: Confidence score, visualisasi gejala, rekomendasi penanganan

### 4. Rekomendasi AI (Gemini API)
- Penjelasan ilmiah penyebab penyakit
- Panduan penanganan dan pengobatan
- Strategi pencegahan untuk peternakan
- Tindakan segera yang harus dilakukan

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPTURE LAYER                             │
│  ├─ Kamera HP (WebRTC)                                      │
│  └─ CCTV IP Camera (RTSP)                                   │
├─────────────────────────────────────────────────────────────┤
│                    AI PROCESSING LAYER                       │
│  ├─ YOLOv11 Disease Detection (15 kelas)                    │
│  ├─ YOLOv11 Behavior Detection                              │
│  └─ Gemini API Recommendation Engine                        │
├─────────────────────────────────────────────────────────────┤
│                    WEB PLATFORM (Django)                     │
│  ├─ Real-time Dashboard                                     │
│  ├─ Auto/Manual Analysis                                    │
│  ├─ Alert & Notification                                    │
│  └─ Historical Analytics                                    │
└─────────────────────────────────────────────────────────────┘
```

## Dataset

| No | Dataset | Sumber | Kelas | Gambar |
|:---|:---|:---|:---|:---|
| 1 | chicken-diseases-with-roboflow-model-c58n3 | Roboflow | healthy, Fowlpox, Coryza | 1,031 |
| 2 | chicken-disease-detection-ncfow | Roboflow | Multiple diseases | ~500+ |
| 3 | chicken-desease | Roboflow | Multiple diseases | ~400+ |
| 4 | sick-and-healthy-chicken | Roboflow | sick, healthy | 425+ |

**Total**: ~2,500+ gambar setelah harmonisasi

## Teknologi

| Komponen | Teknologi |
|:---|:---|
| Object Detection | YOLOv11 (Ultralytics) |
| Backend Framework | Django 5.x + Django REST Framework |
| Real-time | Django Channels + WebSocket |
| AI Recommendation | Gemini 1.5 Flash API |
| Database | SQLite (production: PostgreSQL) |
| Frontend | HTML5 + Tailwind CSS + Chart.js |
| Image Processing | OpenCV + Pillow |

## Instalasi

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (untuk GPU training)
- Git

### Langkah 1: Clone Repository
```bash
git clone <repository-url>
cd chira
```

### Langkah 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Langkah 3: Training Model (WAJIB)
```bash
# Training YOLOv11 untuk deteksi penyakit
python train_disease.py
```

**Catatan**: Training memerlukan:
- GPU dengan VRAM minimal 6GB (RTX 3060 12GB direkomendasikan)
- Waktu: 2-4 jam untuk 100 epoch
- Model akan tersimpan di `models/chira_disease_detection/weights/best.pt`

### Langkah 4: Setup Django
```bash
cd web

# Migrasi database
python manage.py makemigrations
python manage.py migrate

# Buat superuser (opsional)
python manage.py createsuperuser

# Jalankan server
python manage.py runserver 0.0.0.0:8000
```

### Langkah 5: Akses Aplikasi
Buka browser dan akses: `http://localhost:8000`

## Struktur Proyek

```
chira/
├── datasets/                  # Dataset hasil download & merge
│   ├── dataset1_diseases/
│   ├── dataset2_disease_detection/
│   ├── dataset3_desease/
│   ├── dataset4_sick_healthy/
│   └── merged_disease_dataset/
├── models/                    # Model hasil training
│   └── chira_disease_detection/
│       └── weights/
│           ├── best.pt
│           └── last.pt
├── notebooks/                 # Jupyter notebooks
├── results/                   # Hasil evaluasi & grafik
├── train_disease.py          # Script training
├── requirements.txt          # Dependencies
├── README.md                 # Dokumentasi
└── web/                      # Django web application
    ├── chira_project/        # Django project config
    ├── detection/            # Django app
    │   ├── models.py
    │   ├── views.py
    │   ├── urls.py
    │   └── admin.py
    ├── templates/            # HTML templates
    │   └── detection/
    │       ├── base.html
    │       ├── dashboard.html
    │       ├── camera.html
    │       ├── upload.html
    │       ├── history.html
    │       └── detail.html
    ├── static/               # CSS, JS, images
    └── manage.py
```

## API Endpoints

| Endpoint | Method | Deskripsi |
|:---|:---|:---|
| `/` | GET | Dashboard |
| `/camera/` | GET | Halaman kamera real-time |
| `/upload/` | GET | Halaman upload manual |
| `/history/` | GET | Riwayat deteksi |
| `/api/detect/` | POST | Deteksi gambar (upload) |
| `/api/detect-camera/` | POST | Deteksi frame dari kamera |
| `/video_feed/` | GET | Video streaming |
| `/api/statistics/` | GET | Statistik dashboard |
| `/api/alerts/` | GET | Alert terbaru |

## Target Akurasi

| Metrik | Target |
|:---|:---|
| mAP50 | > 85% |
| mAP50-95 | > 70% |
| Precision | > 80% |
| Recall | > 80% |
| F1-Score | > 80% |
| Non-chicken Rejection | > 95% |

## Dukungan SDG & Asta Cita

- **SDG 2**: Zero Hunger - Swasembada pangan melalui peternakan sehat
- **SDG 9**: Industry, Innovation, and Infrastructure - Inovasi teknologi pertanian
- **SDG 12**: Responsible Consumption and Production - Efisiensi produksi pangan
- **Asta Cita**: Swasembada pangan dan ketahanan pangan nasional

## Kontribusi

Proyek ini dikembangkan untuk **GEMASTIK 2026** dalam bidang **ICT Scientific Paper**.

## Lisensi

MIT License - Lihat [LICENSE](LICENSE) untuk detail.

## Kontak

Untuk pertanyaan atau kolaborasi, silakan hubungi tim CHIRA.

---

<p align="center">
  <strong>CHIRA</strong> - Membangun Masa Depan Peternakan Ayam yang Lebih Sehat dengan AI
</p>
