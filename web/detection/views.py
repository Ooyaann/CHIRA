import os
import json
import cv2
import numpy as np
import tempfile
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib import messages
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta

from ultralytics import YOLO
from google import genai

from .models import DetectionResult, CameraConfig, AlertLog

# ============================================================================
# KONFIGURASI GEMINI API (SDK Baru: google-genai)
# ============================================================================
GEMINI_API_KEY = getattr(settings, 'GEMINI_API_KEY', "") or os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = 'gemini-3.6-flash'
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    gemini_client = None

# ============================================================================
# KONFIGURASI MODEL YOLO
# ============================================================================
MODEL_PATH = getattr(settings, 'MODEL_PATH', os.path.join(settings.BASE_DIR.parent, 'models', 'chira_disease_detection', 'weights', 'best.pt'))

# Load model (lazy loading)
_model = None

def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = YOLO(MODEL_PATH)
            print(f"[CHIRA] Model loaded from {MODEL_PATH}")
        else:
            print(f"[CHIRA] WARNING: Model not found at {MODEL_PATH}")
            _model = None
    return _model

# ============================================================================
# MAPPING PENYAKIT
# ============================================================================
CLASS_NAMES = [
    'healthy', 'fowl_pox', 'coryza', 'external_parasites',
    'newcastle_disease', 'mareks_disease', 'avian_influenza'
]

DISEASE_INFO = {
    'healthy': {
        'name': 'Ayam Sehat',
        'scientific_name': 'Gallus gallus domesticus',
        'description': 'Ayam dalam kondisi sehat tanpa gejala penyakit.',
        'symptoms': 'Ayam terlihat aktif, bulu mengkilap dan rapi, mata cerah dan responsif, nafsu makan dan minum normal, tidak ada tanda-tanda kelainan fisik.',
        'causes': 'Tidak ada patogen atau gangguan kesehatan yang terdeteksi pada ayam ini.',
        'severity': 'Aman',
        'treatment': 'Tidak diperlukan pengobatan. Lanjutkan pemberian pakan berkualitas dan air bersih secara rutin. Pantau kondisi ayam secara berkala untuk deteksi dini jika ada perubahan.',
        'prevention': '1. Berikan pakan dengan nutrisi seimbang sesuai umur dan fase produksi. 2. Pastikan air minum selalu bersih dan tersedia. 3. Lakukan vaksinasi sesuai jadwal yang dianjurkan dinas peternakan. 4. Jaga kebersihan kandang dan lakukan sanitasi rutin setiap minggu.',
        'immediate_action': 'Tidak ada tindakan darurat yang diperlukan. Tetap pantau kesehatan ayam secara rutin dan catat perkembangan berat badan serta produksi telur.'
    },
    'fowl_pox': {
        'name': 'Fowl Pox (Cacar Ayam)',
        'scientific_name': 'Avipoxvirus',
        'description': 'Penyakit viral yang menyebabkan lesi pada kulit dan membran mukosa.',
        'symptoms': '1. Bintik-bintik atau nodul berwarna putih kekuningan pada pial, jengger, dan kelopak mata. 2. Lesi kuning keputihan pada rongga mulut dan saluran pernapasan atas (bentuk basah). 3. Penurunan nafsu makan dan produksi telur. 4. Kesulitan bernapas jika lesi menyumbat saluran napas.',
        'causes': 'Disebabkan oleh virus Avipoxvirus dari keluarga Poxviridae. Penularan utama melalui gigitan nyamuk (vektor mekanis), kontak langsung dengan ayam terinfeksi, dan melalui luka terbuka pada kulit.',
        'severity': 'Sedang',
        'treatment': '1. Oleskan larutan Lugol iodine atau gentian violet pada lesi kulit untuk mencegah infeksi sekunder. 2. Berikan antibiotik spektrum luas (seperti Oxytetracycline) untuk mencegah infeksi bakteri sekunder. 3. Tambahkan vitamin A dan E pada pakan untuk mempercepat penyembuhan. 4. Untuk bentuk basah, bersihkan lesi di rongga mulut dengan larutan antiseptik.',
        'prevention': '1. Vaksinasi cacar ayam (fowl pox vaccine) pada umur 4-6 minggu melalui tusuk sayap. 2. Kendalikan populasi nyamuk di sekitar kandang dengan fogging atau pemasangan kelambu. 3. Karantina ayam baru selama minimal 2 minggu sebelum dicampur dengan flock yang ada. 4. Jaga kebersihan kandang dan sanitasi peralatan secara rutin.',
        'immediate_action': 'Segera pisahkan ayam yang terinfeksi ke kandang isolasi terpisah. Oleskan antiseptik pada lesi kulit dan berikan vitamin tambahan pada air minum seluruh flock.'
    },
    'coryza': {
        'name': 'Coryza (Snot)',
        'scientific_name': 'Avibacterium paragallinarum',
        'description': 'Infeksi bakteri akut pada saluran pernapasan atas ayam.',
        'symptoms': '1. Keluarnya cairan berlendir atau bernanah dari hidung. 2. Pembengkakan pada wajah, terutama di sekitar mata dan sinus infraorbital. 3. Bersin-bersin dan suara napas yang mengi. 4. Bau busuk yang khas dari cairan hidung. 5. Penurunan drastis produksi telur hingga 40 persen.',
        'causes': 'Disebabkan oleh bakteri gram negatif Avibacterium paragallinarum. Penularan melalui kontak langsung antar ayam, percikan cairan hidung, air minum yang terkontaminasi, dan peralatan kandang yang tidak dibersihkan.',
        'severity': 'Sedang',
        'treatment': '1. Berikan antibiotik Sulfadimethoxine atau Erythromycin melalui air minum selama 5-7 hari. 2. Kombinasikan dengan Trimethoprim-Sulfa untuk efektivitas lebih tinggi. 3. Berikan elektrolit dan vitamin pada air minum untuk menjaga stamina. 4. Bersihkan cairan hidung ayam secara manual jika menyumbat pernapasan.',
        'prevention': '1. Vaksinasi Coryza (killed bacterin) pada umur 8-12 minggu, booster pada umur 16 minggu. 2. Terapkan sistem all-in all-out dalam manajemen kandang. 3. Hindari mencampur ayam dari berbagai umur dalam satu kandang. 4. Tingkatkan ventilasi kandang untuk mengurangi kelembapan berlebih.',
        'immediate_action': 'Isolasi seluruh ayam yang menunjukkan gejala bersin dan bengkak wajah. Mulai pemberian antibiotik Sulfa melalui air minum hari ini juga dan bersihkan serta disinfeksi tempat minum.'
    },
    'external_parasites': {
        'name': 'Parasit Eksternal',
        'scientific_name': 'Dermanyssus / Menacanthus / Menopon',
        'description': 'Infestasi parasit pada kulit dan bulu ayam.',
        'symptoms': '1. Ayam sering menggaruk dan menggosokkan tubuh ke dinding atau lantai. 2. Bulu rontok tidak merata, terutama di area sekitar kloaka dan perut. 3. Kulit kemerahan, iritasi, dan terdapat kerak. 4. Penurunan berat badan dan produksi telur. 5. Anemia (pial dan jengger pucat) pada infestasi berat.',
        'causes': 'Disebabkan oleh infestasi tungau merah (Dermanyssus gallinae), kutu badan (Menacanthus stramineus), kutu bulu (Menopon gallinae), atau caplak (Argas persicus). Penyebaran melalui kontak langsung dan lingkungan kandang yang kotor.',
        'severity': 'Ringan',
        'treatment': '1. Semprotkan insektisida yang aman untuk unggas (Permethrin 0.25 persen) pada seluruh tubuh ayam. 2. Taburi bedak anti-kutu (Carbaryl dust) pada bulu ayam, fokus di bawah sayap dan sekitar kloaka. 3. Sediakan bak debu (campuran pasir halus, abu, dan sedikit kapur) untuk mandi debu alami. 4. Ulangi perlakuan setelah 10-14 hari untuk memutus siklus hidup parasit.',
        'prevention': '1. Bersihkan dan semprot kandang dengan insektisida secara rutin setiap 2 minggu. 2. Sediakan tempat mandi debu yang selalu tersedia di dalam kandang. 3. Periksa ayam baru sebelum dimasukkan ke flock yang ada. 4. Ganti alas kandang (litter) secara teratur dan jaga agar kandang tetap kering.',
        'immediate_action': 'Periksa seluruh flock untuk menentukan tingkat infestasi. Berikan perlakuan insektisida pada ayam yang terinfeksi dan bersihkan kandang secara menyeluruh hari ini.'
    },
    'newcastle_disease': {
        'name': 'Newcastle Disease (Tetelo)',
        'scientific_name': 'Avian paramyxovirus 1 (APMV-1)',
        'description': 'Penyakit viral sangat menular dengan tingkat kematian tinggi pada unggas.',
        'symptoms': '1. Gangguan pernapasan berat: bersin, batuk, suara ngorok. 2. Gangguan saraf: leher memutar (torticollis), gemetar, kelumpuhan kaki dan sayap. 3. Diare berwarna hijau keputihan. 4. Penurunan produksi telur secara drastis. 5. Kematian mendadak dalam jumlah besar (pada strain velogenik).',
        'causes': 'Disebabkan oleh virus Avian paramyxovirus tipe 1 (APMV-1) dari keluarga Paramyxoviridae. Penularan sangat cepat melalui udara, kontak langsung, feses, perlengkapan kandang, dan bahkan pakaian peternak.',
        'severity': 'Berat',
        'treatment': '1. Tidak ada obat spesifik untuk Newcastle Disease karena bersifat viral. 2. Berikan antibiotik spektrum luas untuk mencegah infeksi bakteri sekunder. 3. Berikan vitamin dan elektrolit melalui air minum untuk menjaga daya tahan tubuh. 4. Berikan pakan yang mudah dicerna dan pastikan ayam tetap terhidrasi. 5. Ayam dengan gejala saraf berat umumnya tidak dapat disembuhkan.',
        'prevention': '1. Vaksinasi ND wajib dilakukan: vaksin ND B1/La Sota tetes mata pada hari ke-4, pengulangan pada minggu ke-3 dan ke-8. 2. Vaksin ND killed (inaktif) injeksi pada umur 4 minggu untuk perlindungan jangka panjang. 3. Terapkan biosecurity ketat: batasi pengunjung, disinfeksi kendaraan dan alas kaki. 4. Laporkan kematian massal ke dinas peternakan setempat.',
        'immediate_action': 'DARURAT - Isolasi seluruh ayam yang menunjukkan gejala saraf dan pernapasan. Hentikan perpindahan ayam keluar masuk kandang. Laporkan kejadian ke dokter hewan atau dinas peternakan terdekat segera.'
    },
    'mareks_disease': {
        'name': "Marek's Disease",
        'scientific_name': 'Gallid alphaherpesvirus 2 (GaHV-2)',
        'description': 'Penyakit viral yang menyebabkan tumor limfoid dan infiltrasi saraf pada ayam.',
        'symptoms': '1. Kelumpuhan asimetris: satu kaki menjuntai ke depan, satu ke belakang (split leg). 2. Sayap terkulai pada satu sisi. 3. Pupil mata menjadi tidak beraturan (irregular pupil) dan perubahan warna iris menjadi keabu-abuan. 4. Penurunan berat badan progresif. 5. Tumor pada organ dalam (hati, limpa, ginjal) pada pemeriksaan post-mortem.',
        'causes': "Disebabkan oleh Gallid alphaherpesvirus 2 (GaHV-2), virus herpes yang sangat stabil di lingkungan. Penularan utama melalui inhalasi debu bulu dan serpihan kulit ayam terinfeksi. Virus dapat bertahan di lingkungan kandang selama berbulan-bulan.",
        'severity': 'Berat',
        'treatment': '1. Tidak ada pengobatan yang efektif untuk Marek Disease setelah gejala muncul. 2. Ayam yang sudah menunjukkan kelumpuhan sebaiknya diafkir secara manusiawi. 3. Berikan pakan bernutrisi tinggi dan vitamin untuk ayam yang masih bisa makan dan minum. 4. Tingkatkan sanitasi kandang untuk mengurangi viral load di lingkungan.',
        'prevention': "1. Vaksinasi Marek's Disease (HVT vaccine) WAJIB diberikan pada hari pertama (DOC) di hatchery. 2. Jangan memasukkan ayam baru yang belum divaksinasi ke dalam flock. 3. Bersihkan dan fumigasi kandang secara menyeluruh sebelum memasukkan batch ayam baru. 4. Gunakan sistem all-in all-out dan hindari mencampur ayam berbeda umur.",
        'immediate_action': 'Pisahkan ayam yang menunjukkan kelumpuhan kaki atau sayap ke kandang isolasi. Periksa seluruh flock untuk mendeteksi ayam lain dengan gejala serupa. Konsultasikan dengan dokter hewan untuk konfirmasi diagnosis.'
    },
    'avian_influenza': {
        'name': 'Avian Influenza (Flu Burung)',
        'scientific_name': 'Influenza A virus (H5N1/H5N6/H7N9)',
        'description': 'Penyakit viral yang sangat menular dan berpotensi zoonosis (menular ke manusia).',
        'symptoms': '1. Kematian mendadak tanpa gejala awal (pada HPAI). 2. Pembengkakan parah pada kepala, wajah, jengger, dan pial. 3. Sianosis (perubahan warna kebiruan) pada jengger dan pial. 4. Keluarnya darah dari hidung dan mulut. 5. Penurunan produksi telur drastis dan telur berkualitas rendah (kerabang tipis).',
        'causes': 'Disebabkan oleh virus Influenza A subtipe H5 atau H7 (Highly Pathogenic Avian Influenza). Penularan melalui kontak dengan unggas liar yang terinfeksi, feses burung migran, air yang terkontaminasi, serta peralatan dan kendaraan.',
        'severity': 'Berat',
        'treatment': '1. TIDAK ADA pengobatan yang efektif untuk Avian Influenza. 2. Pemusnahan (culling) ayam terinfeksi dan flock yang kontak merupakan kebijakan standar pemerintah. 3. Gunakan alat pelindung diri (APD) lengkap saat menangani ayam yang diduga terinfeksi AI. 4. Jangan mengkonsumsi atau menjual ayam dan telur dari flock yang terinfeksi.',
        'prevention': '1. Terapkan biosecurity sangat ketat: kandang tertutup, mencegah kontak dengan unggas liar. 2. Pasang jaring atau kawat pada ventilasi kandang untuk menghalangi burung liar masuk. 3. Disinfeksi alas kaki, kendaraan, dan peralatan sebelum masuk area kandang. 4. Laporkan kematian massal mendadak ke dinas peternakan dan Balai Veteriner setempat.',
        'immediate_action': 'DARURAT TINGGI - Hentikan seluruh aktivitas jual-beli dan perpindahan unggas. Isolasi kandang sepenuhnya. Gunakan APD (masker, sarung tangan, sepatu boot) saat mendekati kandang. Laporkan segera ke dinas peternakan atau Balai Veteriner karena AI wajib dilaporkan (notifiable disease).'
    },
}

# ============================================================================
# GEMINI RECOMMENDATION ENGINE
# ============================================================================
def get_gemini_recommendation(disease_class, confidence, behavior=None):
    """
    Mendapatkan rekomendasi dari Gemini API berdasarkan penyakit yang terdeteksi.
    Jika Gemini gagal, gunakan fallback response.
    """
    disease = DISEASE_INFO.get(disease_class, {})
    d_name = disease.get('name', disease.get('id', disease_class))
    d_sci = disease.get('scientific_name', disease.get('en', disease_class))
    d_desc = disease.get('description', 'Deskripsi tidak tersedia.')
    d_sev = disease.get('severity', 'Sedang' if disease_class != 'healthy' else 'Aman')
    
    # Fallback response SPESIFIK per penyakit (jika Gemini gagal)
    fallback = {
        'penyebab': disease.get('causes', d_desc),
        'gejala_umum': disease.get('symptoms', 'Gejala sesuai dengan hasil deteksi visual AI.'),
        'penanganan': disease.get('treatment', 'Segera isolasi ayam yang terinfeksi dan konsultasikan dengan dokter hewan setempat.'),
        'pencegahan': disease.get('prevention', 'Jaga kebersihan kandang, vaksinasi rutin, dan karantina ayam baru selama 2 minggu.'),
        'tingkat_keparahan': d_sev,
        'rekomendasi_segera': disease.get('immediate_action', 'Pisahkan ayam yang menunjukkan gejala ke kandang isolasi dan konsultasikan dengan dokter hewan.')
    }
    
    try:
        behavior_text = f"Perilaku terdeteksi: {behavior}. " if behavior else ""
        
        prompt = f"""Anda adalah ahli kesehatan ayam dan dokter hewan berpengalaman 20 tahun di Indonesia.
        Berikan analisis medis untuk ayam yang terdeteksi kondisi: {d_name} ({d_sci}).
        Confidence deteksi AI: {confidence:.1f}%.
        {behavior_text}
        
        ATURAN PENTING:
        - DILARANG menggunakan emoji, emotikon, atau simbol unicode apapun (seperti ✅❌⚠️🐔💊 dll).
        - Gunakan HANYA teks biasa tanpa simbol dekoratif.
        - Tulis dalam Bahasa Indonesia yang lugas, praktis, dan mudah dipahami peternak desa.
        - Setiap field harus berisi minimal 2-3 kalimat informatif.
        - Gunakan penomoran (1, 2, 3) untuk daftar langkah-langkah, bukan bullet point atau simbol.
        
        Berikan respons dalam format JSON berikut:
        {{
            "penyebab": "penjelasan ilmiah penyebab penyakit ini (2-3 kalimat, tanpa emoji)",
            "gejala_umum": "daftar gejala klinis yang biasa terlihat pada ayam (gunakan penomoran 1, 2, 3)",
            "penanganan": "langkah pengobatan dan perawatan detail (gunakan penomoran 1, 2, 3)",
            "pencegahan": "cara pencegahan untuk peternakan (gunakan penomoran 1, 2, 3)",
            "tingkat_keparahan": "Ringan/Sedang/Berat",
            "rekomendasi_segera": "tindakan pertama yang harus SECEPATNYA dilakukan peternak (2-3 kalimat konkret)"
        }}
        
        Pastikan respons akurat, praktis, dan sesuai standar medis unggas Indonesia."""
        
        if gemini_client is None:
            return fallback
        
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        response_text = response.text
        
        # Parse JSON dari response
        # Cari JSON di dalam response (kadang ada markdown code block)
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return fallback
            
    except Exception as e:
        print(f"[CHIRA] Gemini API Error: {e}")
        return fallback


# ============================================================================
# VIEWS
# ============================================================================

def home(request):
    """Dashboard utama CHIRA"""
    # Statistik
    today = timezone.now().date()
    today_detections = DetectionResult.objects.filter(timestamp__date=today)
    
    total_detections = DetectionResult.objects.count()
    total_diseases = DetectionResult.objects.exclude(detected_class='healthy').count()
    avg_confidence = DetectionResult.objects.aggregate(avg=Avg('confidence'))['avg'] or 0
    
    # Distribusi penyakit (7 hari terakhir)
    week_ago = today - timedelta(days=7)
    disease_dist = DetectionResult.objects.filter(
        timestamp__date__gte=week_ago
    ).exclude(detected_class='healthy').values('detected_class').annotate(
        count=Count('id')
    ).order_by('-count')[:5]
    
    # Recent detections
    recent_detections = DetectionResult.objects.all()[:10]
    
    # Unread alerts
    unread_alerts = AlertLog.objects.filter(is_read=False).count()
    
    context = {
        'total_detections': total_detections,
        'total_diseases': total_diseases,
        'avg_confidence': round(avg_confidence * 100, 1),
        'today_count': today_detections.count(),
        'disease_distribution': disease_dist,
        'recent_detections': recent_detections,
        'unread_alerts': unread_alerts,
        'disease_info': DISEASE_INFO,
    }
    return render(request, 'detection/dashboard.html', context)


def camera_view(request):
    """Halaman kamera real-time"""
    cameras = CameraConfig.objects.filter(is_active=True)
    is_streaming = request.session.get('is_streaming', False)
    latest_detection = DetectionResult.objects.order_by('-timestamp').first()
    context = {
        'cameras': cameras,
        'is_streaming': is_streaming,
        'latest_detection': latest_detection,
    }
    return render(request, 'detection/camera.html', context)


def start_camera(request):
    """Mulai streaming kamera"""
    request.session['is_streaming'] = True
    return redirect('detection:camera')


def stop_camera(request):
    """Hentikan streaming kamera"""
    request.session['is_streaming'] = False
    return redirect('detection:camera')


def get_latest_detection(request):
    """API untuk melihat deteksi terbaru"""
    latest = DetectionResult.objects.order_by('-timestamp').first()
    if latest:
        return JsonResponse({
            'id': latest.id,
            'detected_class': latest.detected_class,
            'disease_name': latest.disease_name_id,
            'confidence': latest.confidence_percent,
            'severity': latest.severity_level,
            'timestamp': latest.timestamp.isoformat(),
            'is_healthy': latest.detected_class == 'healthy'
        })
    return JsonResponse({'detected_class': None})



def upload_view(request):
    """Halaman upload manual - GET menampilkan form, POST menjalankan analisis AI"""
    if request.method == 'POST':
        # Validasi file gambar
        if 'image' not in request.FILES:
            return render(request, 'detection/upload.html', {
                'error': 'Tidak ada gambar yang dipilih. Silakan pilih foto ayam terlebih dahulu.'
            })
        
        image_file = request.FILES['image']
        
        if not image_file.content_type.startswith('image/'):
            return render(request, 'detection/upload.html', {
                'error': 'File harus berupa gambar (JPG, PNG, JPEG).'
            })
        
        if image_file.size > 10 * 1024 * 1024:  # 10MB limit
            return render(request, 'detection/upload.html', {
                'error': 'Ukuran gambar terlalu besar. Maksimal 10MB.'
            })
        
        try:
            # Simpan file sementara untuk deteksi
            temp_path = default_storage.save(f'temp/{image_file.name}', ContentFile(image_file.read()))
            try:
                full_path = default_storage.path(temp_path)
            except Exception:
                full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
            
            # Load model YOLO
            model = get_model()
            if model is None:
                return render(request, 'detection/upload.html', {
                    'error': 'Model AI belum tersedia. Silakan hubungi admin untuk melatih model terlebih dahulu.'
                })
            
            # Reset file pointer agar bisa disimpan ke DB nanti
            image_file.seek(0)
            
            # Run YOLO detection
            results = model(full_path, conf=0.5, verbose=False)
            result = results[0]
            
            # Jika tidak ada deteksi, coba confidence lebih rendah
            if len(result.boxes) == 0:
                results = model(full_path, conf=0.25, verbose=False)
                result = results[0]
                
                if len(result.boxes) == 0:
                    # Hapus file temp
                    try:
                        os.remove(full_path)
                    except:
                        pass
                    return render(request, 'detection/upload.html', {
                        'error': 'Tidak ada ayam atau gejala penyakit terdeteksi dalam gambar. Pastikan gambar menunjukkan ayam dengan jelas, pencahayaan cukup, dan diambil dari jarak 30-50 cm.'
                    })
            
            # Ambil deteksi dengan confidence tertinggi
            best_idx = 0
            best_conf = float(result.boxes[0].conf[0])
            for i in range(1, len(result.boxes)):
                conf = float(result.boxes[i].conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_idx = i
            
            box = result.boxes[best_idx]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            # Validasi kelas
            if class_name not in DISEASE_INFO:
                try:
                    os.remove(full_path)
                except:
                    pass
                return render(request, 'detection/upload.html', {
                    'error': f'Objek terdeteksi ({class_name}) bukan merupakan ayam atau kelas penyakit yang dikenal. Pastikan gambar menunjukkan ayam.'
                })
            
            # Bounding box
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            
            # Get Gemini AI recommendation
            gemini_response = get_gemini_recommendation(class_name, confidence)
            
            # Simpan hasil ke database
            detection = DetectionResult.objects.create(
                image=image_file,
                detection_type='upload',
                detected_class=class_name,
                confidence=confidence,
                bbox_coords=json.dumps([x1, y1, x2, y2]),
                disease_name_id=DISEASE_INFO.get(class_name, {}).get('name', DISEASE_INFO.get(class_name, {}).get('id', class_name)),
                disease_name_en=DISEASE_INFO.get(class_name, {}).get('scientific_name', DISEASE_INFO.get(class_name, {}).get('en', class_name)),
                causes=gemini_response.get('penyebab', ''),
                symptoms=gemini_response.get('gejala_umum', ''),
                treatment=gemini_response.get('penanganan', ''),
                prevention=gemini_response.get('pencegahan', ''),
                severity_level=gemini_response.get('tingkat_keparahan', 'Tidak Diketahui'),
                immediate_action=gemini_response.get('rekomendasi_segera', ''),
                is_anomaly=(class_name != 'healthy' and confidence > 0.7),
            )
            
            # Buat alert jika penyakit terdeteksi
            if class_name != 'healthy' and confidence > 0.7:
                AlertLog.objects.create(
                    detection=detection,
                    alert_type='disease_detected',
                    message=f"Penyakit {detection.disease_name_id} terdeteksi dengan confidence {detection.confidence_percent}%"
                )
            
            # Hapus file temporary
            try:
                os.remove(full_path)
            except:
                pass
            
            # Redirect ke halaman detail hasil analisis
            return redirect('detection:detail', pk=detection.pk)
        
        except Exception as e:
            import traceback
            print(f"[CHIRA] Upload Detection Error: {e}")
            print(traceback.format_exc())
            return render(request, 'detection/upload.html', {
                'error': f'Terjadi kesalahan saat menganalisis gambar: {str(e)}. Silakan coba lagi.'
            })
    
    # GET request - tampilkan form upload
    return render(request, 'detection/upload.html')


def history_view(request):
    """Halaman riwayat deteksi"""
    detections = DetectionResult.objects.all()
    
    # Filter
    disease_filter = request.GET.get('disease')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    
    if disease_filter:
        detections = detections.filter(detected_class=disease_filter)
    if date_from:
        detections = detections.filter(timestamp__date__gte=date_from)
    if date_to:
        detections = detections.filter(timestamp__date__lte=date_to)
    
    context = {
        'detections': detections[:100],
        'disease_classes': DISEASE_INFO,
        'filters': {
            'disease': disease_filter,
            'date_from': date_from,
            'date_to': date_to,
        }
    }
    return render(request, 'detection/history.html', context)


def detection_detail(request, pk):
    """Detail hasil deteksi"""
    detection = DetectionResult.objects.get(pk=pk)
    return render(request, 'detection/detail.html', {'detection': detection})


def delete_detection(request, pk):
    """Hapus 1 riwayat deteksi secara manual"""
    if request.method == 'POST':
        try:
            detection = DetectionResult.objects.get(pk=pk)
            if detection.image:
                try:
                    if os.path.exists(detection.image.path):
                        os.remove(detection.image.path)
                except Exception as file_err:
                    print(f"[CHIRA Delete File Error]: {file_err}")
            
            detection.delete()
            messages.success(request, f"Riwayat deteksi #{pk} berhasil dihapus.")
        except DetectionResult.DoesNotExist:
            messages.error(request, "Data riwayat tidak ditemukan.")
        except Exception as e:
            messages.error(request, f"Gagal menghapus riwayat: {str(e)}")
            
    return redirect('detection:history')


def clear_history(request):
    """Hapus seluruh riwayat deteksi secara manual"""
    if request.method == 'POST':
        try:
            all_records = DetectionResult.objects.all()
            count = all_records.count()
            
            for rec in all_records:
                if rec.image:
                    try:
                        if os.path.exists(rec.image.path):
                            os.remove(rec.image.path)
                    except Exception:
                        pass
                rec.delete()
                
            messages.success(request, f"Seluruh {count} data riwayat berhasil dibersihkan.")
        except Exception as e:
            messages.error(request, f"Gagal membersihkan riwayat: {str(e)}")
            
    return redirect('detection:history')


def print_full_report(request):
    """Halaman Laporan Diagnosis Kesehatan Ayam Seluruh Populasi (Print-Ready)"""
    detections = DetectionResult.objects.all()
    
    disease_filter = request.GET.get('disease')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    
    if disease_filter:
        detections = detections.filter(detected_class=disease_filter)
    if date_from:
        detections = detections.filter(timestamp__date__gte=date_from)
    if date_to:
        detections = detections.filter(timestamp__date__lte=date_to)
        
    total_count = detections.count()
    healthy_count = detections.filter(detected_class='healthy').count()
    disease_count = total_count - healthy_count
    
    avg_conf = detections.aggregate(avg=Avg('confidence'))['avg'] or 0
    
    context = {
        'detections': detections,
        'total_count': total_count,
        'healthy_count': healthy_count,
        'disease_count': disease_count,
        'avg_conf': round(avg_conf * 100, 1),
        'generated_at': timezone.now(),
        'disease_info': DISEASE_INFO,
        'filters': {
            'disease': disease_filter,
            'date_from': date_from,
            'date_to': date_to,
        }
    }
    return render(request, 'detection/report.html', context)
# Auto-reloaded views module


# ============================================================================
# API ENDPOINTS
# ============================================================================

@csrf_exempt
def detect_image(request):
    """
    API endpoint untuk deteksi gambar.
    Menerima: POST dengan file gambar
    Return: JSON dengan hasil deteksi dan rekomendasi
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        # Ambil gambar dari request
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'Tidak ada gambar yang diupload'}, status=400)
        
        image_file = request.FILES['image']
        
        # Validasi file
        if not image_file.content_type.startswith('image/'):
            return JsonResponse({'error': 'File harus berupa gambar'}, status=400)
        
        # Simpan file sementara
        temp_path = default_storage.save(f'temp/{image_file.name}', ContentFile(image_file.read()))
        try:
            full_path = default_storage.path(temp_path)
        except Exception:
            full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
        
        # Load model
        model = get_model()
        if model is None:
            return JsonResponse({
                'error': 'Model AI belum tersedia. Silakan training model terlebih dahulu.',
                'model_path': MODEL_PATH
            }, status=503)
        
        # Run YOLO detection
        results = model(full_path, conf=0.5, verbose=False)
        result = results[0]
        
        # Cek apakah ada deteksi
        if len(result.boxes) == 0:
            # Coba deteksi dengan confidence lebih rendah
            results = model(full_path, conf=0.25, verbose=False)
            result = results[0]
            
            if len(result.boxes) == 0:
                return JsonResponse({
                    'detected': False,
                    'message': 'Tidak ada ayam atau penyakit terdeteksi dalam gambar. Pastikan gambar menunjukkan ayam dengan jelas.',
                    'suggestion': 'Coba ambil gambar dari sudut yang berbeda atau dengan pencahayaan lebih baik.'
                })
        
        # Ambil deteksi dengan confidence tertinggi
        best_idx = 0
        best_conf = float(result.boxes[0].conf[0])
        
        for i in range(1, len(result.boxes)):
            conf = float(result.boxes[i].conf[0])
            if conf > best_conf:
                best_conf = conf
                best_idx = i
        
        box = result.boxes[best_idx]
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        
        # Cek apakah ini bukan ayam (anti-halusinasi)
        if class_name not in DISEASE_INFO:
            return JsonResponse({
                'detected': False,
                'message': f'Objek terdeteksi: {class_name}. Ini bukan ayam atau kelas tidak dikenal.',
                'note': 'Sistem CHIRA hanya mendeteksi kondisi kesehatan ayam.'
            })
        
        # Bounding box
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        
        # Generate annotated image
        annotated_img = result.plot()
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_file = ContentFile(buffer.tobytes())
        
        # Get Gemini recommendation
        gemini_response = get_gemini_recommendation(class_name, confidence)
        
        # Simpan ke database
        detection = DetectionResult.objects.create(
            image=image_file,
            detection_type='upload',
            detected_class=class_name,
            confidence=confidence,
            bbox_coords=json.dumps([x1, y1, x2, y2]),
            disease_name_id=DISEASE_INFO.get(class_name, {}).get('name', DISEASE_INFO.get(class_name, {}).get('id', class_name)),
            disease_name_en=DISEASE_INFO.get(class_name, {}).get('scientific_name', DISEASE_INFO.get(class_name, {}).get('en', class_name)),
            causes=gemini_response.get('penyebab', ''),
            symptoms=gemini_response.get('gejala_umum', ''),
            treatment=gemini_response.get('penanganan', ''),
            prevention=gemini_response.get('pencegahan', ''),
            severity_level=gemini_response.get('tingkat_keparahan', 'Tidak Diketahui'),
            immediate_action=gemini_response.get('rekomendasi_segera', ''),
            is_anomaly=(class_name != 'healthy' and confidence > 0.7),
        )
        
        # Buat alert jika penyakit terdeteksi
        if class_name != 'healthy' and confidence > 0.7:
            AlertLog.objects.create(
                detection=detection,
                alert_type='disease_detected',
                message=f"Penyakit {detection.disease_name_id} terdeteksi dengan confidence {detection.confidence_percent}%"
            )
        
        # Hapus file temporary
        try:
            os.remove(full_path)
        except:
            pass
        
        return JsonResponse({
            'detected': True,
            'detection_id': detection.id,
            'class': class_name,
            'class_id': detection.disease_name_id,
            'class_en': detection.disease_name_en,
            'confidence': detection.confidence_percent,
            'bbox': [x1, y1, x2, y2],
            'severity': detection.severity_level,
            'recommendation': {
                'penyebab': detection.causes,
                'gejala': detection.symptoms,
                'penanganan': detection.treatment,
                'pencegahan': detection.prevention,
                'tingkat_keparahan': detection.severity_level,
                'tindakan_segera': detection.immediate_action,
            },
            'timestamp': detection.timestamp.isoformat(),
            'is_anomaly': detection.is_anomaly,
        })
        
    except Exception as e:
        import traceback
        print(f"[CHIRA] Detection Error: {e}")
        print(traceback.format_exc())
        return JsonResponse({
            'error': f'Terjadi kesalahan saat analisis: {str(e)}',
            'detail': 'Silakan coba lagi dengan gambar yang berbeda.'
        }, status=500)


# ============================================================================
# AUTO RETENTION & CLEANUP
# ============================================================================

def auto_cleanup_old_detections(days_retention=7):
    """
    Fungsi pembersihan otomatis riwayat deteksi & gambar yang lebih lama dari `days_retention` hari.
    Mencegah penumpukan file gambar di sistem penyimpanan disk.
    """
    try:
        cutoff_date = timezone.now() - timedelta(days=days_retention)
        old_records = DetectionResult.objects.filter(timestamp__lt=cutoff_date)
        
        deleted_count = 0
        for rec in old_records:
            if rec.image:
                try:
                    if os.path.exists(rec.image.path):
                        os.remove(rec.image.path)
                except Exception as file_err:
                    print(f"[CHIRA Cleanup File Error]: {file_err}")
            rec.delete()
            deleted_count += 1
            
        if deleted_count > 0:
            print(f"[CHIRA Auto-Cleanup] Berhasil menghapus {deleted_count} riwayat deteksi & foto berusia > {days_retention} hari.")
    except Exception as e:
        print(f"[CHIRA Auto-Cleanup Error]: {e}")


@csrf_exempt
def detect_camera_frame(request):
    """
    API endpoint untuk deteksi frame dari kamera real-time.
    Menerima: POST dengan base64 image
    Secara otomatis menyimpan tangkapan layar ke Riwayat (History) setiap 10 detik.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        import base64
        
        data = json.loads(request.body)
        image_data = data.get('image', '')
        
        if not image_data:
            return JsonResponse({'error': 'No image data'}, status=400)
        
        # Decode base64
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JsonResponse({'error': 'Invalid image data'}, status=400)
        
        # Save temporary for YOLO
        temp_path = os.path.join(tempfile.gettempdir(), f'chira_frame_{int(timezone.now().timestamp())}.jpg')
        cv2.imwrite(temp_path, frame)
        
        # Detect
        model = get_model()
        if model is None:
            return JsonResponse({'error': 'Model not available'}, status=503)
        
        results = model(temp_path, conf=0.45, verbose=False)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            d_info = DISEASE_INFO.get(class_name, {})
            if class_name in DISEASE_INFO or len(DISEASE_INFO) == 0:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append({
                    'class': class_name,
                    'disease_name': d_info.get('name', d_info.get('id', class_name)),
                    'scientific_name': d_info.get('scientific_name', d_info.get('en', '')),
                    'confidence': round(confidence * 100, 1),
                    'bbox': [x1, y1, x2, y2],
                    'is_healthy': class_name == 'healthy'
                })
        
        # Cleanup temporary file
        try:
            os.remove(temp_path)
        except:
            pass

        # ---------------------------------------------------------------------
        # PENYIMPANAN RIWAYAT REAL-TIME (Setiap 10 Detik Sekali)
        # ---------------------------------------------------------------------
        saved_record_info = None
        last_saved_ts = request.session.get('last_camera_save_time', 0)
        now_ts = timezone.now().timestamp()
        
        # Simpan ke DB jika ada deteksi dan sudah berlalu minimal 10 detik sejak simpanan terakhir
        if len(detections) > 0 and (now_ts - last_saved_ts >= 10):
            best_det = max(detections, key=lambda x: x['confidence'])
            c_name = best_det['class']
            c_conf = best_det['confidence'] / 100.0
            
            # Ambil rekomendasi AI Gemini / Fallback
            gemini_res = get_gemini_recommendation(c_name, c_conf)
            
            # Plot bounding box pada gambar yang disimpan
            annotated_frame = result.plot()
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            file_name = f'camera_{int(now_ts)}.jpg'
            saved_file_path = default_storage.save(f'detections/{file_name}', ContentFile(buffer.tobytes()))
            
            record = DetectionResult.objects.create(
                image=saved_file_path,
                detection_type='camera',
                detected_class=c_name,
                confidence=c_conf,
                bbox_coords=json.dumps([d['bbox'] for d in detections]),
                disease_name_id=best_det['disease_name'],
                disease_name_en=best_det['scientific_name'],
                causes=gemini_res.get('penyebab', ''),
                symptoms=gemini_res.get('gejala_umum', ''),
                treatment=gemini_res.get('penanganan', ''),
                prevention=gemini_res.get('pencegahan', ''),
                severity_level=gemini_res.get('tingkat_keparahan', 'Sedang'),
                immediate_action=gemini_res.get('rekomendasi_segera', ''),
                is_anomaly=(c_name != 'healthy' and c_conf > 0.6)
            )
            
            if record.is_anomaly:
                AlertLog.objects.create(
                    detection=record,
                    alert_type='disease_detected',
                    message=f"Penyakit {record.disease_name_id} terdeteksi dari Kamera Real-time (Confidence {record.confidence_percent}%)"
                )
            
            request.session['last_camera_save_time'] = now_ts
            saved_record_info = {
                'id': record.id,
                'detail_url': f'/detection/{record.id}/'
            }

            # Jalankan auto-cleanup riwayat > 7 hari
            auto_cleanup_old_detections(days_retention=7)
        
        return JsonResponse({
            'detected': len(detections) > 0,
            'detections': detections,
            'frame_time': timezone.now().isoformat(),
            'saved_record': saved_record_info
        })
        
    except Exception as e:
        import traceback
        print(f"[CHIRA Camera Frame Error]: {e}")
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)


# ============================================================================
# VIDEO STREAMING
# ============================================================================

def gen_camera_frames(camera_source='0'):
    """Generator untuk video streaming dengan deteksi real-time"""
    src = int(camera_source) if str(camera_source).isdigit() else camera_source
    camera = cv2.VideoCapture(src)
    
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)
        
        model = get_model()
        frame_count = 0
        consecutive_failures = 0
        
        while True:
            success, frame = camera.read()
            if not success:
                consecutive_failures += 1
                if consecutive_failures > 20:
                    break
                import time
                time.sleep(0.05)
                continue
            
            consecutive_failures = 0
            
            # Deteksi setiap 5 frame untuk performa
            if model is not None and frame_count % 5 == 0:
                try:
                    results = model(frame, conf=0.4, verbose=False)
                    annotated = results[0].plot()
                except Exception:
                    annotated = frame
            else:
                annotated = frame
            
            # Encode JPEG
            ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            frame_count += 1
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        camera.release()
    
    # Standby frame jika kamera fisik tidak terdeteksi / terputus
    import time
    for _ in range(12):
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        error_frame[:, :] = (30, 25, 15)  # Dark slate background
        cv2.putText(error_frame, "KAMERA STANDBY / TIDAK TERHUBUNG", (50, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(error_frame, "Gunakan IP Webcam HP atau Kamera USB", (70, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(1)


def video_feed(request):
    """Endpoint video streaming"""
    camera_id = request.GET.get('camera', '0')
    return StreamingHttpResponse(
        gen_camera_frames(camera_id),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


# ============================================================================
# ALERT API
# ============================================================================

def get_alerts(request):
    """API untuk mendapatkan alert terbaru"""
    alerts = AlertLog.objects.filter(is_read=False).order_by('-timestamp')[:10]
    data = [{
        'id': a.id,
        'type': a.get_alert_type_display(),
        'message': a.message,
        'timestamp': a.timestamp.isoformat(),
        'detection_id': a.detection.id if a.detection else None,
    } for a in alerts]
    return JsonResponse({'alerts': data, 'count': len(data)})


def mark_alert_read(request, pk):
    """Tandai alert sebagai sudah dibaca"""
    try:
        alert = AlertLog.objects.get(pk=pk)
        alert.is_read = True
        alert.save()
        return JsonResponse({'success': True})
    except AlertLog.DoesNotExist:
        return JsonResponse({'error': 'Alert not found'}, status=404)


# ============================================================================
# STATISTICS API
# ============================================================================

def get_statistics(request):
    """API untuk statistik dashboard"""
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    
    stats = {
        'total_detections': DetectionResult.objects.count(),
        'today_detections': DetectionResult.objects.filter(timestamp__date=today).count(),
        'disease_count': DetectionResult.objects.exclude(detected_class='healthy').count(),
        'avg_confidence': round(
            (DetectionResult.objects.aggregate(avg=Avg('confidence'))['avg'] or 0) * 100, 1
        ),
        'weekly_trend': list(DetectionResult.objects.filter(
            timestamp__date__gte=week_ago
        ).extra({'date': "date(timestamp)"}).values('date').annotate(
            count=Count('id')
        ).order_by('date')),
        'top_diseases': list(DetectionResult.objects.filter(
            timestamp__date__gte=week_ago
        ).exclude(detected_class='healthy').values('detected_class').annotate(
            count=Count('id')
        ).order_by('-count')[:5]),
    }
    
    return JsonResponse(stats)
