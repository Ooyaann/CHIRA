import json
from django.db import models


class DetectionResult(models.Model):
    """Model untuk menyimpan hasil deteksi penyakit ayam"""
    
    DETECTION_TYPES = [
        ('realtime', 'Kamera Real-time'),
        ('upload', 'Upload Manual'),
    ]
    
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name='Waktu Deteksi')
    image = models.ImageField(upload_to='detections/%Y/%m/%d/', verbose_name='Gambar')
    detection_type = models.CharField(
        max_length=20, 
        choices=DETECTION_TYPES,
        verbose_name='Tipe Deteksi'
    )
    
    # Hasil deteksi YOLO
    detected_class = models.CharField(max_length=50, verbose_name='Kelas Terdeteksi')
    confidence = models.FloatField(verbose_name='Confidence Score')
    bbox_coords = models.TextField(verbose_name='Koordinat Bounding Box')  # JSON: [x1, y1, x2, y2]
    
    # Analisis perilaku
    behavior_detected = models.CharField(
        max_length=100, 
        blank=True, 
        verbose_name='Perilaku Terdeteksi'
    )
    behavior_confidence = models.FloatField(
        null=True, 
        blank=True, 
        verbose_name='Confidence Perilaku'
    )
    
    # Informasi penyakit (Bahasa Indonesia & Inggris)
    disease_name_id = models.CharField(max_length=100, verbose_name='Nama Penyakit (ID)')
    disease_name_en = models.CharField(max_length=100, verbose_name='Nama Penyakit (EN)')
    
    # Rekomendasi dari Gemini API
    causes = models.TextField(verbose_name='Penyebab')
    symptoms = models.TextField(blank=True, verbose_name='Gejala')
    treatment = models.TextField(blank=True, verbose_name='Penanganan')
    prevention = models.TextField(blank=True, verbose_name='Pencegahan')
    severity_level = models.CharField(
        max_length=20, 
        blank=True,
        verbose_name='Tingkat Keparahan'
    )
    immediate_action = models.TextField(blank=True, verbose_name='Tindakan Segera')
    
    # Status
    is_anomaly = models.BooleanField(default=False, verbose_name='Anomali')
    alert_sent = models.BooleanField(default=False, verbose_name='Alert Terkirim')
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Hasil Deteksi'
        verbose_name_plural = 'Hasil Deteksi'
    
    def __str__(self):
        return f"{self.disease_name_id} ({self.confidence:.1f}%) - {self.timestamp.strftime('%d/%m %H:%M')}"
    
    def get_bbox(self):
        """Parse bounding box dari JSON string"""
        try:
            return json.loads(self.bbox_coords)
        except:
            return None
    
    @property
    def confidence_percent(self):
        """Return confidence dalam persen"""
        return round(self.confidence * 100, 2)
    
    @property
    def is_healthy(self):
        """Return True jika terdeteksi Ayam Sehat"""
        return self.detected_class == 'healthy'
        
    @property
    def status_badge(self):
        """Return status badge color"""
        if self.detected_class == 'healthy':
            return 'success'
        elif self.confidence > 0.8:
            return 'danger'
        elif self.confidence > 0.5:
            return 'warning'
        return 'info'


class CameraConfig(models.Model):
    """Konfigurasi kamera untuk monitoring"""
    
    name = models.CharField(max_length=100, verbose_name='Nama Kamera')
    camera_id = models.CharField(max_length=50, unique=True, verbose_name='ID Kamera')
    source = models.CharField(
        max_length=200, 
        default='0',
        verbose_name='Sumber Kamera (0 untuk webcam, URL untuk IP Camera)'
    )
    is_active = models.BooleanField(default=True, verbose_name='Aktif')
    auto_analysis = models.BooleanField(default=True, verbose_name='Analisis Otomatis')
    analysis_interval = models.IntegerField(
        default=5, 
        verbose_name='Interval Analisis (detik)'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Konfigurasi Kamera'
        verbose_name_plural = 'Konfigurasi Kamera'
    
    def __str__(self):
        return f"{self.name} ({'Aktif' if self.is_active else 'Nonaktif'})"


class AlertLog(models.Model):
    """Log untuk alert/notifikasi"""
    
    ALERT_TYPES = [
        ('disease_detected', 'Penyakit Terdeteksi'),
        ('anomaly_behavior', 'Perilaku Anomali'),
        ('system_error', 'Error Sistem'),
    ]
    
    timestamp = models.DateTimeField(auto_now_add=True)
    detection = models.ForeignKey(
        DetectionResult, 
        on_delete=models.CASCADE,
        related_name='alerts'
    )
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Log Alert'
        verbose_name_plural = 'Log Alert'
    
    def __str__(self):
        return f"{self.get_alert_type_display()} - {self.timestamp.strftime('%d/%m %H:%M')}"
