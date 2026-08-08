from django.contrib import admin
from .models import DetectionResult, CameraConfig, AlertLog


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = [
        'timestamp', 'disease_name_id', 'confidence_percent', 
        'detection_type', 'is_anomaly', 'severity_level'
    ]
    list_filter = [
        'detected_class', 'detection_type', 'is_anomaly', 
        'severity_level', 'timestamp'
    ]
    search_fields = ['disease_name_id', 'disease_name_en', 'detected_class']
    readonly_fields = ['timestamp', 'confidence_percent']
    date_hierarchy = 'timestamp'
    ordering = ['-timestamp']
    
    fieldsets = (
        ('Informasi Dasar', {
            'fields': ('timestamp', 'image', 'detection_type')
        }),
        ('Hasil Deteksi', {
            'fields': ('detected_class', 'confidence', 'bbox_coords', 'is_anomaly')
        }),
        ('Informasi Penyakit', {
            'fields': ('disease_name_id', 'disease_name_en', 'severity_level')
        }),
        ('Analisis Lengkap', {
            'fields': ('causes', 'symptoms', 'treatment', 'prevention', 'immediate_action'),
            'classes': ('collapse',)
        }),
        ('Perilaku', {
            'fields': ('behavior_detected', 'behavior_confidence'),
            'classes': ('collapse',)
        }),
        ('Status', {
            'fields': ('alert_sent',)
        }),
    )


@admin.register(CameraConfig)
class CameraConfigAdmin(admin.ModelAdmin):
    list_display = ['name', 'camera_id', 'source', 'is_active', 'auto_analysis', 'created_at']
    list_filter = ['is_active', 'auto_analysis']
    search_fields = ['name', 'camera_id']
    list_editable = ['is_active', 'auto_analysis']


@admin.register(AlertLog)
class AlertLogAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'alert_type', 'message_preview', 'is_read']
    list_filter = ['alert_type', 'is_read', 'timestamp']
    search_fields = ['message']
    actions = ['mark_as_read']
    
    def message_preview(self, obj):
        return obj.message[:100] + '...' if len(obj.message) > 100 else obj.message
    message_preview.short_description = 'Pesan'
    
    @admin.action(description='Tandai sebagai sudah dibaca')
    def mark_as_read(self, request, queryset):
        queryset.update(is_read=True)
