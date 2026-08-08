from django.urls import path
from . import views

app_name = 'detection'

urlpatterns = [
    # Halaman utama
    path('', views.home, name='home'),
    path('camera/', views.camera_view, name='camera'),
    path('camera/start/', views.start_camera, name='start_camera'),
    path('camera/stop/', views.stop_camera, name='stop_camera'),
    path('camera/latest-detection/', views.get_latest_detection, name='latest_detection'),
    path('upload/', views.upload_view, name='upload'),
    path('history/', views.history_view, name='history'),
    path('history/clear/', views.clear_history, name='clear_history'),
    path('report/', views.print_full_report, name='print_report'),
    path('detection/<int:pk>/', views.detection_detail, name='detail'),
    path('detection/<int:pk>/detail/', views.detection_detail, name='detection_detail'),
    path('detection/<int:pk>/delete/', views.delete_detection, name='delete_detection'),
    
    # Video streaming
    path('video_feed/', views.video_feed, name='video_feed'),
    
    # API endpoints
    path('api/detect/', views.detect_image, name='detect_image'),
    path('api/detect-camera/', views.detect_camera_frame, name='detect_camera_frame'),
    path('api/alerts/', views.get_alerts, name='get_alerts'),
    path('api/alerts/<int:pk>/read/', views.mark_alert_read, name='mark_alert_read'),
    path('api/statistics/', views.get_statistics, name='get_statistics'),
    
]
