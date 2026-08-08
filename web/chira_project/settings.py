"""
Django settings for CHIRA project.
CHIRA = Chicken Health Identification and Recommendation Assistant
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-chira-chicken-health-ai-2026-gemastik'

DEBUG = True

ALLOWED_HOSTS = ['*']

CSRF_TRUSTED_ORIGINS = [
    'https://*.pythonanywhere.com',
    'http://*.pythonanywhere.com',
    'https://*.ngrok-free.dev',
    'https://*.ngrok-free.app',
    'https://*.ngrok.dev',
    'https://*.ngrok.app',
    'https://*.ngrok.io',
    'https://*.loca.lt',
    'http://*.ngrok-free.dev',
    'http://*.ngrok-free.app',
    'http://*.ngrok.dev',
    'http://*.ngrok.app',
    'http://*.ngrok.io',
    'http://*.loca.lt',
    'http://localhost:8000',
    'http://127.0.0.1:8000',
]

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'detection',
    'rest_framework',
    'channels',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'chira_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'chira_project.wsgi.application'
ASGI_APPLICATION = 'chira_project.asgi.application'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer',
    },
}

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'id-id'
TIME_ZONE = 'Asia/Jakarta'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CHIRA Settings
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

default_model_path = os.path.join(BASE_DIR.parent, 'models', 'chira_disease_detection', 'weights', 'best.pt')
if not os.path.exists(default_model_path):
    alt_model_path = os.path.join(BASE_DIR, 'models', 'best.pt')
    if os.path.exists(alt_model_path):
        default_model_path = alt_model_path

MODEL_PATH = os.environ.get('MODEL_PATH', default_model_path)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'chira.log',
        },
    },
    'loggers': {
        'detection': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
