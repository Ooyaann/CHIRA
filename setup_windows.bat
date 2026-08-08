@echo off
chcp 65001 >nul
title CHIRA Setup - Windows

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                                                               ║
echo ║   CHIRA - Chicken Health Identification and Recommendation   ║
echo ║                    Windows Setup Script                       ║
echo ║                                                               ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

set CHIRA_DIR=%~dp0
set VENV_DIR=%CHIRA_DIR%venv
set WEB_DIR=%CHIRA_DIR%web
set MODEL_PATH=%CHIRA_DIR%models\chira_disease_detection\weights\best.pt

cd /d "%CHIRA_DIR%"

:: Step 1: Check Python
echo [1/7] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python tidak ditemukan! Install Python 3.10+ dari python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%a in ('python --version') do set PYVER=%%a
echo    ✅ Python %PYVER% ditemukan

:: Step 2: Create Virtual Environment
echo.
echo [2/7] Membuat virtual environment...
if exist "%VENV_DIR%" (
    echo    ℹ️  Virtual environment sudah ada, menggunakan yang ada...
) else (
    python -m venv venv
    echo    ✅ Virtual environment dibuat
)

:: Step 3: Activate venv & Install dependencies
echo.
echo [3/7] Install dependencies ke virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
echo    ✅ Dependencies terinstall di venv

:: Step 4: Check GPU
echo.
echo [4/7] Checking GPU...
python -c "import torch; print('    GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Tidak ada (akan pakai CPU)')"

:: Step 5: Training model if not exists
echo.
if exist "%MODEL_PATH%" (
    echo [5/7] ✅ Model sudah ada: %MODEL_PATH%
) else (
    echo [5/7] Model belum ditemukan!
    echo.
    set /p TRAIN="🚀 Jalankan training sekarang? (y/n): "
    if /i "%TRAIN%"=="y" (
        echo    ⏳ Training dimulai... (butuh 2-4 jam)
        python train_disease.py
    ) else (
        echo    ⚠️  Training dilewati. Web tidak akan bisa deteksi tanpa model.
    )
)

:: Step 6: Setup Django
echo.
echo [6/7] Setup Django...
cd /d "%WEB_DIR%"
if not exist "media" mkdir media
if not exist "media\temp" mkdir media\temp
python manage.py makemigrations
python manage.py migrate
echo    ✅ Django siap

:: Step 7: Run server
echo.
echo [7/7] Menjalankan CHIRA Server...
echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║  🌐 Buka browser: http://localhost:8000                       ║
echo ║  ⏹️  Tekan Ctrl+C untuk menghentikan server                   ║
echo ║  📁 Virtual env aktif: venv\Scripts\activate.bat              ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

python manage.py runserver 0.0.0.0:8000

echo.
echo 👋 Server dihentikan. Terima kasih telah menggunakan CHIRA!
pause
