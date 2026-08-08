# 🐔 CHIRA - Cara Menjalankan (Panduan Lengkap)

## ⚠️ WAJIB: Gunakan Virtual Environment!

Virtual environment (venv) **wajib** dipakai supaya package Python CHIRA tidak bentrok dengan project lain di laptop Anda.

---

## 🪟 CARA 1: Windows (Paling Mudah)

### Opsi A: Klik File BAT (Otomatis)
```
1. Buka folder CHIRA di File Explorer
2. Klik 2x file: setup_windows.bat
3. Tunggu proses selesai
4. Browser akan otomatis terbuka di http://localhost:8000
```

### Opsi B: Manual dengan Command Prompt

**Buka CMD atau PowerShell, lalu ketik:**

```cmd
:: 1. Masuk ke folder CHIRA
cd C:\Users\NAMA_ANDA\Downloads\chira

:: 2. Buat virtual environment
python -m venv venv

:: 3. Aktifkan virtual environment
venv\Scripts\activate.bat

:: (Kalau pakai PowerShell: venv\Scripts\Activate.ps1)

:: 4. Install dependencies
pip install -r requirements.txt

:: 5. TRAINING MODEL (WAJIB - 2-4 jam)
python train_disease.py

:: 6. Setup Django
cd web
python manage.py makemigrations
python manage.py migrate

:: 7. Jalankan server
python manage.py runserver

:: 8. Buka browser ke http://localhost:8000
```

**Untuk menonaktifkan venv:**
```cmd
deactivate
```

---

## 🐧 CARA 2: Linux / macOS / WSL

```bash
# 1. Masuk ke folder CHIRA
cd ~/Downloads/chira

# 2. Buat virtual environment
python3 -m venv venv

# 3. Aktifkan virtual environment
source venv/bin/activate

# Kalau berhasil, prompt terminal akan ada tulisan (venv) di depannya:
# (venv) user@laptop:~/chira$

# 4. Install dependencies
pip install -r requirements.txt

# 5. TRAINING MODEL (WAJIB - 2-4 jam)
python train_disease.py

# 6. Setup Django
cd web
python manage.py makemigrations
python manage.py migrate

# 7. Jalankan server
python manage.py runserver

# 8. Buka browser ke http://localhost:8000
```

**Untuk menonaktifkan venv:**
```bash
deactivate
```

---

## 🔁 Cara Menjalankan Lagi (Setelah Setup Pertama)

Setelah setup pertama selesai, untuk menjalankan lagi cukup:

### Windows:
```cmd
cd C:\Users\NAMA_ANDA\Downloads\chira
venv\Scripts\activate.bat
cd web
python manage.py runserver
```

### Linux/macOS:
```bash
cd ~/Downloads/chira
source venv/bin/activate
cd web
python manage.py runserver
```

---

## 📋 Checklist Sebelum Mulai

- [ ] Python 3.10+ sudah terinstall (`python --version`)
- [ ] Folder CHIRA sudah di-copy ke laptop
- [ ] Internet tersambung (untuk download dataset & package)
- [ ] GPU NVIDIA sudah terinstall driver & CUDA (opsional tapi sangat direkomendasikan)
- [ ] Waktu luang 3-5 jam (termasuk training model)

---

## ❓ Troubleshooting

### Error: "pip tidak dikenali"
**Solusi:** Install Python dari [python.org](https://python.org) dan centang "Add Python to PATH"

### Error: "No module named 'torch'"
**Solusi:** Pastikan venv sudah aktif (ada tulisan `(venv)` di prompt terminal)

### Error: "CUDA out of memory"
**Solusi:** Edit `train_disease.py`, ganti `BATCH_SIZE = 16` jadi `BATCH_SIZE = 8`

### Error: "Model not found"
**Solusi:** Training belum dilakukan. Jalankan `python train_disease.py` terlebih dahulu.

---

---

## 🌐 CARA 3: Deploy ke PythonAnywhere & Push GitHub

### 🐙 A. Push ke GitHub
``` bash
# 1. Inisialisasi git (jika belum)
git init

# 2. Tambah remote repository GitHub Anda
git remote add origin https://github.com/USERNAME/chira.git

# 3. Add semua file (file sementara & venv otomatis diabaikan oleh .gitignore)
git add .

# 4. Commit & Push
git commit -m "CHIRA Final: AI Chicken Health Identification & Recommendation Assistant"
git branch -M main
git push -u origin main
```

### 🐍 B. Publish ke PythonAnywhere
1. **Daftar / Login** di [PythonAnywhere.com](https://www.pythonanywhere.com).
2. **Buka Bash Console** di PythonAnywhere dan clone repository:
   ```bash
   git clone https://github.com/USERNAME/chira.git
   cd chira
   ```
3. **Buat & Aktifkan Virtualenv**:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.10 chira-venv
   pip install -r requirements.txt
   ```
4. **Collect Static & Migration**:
   ```bash
   cd web
   python manage.py collectstatic --noinput
   python manage.py migrate
   ```
5. **Konfigurasi Web App** di menu **Web** PythonAnywhere:
   - **Source Code Path**: `/home/USERNAME/chira/web`
   - **Working Directory**: `/home/USERNAME/chira/web`
   - **Virtualenv Path**: `/home/USERNAME/.virtualenvs/chira-venv`
   - **WSGI configuration file**: Masukkan konfigurasi WSGI berikut:
     ```python
     import os
     import sys

     path = '/home/USERNAME/chira/web'
     if path not in sys.path:
         sys.path.append(path)

     os.environ['DJANGO_SETTINGS_MODULE'] = 'chira_project.settings'

     from django.core.wsgi import get_wsgi_application
     application = get_wsgi_application()
     ```
   - **Static Files**:
     - URL: `/static/` -> Path: `/home/USERNAME/chira/web/staticfiles`
     - URL: `/media/` -> Path: `/home/USERNAME/chira/web/media`
6. Klik **Reload USERNAME.pythonanywhere.com**. Web CHIRA langsung live secara online! 🎉

---

## 🎯 Ringkasan Perintah Wajib

| Urutan | Perintah | Keterangan |
|:---|:---|:---|
| 1 | `python -m venv venv` | Buat virtual env |
| 2 | `venv\Scripts\activate` atau `source venv/bin/activate` | Aktifkan venv |
| 3 | `pip install -r requirements.txt` | Install package |
| 4 | `python train_disease.py` | **Training AI (2-4 jam)** |
| 5 | `cd web && python manage.py migrate` | Setup database |
| 6 | `python manage.py runserver` | Jalankan web |

---

**Selamat! CHIRA telah siap 100% untuk presentasi Gemastik 2026, GitHub, dan PythonAnywhere!** 🚀
