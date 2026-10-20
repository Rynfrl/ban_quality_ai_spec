# Ban Quality ML — Industrial Quality Classification System

Sistem machine learning end-to-end untuk klasifikasi kualitas proses vulkanisasi ban berdasarkan studi kasus industri manufaktur ban (`data_ban.xlsx`).

---

## 1. Arsitektur & Gambaran Sistem

Sistem memprediksi hasil proses vulkanisasi sebelum ban dikirim untuk uji laboratorium fisik yang bersifat destruktif:
- **Kelas 0: Baik** (mutu prima, homogen, siap lanjut ke tahap berikutnya)
- **Kelas 1: Perlu Perbaikan** (potensi overcure/undercure; batch dikarantina)

### Flow Pipeline
```text
data/data_ban.xlsx 
  -> Validasi Skema & Domain
  -> Stratified Split (70% Train, 15% Validation, 15% Test)
  -> StandardScaler (Fit HANYA pada Train Set)
  -> PyTorch MLP + ReLU + Dropout(0.10)
  -> BCEWithLogitsLoss (pos_weight ≈ 1.87) + Adam
  -> Early Stopping (Patience: 20)
  -> Seleksi Threshold pada Validasi
  -> Evaluasi Satu Kali pada Holdout Test Set (75 sampel)
  -> Simpan Artifacts (model.pt, scaler.json, model_config.json, dll.)
  -> FastAPI REST API & Single-Page Dashboard
```

---

## 2. Kebijakan Fitur Produksi (Feature Policy)

### Fitur Produksi Live (Wajib Berurutan)
```python
FEATURES = (
    "suhu",             # °C      (Dataset: 140.2–180.0 | Ideal: 150.0–165.0)
    "tekanan",          # bar     (Dataset: 15.0–25.0   | Ideal: 18.0–22.0)
    "waktu_curing",     # menit   (Dataset: 8.0–15.0    | Ideal: 10.0–13.0)
    "rasio_karet",      # rasio   (Dataset: 0.40–0.80   | Ideal: 0.50–0.70)
    "kekerasan_karet",  # Shore A (Dataset: 55.1–75.0   | Ideal: 60.0–70.0)
)
```

### Integrasi `kekerasan_karet` Sebagai Input Live
`kekerasan_karet` (Shore A) kini diintegrasikan secara penuh sebagai input live pada pipeline produksi dan model neural network (misalnya melalui sensor durometer otomatis/in-line di lini vulkanisasi). Nilai ini diverifikasi terhadap rentang operasional valid dan rentang mutu ideal (60.0 – 70.0 Shore A).

---

## 3. Hasil Evaluasi Model Aktual (Holdout Test Set: 75 Sampel)

Evaluasi dilakukan secara objektif pada 75 sampel data uji independen yang tidak pernah dilihat model saat pelatihan maupun saat penentuan threshold keputusan:

| Metrik | PyTorch MLP + ReLU | Baseline Logistic Regression |
|---|---:|---:|
| **Accuracy** | **80.00%** | 62.67% |
| **Precision (Kelas 1)** | **68.97%** | 47.22% |
| **Recall (Kelas 1)** | **76.92%** | 65.38% |
| **F1-Score (Kelas 1)** | **72.73%** | 54.84% |
| **ROC-AUC** | **0.8689** | 0.7151 |
| **Balanced Accuracy** | **79.28%** | 63.30% |

### Confusion Matrix (Test Set):
```text
                  Prediksi Baik (0)    Prediksi Perlu Perbaikan (1)
Aktual Baik (0)          40 (TN)                  9 (FP)
Aktual Perbaikan (1)      6 (FN)                  20 (TP)
```
Keunggulan PyTorch MLP atas Logistic Regression membuktikan keberadaan pola batas non-linear antara parameter vulkanisasi terhadap mutu ban.

---

## 4. Cara Menjalankan (Step-by-Step)

### A. Persiapan Lingkungan & Instalasi
```powershell
# Buat dan aktifkan virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Instal paket dependensi
pip install -r requirements.txt
```

### B. Pelatihan Model & Pembuatan Artifacts
Jalankan modul training reproducible:
```powershell
python -m ml.train --data data/data_ban.xlsx --sheet data_ban --epochs 200 --batch-size 32 --lr 0.001 --seed 42
```
Perintah ini akan membaca dataset, memvalidasi integritas, melatih neural network PyTorch, mengevaluasi model pada test set, dan menyimpan seluruh file ke folder `artifacts/`:
- `artifacts/model.pt`
- `artifacts/model_config.json`
- `artifacts/scaler.json`
- `artifacts/hasil_evaluasi.json`
- `artifacts/training_history.json`

### C. Menjalankan REST API & Web Dashboard
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Akses di browser:
- **Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Documentation (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### D. Menjalankan Pengujian Otomatis (Automated Tests)
Jalankan seluruh rangkaian test unit dan integrasi:
```powershell
python -m pytest -v
```

---

## 5. Ringkasan Endpoint REST API

- `GET /health`: Pemeriksaan status operasional dan kesiapan artifact model.
- `GET /api/model-info`: Mengembalikan metadata versi model, fitur, threshold, dan metrik uji.
- `POST /api/predict`: Inferensi satu sampel parameter proses:
  ```json
  {
    "suhu": 160.0,
    "tekanan": 20.0,
    "waktu_curing": 11.0,
    "rasio_karet": 0.60
  }
  ```
- `POST /api/predict/batch`: Inferensi batch beberapa sampel sekaligus.
- `GET /api/evaluation`: Laporan evaluasi lengkap beserta confusion matrix dan baseline comparison.
- `GET /api/training-history`: Data history training loss dan validation loss untuk visualisasi grafik.

---

## 6. Struktur Direktori Proyek

```text
ban_quality_ai_spec/
├── data/
│   └── data_ban.xlsx             # Dataset sumber
├── artifacts/
│   ├── model.pt                  # State dict bobot PyTorch
│   ├── model_config.json         # Konfigurasi arsitektur & metadata
│   ├── scaler.json               # Parameter mean & scale StandardScaler
│   ├── hasil_evaluasi.json       # Laporan metrik uji & confusion matrix
│   └── training_history.json     # Log epoch loss & metrics
├── app/
│   ├── main.py                   # FastAPI server & route handlers
│   ├── config.py                 # Konfigurasi runtime
│   ├── schemas.py                # Pydantic schemas & validasi
│   ├── services/
│   │   ├── predictor.py          # Logika inferensi PyTorch
│   │   ├── artifact_loader.py    # Manajemen lifecycle model
│   │   └── range_checker.py      # Pengecekan rentang ideal & dataset
│   └── web/
│       ├── templates/
│       │   └── index.html        # UI dashboard industri
│       └── static/
│           ├── css/app.css       # Styling visual profesional
│           └── js/app.js         # Interaksi dashboard & live chart
├── ml/
│   ├── config.py                 # Single source of truth fitur & batas
│   ├── data.py                   # Ingestion & validasi skema
│   ├── preprocessing.py          # Split & StandardScaler serialization
│   ├── model.py                  # PyTorch MLP + ReLU architecture
│   ├── train.py                  # CLI & training orchestrator
│   ├── evaluate.py               # Metrik evaluasi & error analysis
│   └── artifacts.py              # Serialization & smoke testing
├── tests/
│   ├── test_data.py              # Unit test data & validasi
│   ├── test_preprocessing.py     # Unit test scaler & leakage check
│   ├── test_model.py             # Unit test PyTorch MLP & threshold
│   ├── test_artifacts.py         # Test integritas reload artifacts
│   └── test_api.py               # Integration test REST API & dashboard
├── requirements.txt              # Daftar dependensi Python
├── .gitignore                    # Berkas ignorasi Git
├── .env.example                  # Template variabel lingkungan
└── README.md                     # Dokumentasi utama proyek
```
