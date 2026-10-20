# 01 — Functional & Non-Functional Requirements

## 1. Functional requirements

### FR-01 — Dataset ingestion

Aplikasi training harus dapat membaca `data_ban.xlsx` dari folder data menggunakan nama sheet `data_ban`.

### FR-02 — Dataset validation

Validasi:

- kolom wajib ada;
- seluruh fitur numerik dapat dikonversi menjadi numeric;
- label hanya bernilai `0` atau `1`;
- missing value dilaporkan;
- duplicate row dilaporkan;
- nilai di luar batas domain dilaporkan;
- jumlah sampel dicatat.

### FR-03 — Preprocessing

Training preprocessing minimal:

1. numeric coercion;
2. missing-value handling yang didefinisikan dalam pipeline;
3. `StandardScaler`;
4. scaler hanya di-fit pada train set.

### FR-04 — Data split

Gunakan stratified split:

- train: 70%
- validation: 15%
- test: 15%
- `random_state = 42`

Target distribusi harus dijaga semirip mungkin antar split.

### FR-05 — Training

Training harus:

- menggunakan PyTorch;
- menggunakan MLP + ReLU;
- menggunakan BCEWithLogitsLoss;
- menggunakan Adam;
- mendukung early stopping;
- menyimpan history loss/metric;
- mencatat konfigurasi training.

### FR-06 — Evaluation

Minimal tampilkan:

- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- confusion matrix;
- classification report.

Metrik kelas `1` harus ditampilkan secara eksplisit.

### FR-07 — Model persistence

Minimal simpan:

- PyTorch weights;
- feature list;
- scaler parameters;
- model architecture/config;
- threshold;
- label mapping;
- training metadata;
- test metrics.

### FR-08 — Inference

Inference satu sampel harus menerima empat fitur proses dan mengembalikan JSON terstruktur.

### FR-09 — Health endpoint

API menyediakan health-check yang menunjukkan status service dan status artifact model.

### FR-10 — Dashboard

Dashboard minimal memiliki:

- overview;
- live prediction;
- input process parameters;
- ideal-range indicator;
- prediction result;
- probability;
- model evaluation;
- dataset summary;
- model information;
- error state.

### FR-11 — Logging

Training dan API harus mencatat error yang relevan tanpa membocorkan detail internal ke response HTTP.

## 2. Non-functional requirements

### NFR-01 — Reproducibility

Training ulang dengan seed/configuration yang sama harus menghasilkan hasil yang sangat dekat. Toleransi floating-point kecil diperbolehkan.

### NFR-02 — Maintainability

Code dipisah menurut concern:

- data;
- preprocessing;
- model;
- training;
- evaluation;
- artifact;
- API;
- dashboard;
- tests.

### NFR-03 — Type safety

Gunakan Python type hints pada public functions/classes dan Pydantic model untuk request/response API.

### NFR-04 — Validation

Input API harus divalidasi sebelum inference.

### NFR-05 — Usability

User harus dapat menjalankan project secara lokal dengan urutan command yang terdokumentasi.

### NFR-06 — Security baseline

- jangan menerima file arbitrary dari endpoint prediction;
- jangan menjalankan shell command dari request;
- jangan expose stack trace internal di production response;
- gunakan CORS yang eksplisit;
- jangan menyimpan secret di source code.

## 3. Out of scope

Jangan implementasikan kecuali diminta secara terpisah:

- login multi-user;
- database produksi;
- deployment cloud;
- Kubernetes;
- model registry enterprise;
- real-time sensor/MQTT;
- automated retraining from production feedback.
