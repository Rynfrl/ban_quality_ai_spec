# 00 — AI Build Brief

## Peran AI

Bertindak sebagai senior Python/ML engineer sekaligus full-stack engineer. Tugas AI adalah membangun aplikasi production-like untuk studi kasus klasifikasi kualitas ban, bukan sekadar membuat notebook eksperimen.

## Problem statement

Sebuah pabrik memiliki data parameter proses vulkanisasi/curing ban. Tujuan sistem adalah memprediksi apakah hasil proses akan:

- `0 — Baik`
- `1 — Perlu Perbaikan`

sebelum ban harus melalui pengujian fisik.

Hubungan fitur terhadap kualitas bersifat non-linear, sehingga model menggunakan MLP dengan ReLU.

## Input produksi

Gunakan empat fitur berikut sebagai input default:

| Feature | Type | Unit | Dataset range | Ideal range |
|---|---|---:|---:|---:|
| `suhu` | float | °C | 140–180 | 150–165 |
| `tekanan` | float | bar | 15–25 | 18–22 |
| `waktu_curing` | float | menit | 8–15 | 10–13 |
| `rasio_karet` | float | - | 0.4–0.8 | 0.5–0.7 |

`kekerasan_karet` tersedia di Excel pada rentang sekitar 55–75 Shore A dan digunakan sebagai variabel analisis offline, bukan sebagai input live default.

## Output produksi

API/dashboard harus memberikan:

- prediksi kelas;
- nama kelas;
- probabilitas `Baik`;
- probabilitas `Perlu Perbaikan`;
- threshold keputusan;
- status masing-masing input terhadap rentang ideal;
- warning bila input berada di luar rentang dataset atau rentang proses yang diperbolehkan;
- versi model.

## Model

Arsitektur default:

```text
Input(4)
  -> Linear(4, 32)
  -> ReLU
  -> Linear(32, 16)
  -> ReLU
  -> Dropout(0.10)
  -> Linear(16, 1)
  -> Logit
  -> Sigmoid pada inference
```

Gunakan `BCEWithLogitsLoss`, bukan `Sigmoid + BCELoss` terpisah pada training.

## Objective function operasional

Kesalahan paling berbahaya adalah memprediksi `Baik` ketika sebenarnya `Perlu Perbaikan`. Karena itu, recall untuk kelas `1` harus menjadi metrik prioritas bersama F1 dan ROC-AUC. Threshold keputusan harus dapat dikonfigurasi dan dilaporkan.

## Teknologi wajib

Backend / ML:

- Python 3.11+
- pandas
- numpy
- scikit-learn
- PyTorch
- FastAPI
- Uvicorn
- Pydantic

Frontend:

- HTML
- CSS
- JavaScript vanilla

Jangan menambah framework frontend besar kecuali ada alasan teknis yang kuat. Untuk studi kasus ini dashboard server-side/static sederhana lebih mudah dipelihara dan dijalankan.

## Aturan penting

1. Jangan hard-code hasil training.
2. Jangan menggunakan `kekerasan_karet` sebagai input live tanpa mengubah spesifikasi dan menjelaskan bahwa model tersebut adalah eksperimen offline.
3. Jangan melakukan scaling sebelum split.
4. Jangan fit scaler pada validation/test.
5. Gunakan seed untuk reproducibility.
6. Simpan preprocessing dan metadata model bersama artifact.
7. API harus menolak input dengan tipe atau range yang tidak valid.
8. Dashboard tidak boleh membuat prediksi sendiri dengan rumus berbeda dari backend.
9. Semua prediksi live harus menggunakan model artifact yang sama dengan hasil training.
10. Sertakan test otomatis.

## Definition of done

Build hanya dianggap selesai ketika seluruh acceptance criteria pada `13_ACCEPTANCE_CRITERIA.md` terpenuhi.
