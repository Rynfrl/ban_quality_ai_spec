# 07 — Dashboard Specification

## 1. Goal

Dashboard adalah antarmuka demonstrasi industri untuk melihat pipeline ML dan melakukan prediksi kualitas ban secara live.

## 2. Navigation

Gunakan navigasi/tab:

1. Gambaran
2. Data
3. Preprocessing
4. Model ReLU
5. Training
6. Evaluasi
7. Simpan JSON
8. Dashboard
9. Demo Live
10. Cara Menjalankan

Tidak semua tab harus berupa halaman terpisah; beberapa dapat menjadi section pada satu SPA/static page. Namun seluruh informasi harus mudah ditemukan.

## 3. Gambaran

Tampilkan:

- masalah industri;
- tujuan;
- flow:

```text
Excel
  -> Python/Pandas
  -> Preprocessing
  -> PyTorch MLP + ReLU
  -> Evaluation
  -> JSON/Model Artifact
  -> Web Dashboard
```

- jumlah data;
- label distribution;
- daftar input production.

## 4. Data

Tampilkan:

- 500 records;
- schema;
- min/max;
- ideal range;
- label distribution;
- catatan bahwa `kekerasan_karet` tersedia namun tidak digunakan sebagai input live default.

## 5. Preprocessing

Tampilkan diagram/section:

```text
raw -> validate -> split -> StandardScaler -> tensor
```

Tekankan bahwa scaler fit hanya pada train.

## 6. Model ReLU

Tampilkan architecture card:

```text
4 -> 32 -> ReLU -> 16 -> ReLU -> Dropout(0.10) -> 1
```

Tambahkan penjelasan sederhana bahwa ReLU membantu jaringan memodelkan hubungan non-linear.

## 7. Training

Tampilkan:

- epoch terbaik;
- training loss;
- validation loss;
- training/validation metric;
- hyperparameters;
- random seed.

Bila history tersedia, gunakan line chart.

## 8. Evaluasi

Tampilkan KPI:

- Accuracy
- Precision class 1
- Recall class 1
- F1 class 1
- ROC-AUC

Tampilkan confusion matrix.

Gunakan penekanan visual pada recall class `Perlu Perbaikan`.

## 9. Simpan JSON

Tampilkan artifact:

- model.pt
- model_config.json
- scaler.json
- hasil_evaluasi.json
- training_history.json

Berikan ringkasan metadata, bukan isi file panjang.

## 10. Dashboard / Demo Live

Form input:

| Input | Min | Max | Ideal |
|---|---:|---:|---:|
| Suhu | 140 | 180 | 150–165 |
| Tekanan | 15 | 25 | 18–22 |
| Waktu curing | 8 | 15 | 10–13 |
| Rasio karet | 0.4 | 0.8 | 0.5–0.7 |

Gunakan number input atau slider + numeric field.

Setiap perubahan harus memperbarui status range. Tombol `Prediksi` mengirim data ke `/api/predict`.

## 11. Prediction result

Result card wajib menampilkan:

- `Baik` atau `Perlu Perbaikan`;
- confidence/probability;
- threshold;
- model version;
- ringkasan input;
- ideal-range status;
- warning.

Jangan menampilkan prediction sebagai "jaminan" kualitas fisik. Gunakan label seperti `Prediksi Model`.

## 12. UX state

Wajib ada:

- initial state;
- loading;
- success;
- validation error;
- API unavailable;
- invalid/out-of-range input warning.

## 13. Responsive

Dashboard harus dapat dipakai pada:

- desktop;
- laptop;
- tablet.

Minimum width target 1024 px untuk demo desktop, tetapi layout tidak boleh pecah pada viewport lebih kecil.

## 14. Accessibility

Minimal:

- semantic labels;
- keyboard-accessible controls;
- visible focus;
- sufficient text contrast;
- status tidak disampaikan hanya lewat warna.
