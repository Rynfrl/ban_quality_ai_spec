# 02 — Data Specification

## 1. Source

File: `data_ban.xlsx`

Sheet:

- `data_ban`
- `keterangan`

Sheet `data_ban` berisi 500 record.

## 2. Schema

| Column | Type | Role | Unit | Dataset min | Dataset max |
|---|---|---|---|---:|---:|
| `suhu` | float | production input | °C | 140.2 | 180 |
| `tekanan` | float | production input | bar | 15 | 25 |
| `waktu_curing` | float | production input | menit | 8 | 15 |
| `rasio_karet` | float | production input | - | 0.4 | 0.8 |
| `kekerasan_karet` | float | offline quality/analysis feature | Shore A | 55.1 | 75 |
| `label` | int | target | - | 0 | 1 |

## 3. Business ranges

Referensi sheet `keterangan`:

| Feature | Minimum | Maximum | Ideal |
|---|---:|---:|---:|
| `suhu` | 140 | 180 | 150–165 °C |
| `tekanan` | 15 | 25 | 18–22 bar |
| `waktu_curing` | 8 | 15 | 10–13 menit |
| `rasio_karet` | 0.4 | 0.8 | 0.5–0.7 |
| `kekerasan_karet` | 55 | 75 | 60–70 Shore A |

## 4. Label

```text
0 = Baik
1 = Perlu Perbaikan
```

Pada dataset yang diberikan:

- label `0`: 325 record
- label `1`: 175 record
- proporsi label `0`: 65%
- proporsi label `1`: 35%

Distribusi ini tidak cukup ekstrem untuk mengharuskan oversampling. Gunakan stratification dan pertimbangkan `pos_weight` yang dihitung dari train set.

## 5. Data quality baseline

Dataset yang diberikan:

- missing value: 0 pada kolom yang diperiksa;
- duplicate complete row: 0;
- seluruh nilai label berada pada {0, 1}.

Walaupun dataset bersih, implementasikan validasi agar pipeline tetap aman terhadap dataset baru.

## 6. Feature policy

### Production model

Feature vector:

```python
FEATURES = [
    "suhu",
    "tekanan",
    "waktu_curing",
    "rasio_karet",
]
TARGET = "label"
```

### Why hardness is excluded

`kekerasan_karet` adalah pengukuran kualitas/hasil fisik. Memasukkannya ke model live dapat membuat sistem terlihat akurat karena memakai informasi yang kemungkinan belum tersedia sebelum pengujian fisik. Ini bertentangan dengan tujuan "prediksi tanpa harus diuji fisik".

Boleh dibuat eksperimen kedua bernama `offline_with_hardness`, tetapi harus:

- dipisahkan dari model production;
- disebut eksplisit pada laporan;
- tidak dipakai dashboard live default;
- tidak menimpa artifact production.

## 7. Range policy

Ada tiga level:

1. **Hard validation**: type, NaN/inf, dan batas domain yang tidak masuk akal.
2. **Dataset range warning**: input di luar observed training range.
3. **Ideal range indicator**: apakah input masuk rentang ideal proses.

Input di luar ideal range tidak otomatis berarti label `1`. Model tetap menjadi sumber prediksi, sedangkan ideal-range indicator adalah business context.

## 8. Data leakage rules

Dilarang:

- fit scaler pada seluruh dataset sebelum split;
- memilih threshold memakai test set;
- tuning hyperparameter memakai test set;
- memilih feature berdasarkan test performance;
- menggunakan `kekerasan_karet` pada production model tanpa justifikasi ketersediaan saat prediction time.
