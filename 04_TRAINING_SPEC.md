# 04 — Training Pipeline Specification

## 1. Pipeline stages

Training harus berjalan dengan urutan:

```text
Load Excel
  -> Validate schema
  -> Clean/coerce numeric values
  -> Split train/validation/test
  -> Fit preprocessing on train only
  -> Transform train/validation/test
  -> Build MLP
  -> Train
  -> Select threshold on validation
  -> Evaluate once on test
  -> Save artifacts
  -> Write evaluation JSON
```

## 2. File entry point

Recommended CLI:

```bash
python -m ml.train
```

Optional arguments:

```bash
python -m ml.train ^
  --data data/data_ban.xlsx ^
  --sheet data_ban ^
  --epochs 200 ^
  --batch-size 32 ^
  --lr 0.001 ^
  --seed 42
```

Gunakan `argparse` atau CLI library ringan.

## 3. Data split

Gunakan:

```python
train_test_split(..., test_size=0.30, stratify=y, random_state=42)
```

kemudian split sementara menjadi validation/test 50:50.

Hasil akhir:

```text
70% train
15% validation
15% test
```

## 4. Training loop

Minimal harus memiliki:

- DataLoader;
- `model.train()`;
- forward;
- loss;
- backward;
- optimizer step;
- `model.eval()` untuk validation;
- no-grad validation;
- checkpoint terbaik berdasarkan validation loss atau validation F1;
- early stopping.

Jangan menggunakan test set di loop training.

## 5. Best checkpoint

Simpan state model terbaik berdasarkan metric yang konsisten. Rekomendasi:

1. primary: validation loss;
2. secondary/reporting: validation F1 dan recall class 1.

Simpan hanya checkpoint terbaik sebagai production artifact.

## 6. Output directory

Default:

```text
artifacts/
  model.pt
  model_config.json
  scaler.json
  hasil_evaluasi.json
  training_history.json
```

## 7. Evaluation JSON

`hasil_evaluasi.json` minimal:

```json
{
  "model_version": "ban-mlp-v1",
  "dataset": {
    "file": "data_ban.xlsx",
    "sheet": "data_ban",
    "rows": 500
  },
  "features": [
    "suhu",
    "tekanan",
    "waktu_curing",
    "rasio_karet"
  ],
  "target": "label",
  "split": {
    "train": 350,
    "validation": 75,
    "test": 75,
    "random_state": 42
  },
  "threshold": 0.5,
  "metrics": {
    "accuracy": null,
    "precision_class_1": null,
    "recall_class_1": null,
    "f1_class_1": null,
    "roc_auc": null,
    "balanced_accuracy": null
  },
  "confusion_matrix": [[0, 0], [0, 0]]
}
```

Nilai metrics harus berasal dari eksekusi nyata, bukan placeholder, pada artifact final.

## 8. Failure handling

Training harus gagal dengan pesan yang jelas bila:

- file tidak ditemukan;
- sheet tidak ditemukan;
- kolom wajib hilang;
- target tidak binary;
- data training kosong;
- fitur tidak numeric;
- model artifact gagal ditulis.

Jangan menelan exception secara diam-diam.
