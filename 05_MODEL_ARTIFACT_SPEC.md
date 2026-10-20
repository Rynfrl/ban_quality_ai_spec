# 05 — Model Artifact Specification

## 1. Purpose

Artifact membuat inference reproducible dan memastikan backend/dashboard tidak memiliki preprocessing yang berbeda dengan training.

## 2. Required files

### `artifacts/model.pt`

PyTorch state dictionary.

Simpan menggunakan:

```python
torch.save(model.state_dict(), path)
```

Model architecture harus dapat direkonstruksi dari config.

### `artifacts/model_config.json`

Contoh schema:

```json
{
  "model_version": "ban-mlp-v1",
  "architecture": {
    "input_dim": 4,
    "hidden_layers": [32, 16],
    "dropout": 0.1,
    "activation": "ReLU",
    "output_dim": 1
  },
  "features": [
    "suhu",
    "tekanan",
    "waktu_curing",
    "rasio_karet"
  ],
  "target": "label",
  "label_mapping": {
    "0": "Baik",
    "1": "Perlu Perbaikan"
  },
  "threshold": 0.5,
  "pytorch_version": "",
  "numpy_version": "",
  "sklearn_version": ""
}
```

### `artifacts/scaler.json`

Simpan parameter StandardScaler dalam bentuk JSON, minimal:

```json
{
  "feature_names": ["suhu", "tekanan", "waktu_curing", "rasio_karet"],
  "mean": [],
  "scale": [],
  "var": [],
  "n_samples_seen": 0
}
```

Inference harus melakukan transform ekuivalen dengan scaler training.

### `artifacts/hasil_evaluasi.json`

Berisi dataset info, split, threshold, confusion matrix, metrics, dan metadata training.

### `artifacts/training_history.json`

Berisi history epoch.

## 3. Compatibility checks

Saat API startup:

1. cek semua artifact ada;
2. baca `model_config.json`;
3. validasi `feature_names`;
4. konstruksi model;
5. load state dict;
6. load scaler;
7. jalankan satu smoke-test inference;
8. hanya setelah semuanya sukses tandai service siap.

## 4. Artifact integrity

Jangan mengandalkan nama file saja. Metadata harus menunjukkan:

- model version;
- training timestamp;
- feature names;
- threshold;
- dataset file;
- random seed;
- framework versions.

## 5. Production rule

Artifact yang dipakai API harus sama dengan artifact yang dipakai saat test acceptance. Jangan melatih ulang saat server start.
