# 06 — REST API Specification

## 1. Base

Default local URL:

```text
http://127.0.0.1:8000
```

Swagger:

```text
/docs
```

## 2. GET `/health`

Response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "ban-mlp-v1"
}
```

Status harus menunjukkan error bila artifact tidak valid.

## 3. GET `/api/model-info`

Response minimal:

```json
{
  "model_version": "ban-mlp-v1",
  "features": [
    "suhu",
    "tekanan",
    "waktu_curing",
    "rasio_karet"
  ],
  "threshold": 0.5,
  "label_mapping": {
    "0": "Baik",
    "1": "Perlu Perbaikan"
  },
  "metrics": {
    "accuracy": 0.0,
    "precision_class_1": 0.0,
    "recall_class_1": 0.0,
    "f1_class_1": 0.0,
    "roc_auc": 0.0
  }
}
```

Nilai metric diambil dari artifact evaluation, bukan dihitung ulang setiap request.

## 4. POST `/api/predict`

Request:

```json
{
  "suhu": 160.0,
  "tekanan": 20.0,
  "waktu_curing": 11.0,
  "rasio_karet": 0.60
}
```

Response:

```json
{
  "label": 0,
  "label_name": "Baik",
  "probability_good": 0.92,
  "probability_bad": 0.08,
  "threshold": 0.5,
  "model_version": "ban-mlp-v1",
  "feature_checks": {
    "suhu": {
      "value": 160.0,
      "in_dataset_range": true,
      "in_ideal_range": true,
      "ideal_min": 150.0,
      "ideal_max": 165.0,
      "unit": "°C"
    }
  },
  "warnings": []
}
```

Jangan mengembalikan path file lokal, stack trace, atau secret.

## 5. Optional POST `/api/predict/batch`

Gunakan hanya bila dashboard/reporting membutuhkan batch prediction.

Request:

```json
{
  "items": [
    {
      "suhu": 160,
      "tekanan": 20,
      "waktu_curing": 11,
      "rasio_karet": 0.60
    }
  ]
}
```

## 6. Validation

Pydantic harus memastikan:

- semua field tersedia;
- tipe numeric;
- finite values;
- batas domain tidak absurd.

Use business range as warning unless project owner explicitly decides it must be hard rejection.

## 7. Error codes

Minimal:

- `400` malformed payload;
- `422` schema validation error;
- `503` model unavailable;
- `500` unexpected internal error.

Untuk prediction error, beri pesan aman dan log exception internal di server.
