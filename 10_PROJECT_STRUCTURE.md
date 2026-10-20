# 10 — Recommended Project Structure

Gunakan struktur berikut sebagai default:

```text
ban-quality-ml/
├── data/
│   └── data_ban.xlsx
│
├── artifacts/
│   ├── model.pt
│   ├── model_config.json
│   ├── scaler.json
│   ├── hasil_evaluasi.json
│   └── training_history.json
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   ├── artifact_loader.py
│   │   └── range_checker.py
│   │
│   └── web/
│       ├── templates/
│       │   └── index.html
│       └── static/
│           ├── css/
│           │   └── app.css
│           └── js/
│               └── app.js
│
├── ml/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── artifacts.py
│
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_api.py
│   └── test_artifacts.py
│
├── docs/
│   └── ...
│
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## Module responsibilities

### `ml/data.py`

- read Excel;
- validate schema;
- return DataFrame/features/target.

### `ml/preprocessing.py`

- split;
- fit scaler;
- transform;
- serialize scaler.

### `ml/model.py`

- define `BanQualityMLP`;
- model factory from config.

### `ml/train.py`

- CLI;
- training loop;
- early stopping;
- checkpoint.

### `ml/evaluate.py`

- test metrics;
- confusion matrix;
- JSON report.

### `ml/artifacts.py`

- save/load model metadata;
- save/load scaler;
- compatibility checks.

### `app/services/predictor.py`

- load artifact;
- preprocessing;
- inference;
- response assembly.

### `app/services/range_checker.py`

- dataset range;
- ideal range;
- warnings.

### `app/schemas.py`

- Pydantic request/response models.

### `app/main.py`

- FastAPI app;
- routes;
- startup artifact loading;
- static/template serving.
