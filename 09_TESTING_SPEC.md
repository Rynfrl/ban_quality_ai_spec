# 09 — Testing Specification

## 1. Unit tests

### Data validation

Test:

- valid schema accepted;
- missing required column rejected;
- invalid label rejected;
- non-numeric feature rejected;
- NaN/inf handled/rejected according to policy;
- duplicate detection works.

### Preprocessing

Test:

- scaler output shape correct;
- feature order preserved;
- scaler fit only on train;
- saved scaler reproduces transform values.

### Model

Test:

- model accepts shape `(batch, 4)`;
- output shape `(batch, 1)`;
- no NaN logits for valid input;
- artifact reload gives equivalent prediction.

### Threshold

Test:

- probability below threshold => label 0;
- probability at threshold => label 1;
- probability above threshold => label 1.

## 2. Integration tests

### Training

Run training on dataset fixture and verify:

- artifacts created;
- JSON valid;
- metrics populated;
- confusion matrix dimensions 2x2;
- feature names exactly match production list.

### API

Test:

- `GET /health`;
- `GET /api/model-info`;
- `POST /api/predict`;
- malformed payload returns validation error;
- model unavailable returns `503`.

## 3. Frontend tests

At minimum verify:

- form renders;
- submit calls API;
- result card renders;
- loading state renders;
- error state renders;
- all four production features are present;
- hardness does not appear as a live required input.

## 4. Acceptance test

A fresh environment must be able to:

```text
install dependencies
-> train model
-> start API
-> open dashboard
-> enter sample input
-> obtain prediction
```

## 5. Model sanity tests

Use at least:

### Test input A — ideal center

```json
{
  "suhu": 157.5,
  "tekanan": 20.0,
  "waktu_curing": 11.5,
  "rasio_karet": 0.60
}
```

### Test input B — lower temperature boundary

```json
{
  "suhu": 140.0,
  "tekanan": 20.0,
  "waktu_curing": 11.5,
  "rasio_karet": 0.60
}
```

### Test input C — upper temperature boundary

```json
{
  "suhu": 180.0,
  "tekanan": 20.0,
  "waktu_curing": 11.5,
  "rasio_karet": 0.60
}
```

Prediksi aktual untuk ketiga input harus dihasilkan oleh model setelah training; jangan hard-code expected label tanpa bukti empiris. Test hanya assertion pada schema, finite probability, dan valid output.

## 6. No silent fallback

Jika model artifact tidak ada, aplikasi tidak boleh diam-diam membuat model random dan menerima prediction. Service harus menunjukkan model unavailable.
