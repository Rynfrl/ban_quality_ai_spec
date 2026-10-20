# 03 — Machine Learning Specification

## 1. Task

Binary classification:

```text
X -> P(label = 1)
```

Interpretasi:

```text
label 0 = Baik
label 1 = Perlu Perbaikan
```

## 2. Input vector

Urutan feature wajib:

```text
[suhu, tekanan, waktu_curing, rasio_karet]
```

Jangan mengubah urutan di inference.

## 3. Preprocessing

Default:

```text
Raw input
  -> numeric conversion
  -> missing handling
  -> StandardScaler
  -> Tensor float32
  -> MLP
```

Scaler harus di-fit hanya pada training data.

## 4. Neural network

Recommended baseline:

```text
Linear(4, 32)
ReLU
Linear(32, 16)
ReLU
Dropout(0.10)
Linear(16, 1)
```

Forward pass mengembalikan logit.

Inference:

```python
probability_bad = torch.sigmoid(logit).item()
probability_good = 1.0 - probability_bad
```

Decision:

```python
label = 1 if probability_bad >= threshold else 0
```

Threshold berasal dari artifact/config. Default `0.50`.

## 5. Loss

Gunakan:

```python
torch.nn.BCEWithLogitsLoss(pos_weight=...)
```

`pos_weight` dihitung dari data training:

```text
negative_count / positive_count
```

Hal ini membuat kelas `Perlu Perbaikan` lebih diperhatikan tanpa melakukan oversampling secara default.

## 6. Optimizer

Default:

```text
Adam
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
max_epochs = 200
early_stopping_patience = 20
```

Semua parameter harus dapat dikonfigurasi.

## 7. Reproducibility

Set seed:

- Python random;
- NumPy;
- PyTorch.

Bila memungkinkan aktifkan deterministic behavior. Dokumentasikan trade-off performa bila diperlukan.

## 8. Metrics

Wajib:

- accuracy;
- precision;
- recall;
- F1;
- ROC-AUC;
- confusion matrix.

Tambahkan:

- specificity;
- balanced accuracy;
- average precision bila tersedia.

## 9. Threshold selection

Jangan memilih threshold berdasarkan test set.

Alur:

```text
Train -> fit model
Validation -> choose threshold
Test -> final one-time evaluation
```

Default threshold 0.50.

Opsional mode bisnis:

- target minimum recall untuk kelas 1;
- atau threshold yang memaksimalkan F1 pada validation set.

Threshold yang dipilih harus disimpan dalam `model_config.json`.

## 10. Training history

Simpan per epoch minimal:

```json
{
  "epoch": 1,
  "train_loss": 0.123,
  "val_loss": 0.145,
  "train_accuracy": 0.90,
  "val_accuracy": 0.88
}
```

History dapat disimpan sebagai JSON.

## 11. Baseline comparison

Program boleh menyediakan baseline Logistic Regression dari scikit-learn sebagai sanity check. Baseline bukan model production.

Tujuan baseline:

- memastikan data pipeline valid;
- mendeteksi apakah MLP justru jauh lebih buruk;
- memberi konteks performa.

## 12. Error analysis

Setelah test evaluation:

- tampilkan confusion matrix;
- daftar false negative (`actual=1, predicted=0`);
- daftar false positive (`actual=0, predicted=1`);
- analisis input feature pada error jika memungkinkan.

Jangan menghapus error record dari hasil evaluasi.

## 13. Model version

Gunakan version sederhana, misalnya:

```text
ban-mlp-v1
```

Version dicatat pada artifact metadata dan response API.
