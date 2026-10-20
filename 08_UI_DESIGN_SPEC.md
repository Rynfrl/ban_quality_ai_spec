# 08 — UI Design Specification

## 1. Visual direction

Karakter visual yang dituju:

- modern;
- profesional;
- industrial/data-oriented;
- banyak whitespace;
- card surfaces;
- informasi padat tetapi mudah dipindai;
- bukan tampilan admin template generik.

## 2. Layout

Recommended:

```text
┌──────────────────────────────────────────────┐
│ Header / project title / model version       │
├──────────────────────────────────────────────┤
│ Tab navigation                               │
├───────────────┬──────────────────────────────┤
│ Sidebar/meta  │ Main content                 │
│ optional      │ cards / charts / form        │
└───────────────┴──────────────────────────────┘
```

Untuk layout sederhana, sidebar boleh dihilangkan dan menggunakan horizontal navigation.

## 3. Components

Minimal component set:

- Header;
- Navigation tabs;
- KPI card;
- Information card;
- Feature input card;
- Ideal range indicator;
- Prediction result card;
- Probability bar;
- Confusion matrix;
- Line chart;
- Status badge;
- Warning alert;
- Artifact list;
- Code/architecture block.

## 4. Prediction status

Jangan hanya mengandalkan warna:

```text
Baik
Perlu Perbaikan
```

Warna hanya sebagai secondary encoding.

## 5. Content tone

Gunakan Bahasa Indonesia teknis yang jelas.

Contoh:

- `Prediksi Model`
- `Probabilitas Perlu Perbaikan`
- `Rentang Ideal`
- `Di luar rentang dataset`
- `Parameter valid`
- `Model aktif`

Hindari klaim absolut seperti:

- "pasti baik";
- "dijamin lolos";
- "100% akurat".

## 6. Charts

Chart default:

1. training vs validation loss;
2. optional metric comparison;
3. confusion matrix.

Jangan menampilkan chart yang tidak memiliki interpretasi.

## 7. Responsive card behavior

Pada desktop:

```text
KPI: 4 columns
Content: 2 columns
```

Pada tablet:

```text
KPI: 2 columns
Content: 1 column
```

## 8. Frontend architecture

Untuk implementasi ringan:

```text
frontend/
  static/
    css/
    js/
    img/
  templates/
```

Backend FastAPI dapat serve static assets dan template Jinja2, atau static frontend dapat berjalan terpisah bila diperlukan. Untuk local demo, satu server lebih sederhana.
