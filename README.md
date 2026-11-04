# Prediksi Telco Customer Churn
 
Proyek analitika data untuk memprediksi pelanggan telekomunikasi yang berisiko berhenti berlangganan (*churn*) menggunakan model klasifikasi *machine learning*.
 
| | |
|---|---|
| **Nama** | Adelia Rahmawati |
| **NIM** | F12.2024.00154 |
| **Notebook** | `project-uts.ipynb` |
 
---
 
## Daftar Isi
 
- [Latar Belakang](#latar-belakang)
- [Tujuan](#tujuan)
- [Dataset](#dataset)
- [Alur Pengerjaan](#alur-pengerjaan)
- [Hasil](#hasil)
- [Kesimpulan dan Rekomendasi](#kesimpulan-dan-rekomendasi)
- [Struktur Proyek](#struktur-proyek)
- [Cara Menjalankan](#cara-menjalankan)
---
 
## Latar Belakang
 
Banyak pelanggan telco yang berhenti berlangganan sehingga pendapatan perusahaan ikut menurun. Dari data yang dianalisis, *churn rate* saat ini mencapai **26,54%** (1.869 dari 7.043 pelanggan), artinya lebih dari seperempat pelanggan pergi.
 
- **Stakeholder:** Tim Marketing dan Tim Retensi Pelanggan.
- **Keputusan yang didukung:** menentukan pelanggan mana yang perlu diberi promo retensi, sehingga anggaran promo tepat sasaran dan tidak terbuang ke pelanggan yang memang sudah loyal.
## Tujuan
 
- **Business objective:** menurunkan *churn rate* sebesar 5% pada kuartal berikutnya.
- **Analytical objective:** membangun model klasifikasi untuk memprediksi pelanggan yang berisiko *churn* bulan depan.
- **Pertanyaan analitik:** fitur apa yang paling berpengaruh terhadap keputusan pelanggan berhenti (lama langganan, tagihan bulanan, jenis layanan)?
## Dataset
 
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Ukuran awal:** 7.043 baris × 21 kolom
- **Target:** `Churn` (Yes/No)
- **Isi data:** profil pelanggan, layanan yang digunakan, jenis kontrak, tagihan, dan label `Churn`.
- **Catatan:** kelas target **tidak seimbang** (±74% bertahan vs ±26% churn).
 
## Alur Pengerjaan
 
### 1. Menentukan Tujuan dan Kebutuhan Pemodelan
Merumuskan masalah bisnis, *business objective*, *analytical objective*, dan menghitung *churn rate* awal.
 
### 2. Menelaah, Memvalidasi, dan Menyiapkan Data
 
**Eksplorasi.** Analisis statistik deskriptif dan *bar plot* `tenure` serta `MonthlyCharges` terhadap `Churn`. Pelanggan yang churn rata-rata memiliki **lama langganan lebih pendek** dan **tagihan bulanan lebih tinggi**.
 
**Hipotesis awal:**
1. Pelanggan baru (tenure pendek) lebih rawan churn daripada pelanggan lama.
2. Makin tinggi tagihan bulanan, makin besar kemungkinan churn.
3. Pelanggan kontrak 1 atau 2 tahun lebih kecil kemungkinan churn daripada kontrak bulanan (masih dugaan, belum diuji).
**Pembersihan data:**
 
| Temuan | Penanganan | Alasan |
|---|---|---|
| `TotalCharges` terbaca sebagai teks karena ada spasi kosong (pelanggan baru, tenure = 0) | Spasi diubah menjadi NaN, tipe diubah ke numerik, lalu 11 baris NaN dihapus | Jumlahnya sangat sedikit, lebih aman dihapus daripada diimputasi |
| Data duplikat | Dihapus (22 baris) | Menjaga konsistensi data |
| `customerID` | Dibuang | Hanya ID unik, tidak punya pola prediktif |
 
Hasil: 7.043 baris menjadi **7.010 baris** (33 baris dihapus), dan 21 kolom menjadi 20 kolom.
 
**Konstruksi data:**
1. **Target encoding:** `Churn` Yes/No diubah menjadi 1/0.
2. **One-hot encoding:** fitur kategorikal diubah menjadi kolom biner dengan `drop_first=True`.
3. **Standardisasi:** `tenure`, `MonthlyCharges`, dan `TotalCharges` distandardisasi dengan `StandardScaler`.
Hasil akhir (*analytical dataset*) berukuran **7.010 baris × 31 kolom** (30 fitur + 1 target) dan disimpan ke `analytical_dataset.csv`.
 
### 3. Merancang Skenario Model dan Pengujian
 
- **Jenis masalah:** klasifikasi (*supervised learning*).
- **Model utama:** Random Forest, karena tahan terhadap outlier dan mampu menangkap pola non-linear.
- **Model pembanding (baseline):** Logistic Regression, karena sederhana, cepat, dan mudah diinterpretasi.
- **Skenario pengujian:** *percentage split* **80:20** dengan `stratify=y` agar proporsi churn sama di data latih dan data uji.
- **Metrik utama:** Recall dan F1-Score kelas Churn, dengan Accuracy sebagai gambaran umum. Salah menebak pelanggan yang akan pergi sebagai "aman" (*False Negative*) lebih merugikan daripada sebaliknya.
- **Kriteria keberhasilan:** Accuracy > 75% dan F1-Score kelas Churn > 0,55.
### 4. Membangun dan Menguji Model
 
- **Baseline:** `LogisticRegression(max_iter=1000, random_state=42)`.
- **Optimasi Random Forest** dengan `GridSearchCV` (`cv=3`, `scoring='f1'`):
| Parameter | Nilai yang diuji |
|---|---|
| `n_estimators` | 50, 100 |
| `max_depth` | 5, 10 |
 
- **Parameter terbaik:** `max_depth=10`, `n_estimators=50`.
### 5. Evaluasi dan Review
Kedua model diuji pada data uji (`X_test`) yang tidak dipakai saat pelatihan maupun *tuning*, kemudian dianalisis menggunakan *classification report* dan *confusion matrix*.
 
## Hasil
 
Evaluasi pada data uji (metrik Precision, Recall, dan F1 untuk kelas Churn = 1):
 
| Model | Accuracy | Precision | Recall | F1-Score | Lolos kriteria? |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0,8074 | 0,67 | 0,54 | 0,60 | Ya |
| Random Forest (optimasi) | 0,8010 | 0,66 | 0,51 | 0,57 | Ya |
 
Kedua model memenuhi kriteria keberhasilan. Logistic Regression yang sederhana justru sedikit lebih unggul dibanding Random Forest hasil *tuning* pada semua metrik.
 
## Kesimpulan dan Rekomendasi
 
**Keterbatasan:**
- Recall churn hanya sekitar 51-54%, sehingga hampir separuh pelanggan yang akan pergi masih lolos dari deteksi. Penyebabnya adalah data yang tidak seimbang.
- *Grid search* masih kecil (2 parameter × 2 nilai), sehingga masih ada ruang eksplorasi.
**Rekomendasi teknis:**
- Menyeimbangkan data (SMOTE atau `class_weight`).
- Menurunkan *threshold* prediksi untuk meningkatkan recall.
- Memperluas ruang pencarian *grid search*.
**Rekomendasi bisnis:**
- Tim Marketing dapat memakai Logistic Regression (sedikit lebih baik dan mudah dijelaskan) untuk membuat daftar pelanggan berisiko churn dan memberi promo retensi.
- Target penurunan churn 5% sebaiknya dibuktikan terlebih dahulu lewat uji coba kampanye.
## Struktur Proyek
 
```
.
├── project-uts.ipynb                      # Notebook utama (analisis dan pemodelan)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset mentah (disertakan)
├── analytical_dataset.csv                 # Dataset hasil preprocessing (dihasilkan notebook)
└── README.md
```
 
## Cara Menjalankan
 
1. **Clone repositori**
```bash
   git clone <url-repositori>
   cd <nama-folder>
```
 
2. **Install dependensi**
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```
 
3. **Siapkan dataset:** letakkan `WA_Fn-UseC_-Telco-Customer-Churn.csv` di folder yang sama dengan notebook.
4. **Jalankan notebook**
```bash
   jupyter notebook project-uts.ipynb
```
   Lalu jalankan seluruh sel secara berurutan (*Run All*).
 
## Teknologi
 
- Python 3
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn (`StandardScaler`, `train_test_split`, `LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, metrik evaluasi)
