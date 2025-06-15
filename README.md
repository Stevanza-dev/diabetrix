# Diabetrix

Diabetrix adalah aplikasi web berbasis Flask yang memanfaatkan kecerdasan buatan untuk mendeteksi risiko diabetes tipe 2 berdasarkan data medis pengguna. Proyek ini menyediakan antarmuka yang ramah pengguna, fitur riwayat pengecekan, serta statistik risiko yang mudah dipahami.

---

## Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Metode dan Teknologi](#metode-dan-teknologi)
- [Struktur Proyek](#struktur-proyek)
- [Cara Instalasi & Menjalankan](#cara-instalasi--menjalankan)
- [Cara Penggunaan](#cara-penggunaan)
- [Tim Pengembang](#tim-pengembang)
- [Lisensi](#lisensi)

---

## Fitur Utama

- **Prediksi Diabetes:** Menggunakan model machine learning ensemble untuk memprediksi risiko diabetes berdasarkan input medis.
- **Riwayat Pengecekan:** Menyimpan dan menampilkan riwayat pengecekan pengguna beserta statistik risiko.
- **Filter & Ekspor:** Filter riwayat berdasarkan tanggal dan fitur ekspor (dalam pengembangan).
- **Dashboard Statistik:** Menampilkan statistik jumlah pengecekan, risiko tinggi, dan risiko rendah.
- **Antarmuka Modern:** Menggunakan Tailwind CSS dan Font Awesome untuk tampilan yang responsif dan modern.

---

## Metode dan Teknologi

### 1. **Machine Learning**

- **Dataset:** Menggunakan dataset [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (file: `Diabetes.xlsx`).
- **Preprocessing:**
  - Mengganti nilai 0 pada kolom medis tertentu (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) dengan nilai rata-rata kolom.
  - Standarisasi fitur menggunakan `StandardScaler`.
- **Model:**
  - **Ensemble Voting Classifier** (Soft Voting) yang terdiri dari:
    - Logistic Regression
    - Random Forest Classifier
    - XGBoost Classifier
  - Model dilatih pada data training dan dievaluasi menggunakan data testing.
- **Penyimpanan Model:**
  - Model dan scaler disimpan dalam folder `model/` menggunakan `joblib`.

### 2. **Backend**

- **Flask:** Framework utama untuk backend dan routing.
- **SQLite:** Database lokal untuk menyimpan riwayat pengecekan pengguna (`diabetes_history.db`).
- **Pandas & Numpy:** Untuk manipulasi data dan preprocessing.
- **Joblib:** Untuk serialisasi model dan scaler.

### 3. **Frontend**

- **HTML5 & Jinja2:** Template engine untuk rendering dinamis.
- **Tailwind CSS:** Framework CSS untuk desain responsif dan modern.
- **Font Awesome:** Ikon-ikon untuk mempercantik tampilan.
- **JavaScript:** Untuk fitur interaktif seperti filter tanggal, hapus riwayat, dan ekspor data.

---

## Struktur Proyek

```
.
├── app.py
├── diabetes_history.db
├── Diabetes.xlsx
├── README.md
├── requirement.txt
├── train_model.py
├── model/
│   ├── diabetes_ensemble_model.joblib
│   ├── model.pkl
│   └── scaler.joblib
├── static/
│   ├── banu.jpg
│   ├── faizal.png
│   ├── isna.jpg
│   └── stevan.JPG
└── templates/
    ├── about.html
    ├── detect.html
    ├── history.html
    └── index.html
```

- **app.py:** Main Flask app, routing, dan logic backend.
- **train_model.py:** Script training dan penyimpanan model ML.
- **model/:** Folder penyimpanan model dan scaler hasil training.
- **static/:** Asset gambar untuk tampilan web.
- **templates/:** File HTML dengan Jinja2 untuk frontend.

---

## Cara Instalasi & Menjalankan

1. **Clone Repository**
    ```sh
    git clone https://github.com/username/diabetrix.git
    cd diabetrix
    ```

2. **Buat Virtual Environment (Opsional tapi Disarankan)**
    ```sh
    python -m venv venv
    venv\Scripts\activate  # Windows
    # source venv/bin/activate  # Linux/Mac
    ```

3. **Install Dependencies**
    ```sh
    pip install flask joblib numpy pandas scikit-learn xgboost openpyxl
    ```

4. **Training Model (Opsional, jika model sudah tersedia di folder model/)**
    ```sh
    python train_model.py
    ```

5. **Jalankan Aplikasi**
    ```sh
    python app.py
    ```

6. **Akses di Browser**
    ```
    http://localhost:5000
    ```

---

## Cara Penggunaan

1. **Home:** Lihat deskripsi aplikasi dan fitur utama.
2. **Detect:** Masukkan data medis Anda (kehamilan, glukosa, tekanan darah, dll), lalu klik "Prediksi". Hasil prediksi dan tingkat kepercayaan akan muncul.
3. **History:** Lihat riwayat pengecekan Anda, filter berdasarkan tanggal, dan hapus data jika diperlukan.
4. **About:** Informasi tentang tim pengembang dan tujuan aplikasi.

---

## Tim Pengembang

- **Itsna Sabila Hidayati**  
  *Laporan Projek Akhir*  
  [Instagram](https://www.instagram.com/sabitsna_/)

- **Bhanu Rizqi Marzaki**  
  *ML Model Developer*  
  [Instagram](https://www.instagram.com/nolluiymon/)

- **Faizal Rifky Abdilah**  
  *Laporan Projek Akhir*  
  [Instagram](https://www.instagram.com/faizal_rifky100/)

- **M. Stevanza Sylvester**  
  *Frontend Developer*  
  [Instagram](https://www.instagram.com/stevanzasyl/)

---

## Lisensi

Proyek ini dibuat untuk tujuan edukasi dan non-komersial. Silakan gunakan, modifikasi, dan distribusikan dengan mencantumkan kredit kepada tim pengembang.

---

## Kontribusi

Kontribusi sangat terbuka! Silakan buat pull request atau buka issue untuk perbaikan dan pengembangan fitur baru.

---

## Kontak

Untuk pertanyaan lebih lanjut, silakan hubungi kami melalui halaman [Contact](#) di aplikasi.

---

Terima