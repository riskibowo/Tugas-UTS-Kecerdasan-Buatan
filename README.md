# Hate Speech Detection using Logistic Regression & Bag-of-Words

Project ini bertujuan untuk melakukan klasifikasi teks untuk mendeteksi **ujaran kebencian (hate speech)**. Sistem ini bekerja secara **biner**:
- **0** → Tidak mengandung ujaran kebencian
- **1** → Mengandung ujaran kebencian

---

## 📂 Dataset
Dataset yang digunakan adalah dataset ujaran kebencian berbahasa Indonesia. Format dataset harus memiliki dua kolom:
| Kolom | Deskripsi |
|------|-----------|
| `text` | Isi kalimat / ujaran |
| `label` | 0 = non-hate speech, 1 = hate speech |

Contoh isi dataset:
```
text,label
"kamu itu bodoh banget",1
"selamat pagi semuanya",0
```

---

## 🔧 Tahapan (Workflow)

1. **Load Dataset**
2. **Preprocessing Teks**
   - Lowercasing
   - Menghapus angka, tanda baca, dan simbol
   - Tokenisasi
   - Stopword Removal
   - Stemming (Sastrawi)
3. **Feature Extraction**
   - Bag-of-Words (`CountVectorizer`)
4. **Model Training**
   - Logistic Regression
5. **Evaluasi**
6. **Visualisasi & Analisis Hasil**

---

## 🧹 Contoh Preprocessing

| Sebelum | Sesudah |
|--------|---------|
| `"KAMU ITU BENAR-BENAR TIDAK BERGUNA!!!"` | `kamu benar berguna` |

---

## 🔠 Representasi Fitur (Bag-of-Words)

Fitur teks diubah menjadi matriks frekuensi menggunakan:
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_text)
```

---

## 🤖 Model Klasifikasi
Model utama yang digunakan:
```
LogisticRegression(solver="lbfgs", max_iter=1000)
```

Alasan pemilihan Logistic Regression:
- Efisien untuk klasifikasi biner
- Akurasi baik pada dataset teks dengan vektor sparse
- Mudah untuk interpretasi bobot fitur

---

## 📊 Hasil Evaluasi Model

| Metrik | Nilai |
|-------|-------|
| **Accuracy** | *0.8955* |
| **Precision** | *0.9009* |
| **Recall** | *0.8955* |
| **F1 Score** | *0.8972* |


## 🎨 Visualisasi

### 1. Distribusi Label
Digunakan untuk melihat keseimbangan dataset.

```
0 → Tidak Hate Speech
1 → Hate Speech
```

### 2. Word Cloud (Ujaran Kebencian)
Menampilkan kata yang paling sering muncul pada data hate speech.

Contoh kode:
```python
from wordcloud import WordCloud
wc = WordCloud(width=800, height=400).generate(" ".join(hate_speech_text))
plt.imshow(wc); plt.axis("off")
```

---

## 🔍 Contoh Predict Model
Gunakan kode berikut untuk uji manual:
```python
sample = ["dasar kamu tidak berguna"]
sample_clean = preprocess_text(sample[0])
vector = vectorizer.transform([sample_clean])
prediction = model.predict(vector)
print("Prediksi:", prediction[0])
```

Output:
```
Prediksi: 1  # (mengandung hate speech)
```

---

## ▶️ Cara Menjalankan di Google Colab

1. Upload dataset `.csv`
2. Jalankan block code preprocessing
3. Train model
4. Jalankan evaluasi & visualisasi
5. Coba prediksi manual

> Pastikan runtime GPU diaktifkan:
`Runtime > Change Runtime Type > GPU`

---

## 🚀 Pengembangan Lanjutan
| Upgrade | Penjelasan |
|--------|------------|
| **TF-IDF Vectorizer** | Representasi kata berbobot lebih baik |
| **SVM / Random Forest** | Model alternatif dengan performa kuat |
| **Fine-Tuning IndoBERT** | Performansi terbaik pada tugas NLP Indonesia |

---

## 📚 Referensi
- IndoBERT Model → https://huggingface.co/indobenchmark/indobert-base-p1
- Sastrawi Stemmer → https://github.com/har07/PySastrawi

---

*Dibuat sebagai bagian dari tugas UTS — Deteksi Ujaran Kebencian Menggunakan Machine Learning.*
