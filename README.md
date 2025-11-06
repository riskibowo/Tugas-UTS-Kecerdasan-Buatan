# Tugas-UTS-Kecerdasan-Buatan

# #############################################################
# ## 🚀 1. Setup dan Import Library
# #############################################################
#
# Kita impor semua library yang dibutuhkan di awal
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from wordcloud import WordCloud

# Mengunduh resource NLTK yang diperlukan
print("Mengunduh NLTK package...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK (punkt, stopwords) siap.")
except Exception as e:
    print(f"Error mengunduh NLTK: {e}")

print("-" * 50)
<img width="522" height="220" alt="image" src="https://github.com/user-attachments/assets/1ea92132-c2ca-4b72-b45e-8278c5aded41" />



# #############################################################
# ## 📥 2. Langkah 1: Memuat dan Menyiapkan Dataset
# #############################################################
print("\n[Langkah 1: Memuat Dataset]")

try:
    # Membaca file CSV dari dataset Davidson
    # index_col=0 digunakan agar kolom 'Unnamed: 0' menjadi index
    df = pd.read_csv('labeled_data.csv', index_col=0)

    # Menyesuaikan Label (Sesuai Soal [cite: 13, 14, 15])
    # Soal: 0 = tidak mengandung, 1 = mengandung
    # Dataset: 0 = hate, 1 = offensive, 2 = neither
    # Maka: 0 dan 1 -> 1 (mengandung), 2 -> 0 (tidak mengandung)
    
    df['label'] = df['class'].apply(lambda x: 0 if x == 2 else 1)
    
    # Membuat DataFrame baru yang lebih bersih
    df_bersih = df[['tweet', 'label']].copy()
    
    print("Dataset berhasil dimuat dan label disesuaikan.")
    print(df_bersih.head())
    
except FileNotFoundError:
    print("ERROR: File 'labeled_data.csv' tidak ditemukan.")
    print("Pastikan Anda sudah meng-upload file tersebut ke Colab.")
except Exception as e:
    print(f"ERROR saat memuat data: {e}")

print("-" * 50)
<img width="469" height="179" alt="image" src="https://github.com/user-attachments/assets/9f9e514f-c96f-4886-8eb5-abdc178a6835" />



# #############################################################
# ## 🧹 3. Langkah 2: Preprocessing Teks
# #############################################################
print("\n[Langkah 2: Preprocessing Teks]")

# Inisialisasi stopword Bahasa Inggris
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Case folding (ubah ke huruf kecil) [cite: 18]
    text = text.lower()
    
    # 2. Hapus URL, username (@), hashtag (#), dan karakter non-alfabet
    # (Termasuk menghapus tanda baca dan angka [cite: 19])
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Tokenisasi [cite: 20]
    tokens = word_tokenize(text)
    
    # 4. Stopword removal [cite: 21]
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Gabungkan kembali token
    return " ".join(filtered_tokens)

# Terapkan fungsi preprocessing
print("Memulai proses preprocessing data teks...")
# Hanya proses jika df_bersih ada (untuk menghindari error jika file not found)
if 'df_bersih' in locals():
    df_bersih['teks_bersih'] = df_bersih['tweet'].apply(preprocess_text)
    print("Preprocessing selesai.")
    print(df_bersih[['teks_bersih', 'label']].head())
else:
    print("Preprocessing dilewati karena dataset tidak ter-load.")

print("-" * 50)
<img width="595" height="92" alt="image" src="https://github.com/user-attachments/assets/320a65e7-5fb6-4550-977a-dd0d6468879f" />


# #############################################################
# ## 🔢 4. Langkah 3: Representasi Fitur (TF-IDF)
# #############################################################
print("\n[Langkah 3: Representasi Fitur (TF-IDF)]")

if 'df_bersih' in locals():
    # Pisahkan data (X) dan label (y)
    X = df_bersih['teks_bersih']
    y = df_bersih['label']

    # Bagi data menjadi 80% data latih (train) dan 20% data uji (test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Inisialisasi TF-IDF Vectorizer [cite: 26]
    # max_features=5000 berarti kita hanya ambil 5000 kata paling penting
    tfidf_vectorizer = TfidfVectorizer(max_features=22650)

    # 'fit_transform' HANYA pada data latih (X_train)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # 'transform' pada data uji (X_test)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(f"Data latih (train) diubah menjadi matriks TF-IDF dengan shape: {X_train_tfidf.shape}")
    print(f"Data uji (test) diubah menjadi matriks TF-IDF dengan shape: {X_test_tfidf.shape}")
else:
    print("Langkah 3 dilewati karena data tidak siap.")
    
print("-" * 50)

<img width="421" height="65" alt="image" src="https://github.com/user-attachments/assets/257a8a4c-a3e9-4175-bc5c-87fe5416259d" />

# #############################################################
# ## 🤖 5. Langkah 4: Pembangunan Model (Naive Bayes)
# #############################################################
print("\n[Langkah 4: Pembangunan Model (Naive Bayes)]")

if 'X_train_tfidf' in locals():
    # Kita menggunakan Naive Bayes (MultinomialNB) [cite: 30]
    model_nb = MultinomialNB()

    # Latih model menggunakan data latih TF-IDF
    model_nb.fit(X_train_tfidf, y_train)

    print("Model Naive Bayes berhasil dilatih.")
else:
    print("Langkah 4 dilewati karena data latih tidak siap.")

print("-" * 50)
<img width="841" height="413" alt="image" src="https://github.com/user-attachments/assets/b0d6cef2-76b6-4998-8118-d9c84ef11464" />


# #############################################################
# ## 📊 6. Langkah 5: Evaluasi Model
# #############################################################
print("\n[Langkah 5: Evaluasi Model]")

if 'model_nb' in locals():
    # Lakukan prediksi pada data uji
    y_pred = model_nb.predict(X_test_tfidf)

    # Tampilkan hasil evaluasi
    
    # 1. Accuracy [cite: 35]
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model: {accuracy * 100:.2f}%\n")

    # 2. Confusion Matrix [cite: 34]
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\n")
    
    # 3. Precision, Recall, F1-Score [cite: 36, 37, 38]
    print("Laporan Klasifikasi (Precision, Recall, F1-Score):")
    # 'target_names' membantu membaca label 0 dan 1
    report = classification_report(y_test, y_pred, target_names=['Bukan Hate Speech (0)', 'Hate Speech (1)'])
    print(report)

    print("\n--- Interpretasi Singkat  ---")
    print(" - Akurasi: Persentase total prediksi yang benar.")
    print(" - Precision (Label 1): Dari semua yang diprediksi 'Hate Speech', berapa persen yang benar.")
    print(" - Recall (Label 1): Dari semua 'Hate Speech' asli, berapa persen yang berhasil dideteksi model.")
    print(" - F1-Score: Rata-rata harmonis dari Precision dan Recall.")
    
else:
    print("Langkah 5 dilewati karena model tidak dilatih.")

print("-" * 50)
<img width="834" height="495" alt="image" src="https://github.com/user-attachments/assets/de40b4bc-a92e-4048-a905-08cd57091bf5" />


# #############################################################
# ## 📈 7. Langkah 6: Visualisasi
# #############################################################
print("\n[Langkah 6: Visualisasi]")

if 'df_bersih' in locals():
    # 1. Visualisasi Distribusi Label 
    print("Menampilkan visualisasi 'Distribusi Label'...")
    plt.figure(figsize=(7, 5))
    sns.countplot(x='label', data=df_bersih)
    plt.title('Distribusi Label (0: Bukan Hate, 1: Mengandung Hate/Offensive)')
    plt.xlabel('Label')
    plt.ylabel('Jumlah Teks')
    plt.xticks([0, 1], ['Bukan Hate Speech (0)', 'Hate Speech (1)'])
    plt.show() # Tampilkan plot

    # 2. Word Cloud dari Ujaran Kebencian 
    print("\nMenampilkan visualisasi 'Word Cloud Ujaran Kebencian'...")
    # Gabungkan semua teks bersih yang merupakan 'hate speech' (label == 1)
    teks_hate = " ".join(df_bersih[df_bersih['label'] == 1]['teks_bersih'])

    if teks_hate.strip(): # Pastikan ada teks untuk dibuat wordcloud
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              colormap='Reds', max_words=200).generate(teks_hate)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud untuk Teks "Hate Speech" (Label 1)')
        plt.show() # Tampilkan plot
    else:
        print("Tidak ada teks 'hate speech' yang ditemukan untuk membuat Word Cloud.")

else:
    print("Langkah 6 dilewati karena data tidak ter-load.")

print("-" * 50)
print("Semua langkah pengerjaan telah selesai dieksekusi.")
<img width="895" height="462" alt="image" src="https://github.com/user-attachments/assets/53b88455-a6da-49fd-aa60-363a8d2ab250" />
