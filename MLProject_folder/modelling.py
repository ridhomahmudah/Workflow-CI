import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline  # <-- TAMBAHKAN INI
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import dagshub
import joblib

def main():
    # 1. INTEGRASI DAGSHUB & MLFLOW (Mendukung CI/CD GitHub Actions & Lokal)
    if os.getenv('GITHUB_ACTIONS'):
        # Membaca dari environment variable di file YAML GitHub Actions
        remote_url = os.getenv('MLFLOW_TRACKING_URI')
        mlflow.set_tracking_uri(remote_url)
    else:
        # Tetap bisa jalan normal kalau running lokal di komputer
        print("[*] Menghubungkan ke DagsHub Tracker...")
        dagshub.init(repo_owner='ridhomahmudah', repo_name='Eksperimen_SML_Ridho-nur-mahmudah', mlflow=True)

    # 2. AKTIFKAN AUTOLOG SCIKIT-LEARN DENGAN REGISTRASI MODEL
    # Karena kita pakai Pipeline, Autolog akan merekam seluruh tahapan Pipeline tersebut
    mlflow.sklearn.autolog(log_models=True, registered_model_name="Mobile_Legends_SVM_Model")

    # Path dinamis untuk file dan folder output fisik lokal
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'output', 'modelling')
    os.makedirs(output_dir, exist_ok=True) # Memastikan folder output/modelling ada

    data_path = os.path.join(base_dir, 'dataset_mobile_legends_preprocessed.csv')

    print(f"[*] Memuat data hasil preprocessing dari: {data_path}")
    df = pd.read_csv(data_path)
    
    # MENANGANI NILAI KOSONG (Wajib ada agar TF-IDF tidak error saat CI berjalan)
    df['content_clean'] = df['content_clean'].fillna('missing')

    # Pemisahan Fitur dan Target (Murni Teks dan Label)
    X = df['content_clean'] 
    y = df['sentiment_label'].values

    # Split data train dan test dengan stratifikasi data timpang
    # KITA TETAP GUNAKAN DATA TEKS MENTAH (_raw) KARENA VEKTORISASI DIKUNCI DI DALAM PIPELINE
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Training Model dalam MLflow Run Block Menggunakan Pipeline
    with mlflow.start_run(run_name="SVM_Pipeline_Text_Sentiment"):
        print("[*] Membuat Bundling Pipeline (TF-IDF + SVM)...")
        
        # Satukan TF-IDF dan SVM ke dalam satu kesatuan alur kerja
        sentiment_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2000)),
            ('svm', SVC(
                kernel='rbf',
                C=1.0,
                class_weight='balanced', 
                random_state=42
            ))
        ])
        
        print("[*] Memulai Training Pipeline langsung dari teks mentah...")
        # .fit() di pipeline otomatis menjalankan fit_transform TF-IDF lalu ditraining ke SVM
        sentiment_pipeline.fit(X_train_raw, y_train)

        # Prediksi hasil evaluasi menggunakan data teks mentah langsung
        preds = sentiment_pipeline.predict(X_test_raw)
        acc = accuracy_score(y_test, preds)
        
        # --- PENYIMPANAN FISIK DI SUB-FOLDER OUTPUT LOKAL ---
        # Sekarang cukup simpan SATU file tunggal bernama model_base.pkl
        model_save_path = os.path.join(output_dir, 'model_base.pkl')
        
        # Menyimpan objek pipeline utuh (TF-IDF + SVM ada di dalam sini)
        joblib.dump(sentiment_pipeline, model_save_path)

        # Mengunggah pkl terintegrasi ke root artifact MLflow DagsHub
        mlflow.log_artifact(model_save_path)

        print("\n--- PIPELINE MODEL LOGGED TO LOCAL & DAGSHUB VIA AUTOLOG ---")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"File model tunggal (TF-IDF + SVM) tersimpan di: {model_save_path}\n")
        print(classification_report(y_test, preds))

if __name__ == "__main__":
        main()