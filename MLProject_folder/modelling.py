import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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
        dagshub.init(repo_owner='pyogaaa', repo_name='Eksperimen_SML', mlflow=True)
    
    # Menyesuaikan nama eksperimen agar relevan di dashboard
    mlflow.set_experiment("Eksperimen_Mobile_Legends_Sentiment")

    # 2. AKTIFKAN AUTOLOG SCIKIT-LEARN DENGAN REGISTRASI MODEL
    # log_models=True memastikan folder "model" beserta MLmodel, conda.yaml, dll. diunggah otomatis
    mlflow.sklearn.autolog(log_models=True, registered_model_name="Mobile_Legends_SVM_Model")

    # Path dinamis untuk file dan folder output fisik lokal
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'output', 'modelling')
    os.makedirs(output_dir, exist_ok=True) # Memastikan folder output/modelling ada

    data_path = os.path.join(base_dir, 'dataset_mobile_legends_preprocessed.csv')
    
    # Fallback check struktur folder kriteria 2
    if not os.path.exists(data_path):
        data_path = os.path.join('Membangun_model', 'namadataset_preprocessing', 'dataset_mobile_legends_preprocessed.csv')

    if not os.path.exists(data_path):
        print(f"[!] File {data_path} tidak ditemukan!")
        return

    print(f"[*] Memuat data hasil preprocessing dari: {data_path}")
    df = pd.read_csv(data_path)
    
    # MENANGANI NILAI KOSONG (Wajib ada agar TF-IDF tidak error saat CI berjalan)
    df['content_clean'] = df['content_clean'].fillna('missing')

    # Pemisahan Fitur dan Target (Murni Teks dan Label)
    X = df['content_clean'] 
    y = df['sentiment_label'].values

    # Split data train dan test dengan stratifikasi data timpang
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vektorisasi Teks (TF-IDF) murni
    print("[*] Mengekstrak fitur teks menggunakan TF-IDF...")
    tfidf = TfidfVectorizer(max_features=2000)
    X_train = tfidf.fit_transform(X_train_raw).toarray()
    X_test = tfidf.transform(X_test_raw).toarray()

    # 3. Training Model dalam MLflow Run Block
    with mlflow.start_run(run_name="SVM_Pure_Text_Sentiment"):
        print("[*] Memulai Training Support Vector Classifier (SVM)...")
        
        # Konfigurasi model SVM dengan penyeimbang bobot kelas otomatis
        model = SVC(
            kernel='rbf',
            C=1.0,
            class_weight='balanced', 
            random_state=42
        )
        
        # Proses training model (Otomatis direkam oleh mlflow)
        model.fit(X_train, y_train)

        # Prediksi hasil evaluasi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # --- PENYIMPANAN FISIK DI SUB-FOLDER OUTPUT LOKAL ---
        model_save_path = os.path.join(output_dir, 'model_base.pkl')
        tfidf_save_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
        
        joblib.dump(model, model_save_path)
        joblib.dump(tfidf, tfidf_save_path)

        # Mengunggah pkl tambahan ke root artifact MLflow DagsHub
        mlflow.log_artifact(model_save_path)
        mlflow.log_artifact(tfidf_save_path)

        print("\n--- BASE MODEL LOGGED TO LOCAL & DAGSHUB VIA AUTOLOG ---")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"File output tersimpan di: {output_dir}\n")
        print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()