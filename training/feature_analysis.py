import joblib
import pandas as pd
import os

print("Menganalisis Feature Importance dari model terbaik...")

# Path ke model dan data
model_dir = "../model"
model_path = os.path.join(model_dir, "best_model.joblib")

try:
    # Memuat model yang sudah dilatih
    best_model_pipeline = joblib.load(model_path)
    print(f"Model berhasil dimuat dari {model_path}")
except FileNotFoundError:
    print(
        "Error: File model tidak ditemukan. Jalankan script '1_model_training.py' terlebih dahulu."
    )
    exit()

# Memeriksa apakah model yang dipilih adalah model berbasis pohon (punya feature_importances_)
try:
    # Mengakses langkah klasifikasi di dalam pipeline
    classifier = best_model_pipeline.named_steps["classifier"]
    importances = classifier.feature_importances_
except AttributeError:
    print(
        f"Error: Model terpilih ({type(classifier).__name__}) tidak memiliki atribut 'feature_importances_'."
    )
    print(
        "Metode ini hanya berfungsi untuk model berbasis pohon seperti Random Forest atau Gradient Boosting."
    )
    exit()

# Mengakses langkah pra-pemrosesan untuk mendapatkan nama fitur
preprocessor = best_model_pipeline.named_steps["preprocessor"]

# Mendapatkan nama semua fitur setelah di-transformasi
# Ini akan menghasilkan nama seperti 'num__Price', 'cat__RestaurantID_R001', 'txt__cheese', 'txt__truffle', dll.
feature_names = preprocessor.get_feature_names_out()

# Membuat DataFrame untuk visualisasi yang lebih baik
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

# Mengurutkan DataFrame berdasarkan skor importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Menampilkan 20 fitur teratas
print("\n--- Top 20 Fitur Paling Penting Menurut Model ---")
# Mengingat preferensi Anda untuk tabel yang rapi, saya akan format outputnya
print(importance_df.head(20).to_string())

# Menganalisis hasil
print("\n--- Analisis ---")
# Menghitung berapa banyak fitur 'ingredients' (txt__) yang masuk 20 besar
top_20_features = importance_df.head(20)["Feature"]
ingredient_features_in_top_20 = sum(f.startswith("txt__") for f in top_20_features)

if ingredient_features_in_top_20 > 0:
    print(
        f"✅ Terbukti! Sebanyak {ingredient_features_in_top_20} dari 20 fitur teratas berasal dari kolom 'Ingredients'."
    )
    print(
        "Ini menunjukkan bahwa kata-kata spesifik dalam bahan-bahan sangat memengaruhi prediksi profitabilitas."
    )
else:
    print(
        "⚠️ Anehnya, tidak ada fitur dari 'Ingredients' yang masuk ke 20 besar dalam analisis ini."
    )
