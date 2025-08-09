import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json
import time

print("Mulai proses training dan tuning dengan hyperparameter kompleks dan CV=5...")

# Path yang sudah disesuaikan
data_path = "data/restaurant_profit.csv"
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# 1. Memuat dan Mempersiapkan Data
df = pd.read_csv(data_path, delimiter=";")

# Menyimpan daftar unik MenuItem untuk form input di web
menu_items_list = sorted(df["MenuItem"].unique().tolist())
with open(os.path.join(model_dir, "menu_items.json"), "w") as f:
    json.dump(menu_items_list, f)
print("Daftar unik 'MenuItem' telah disimpan ke model/menu_items.json")

# Menggunakan semua kolom kecuali Profitability
X = df.drop("Profitability", axis=1)
y = df["Profitability"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Definisikan preprocessor untuk setiap jenis kolom
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Price"]),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["RestaurantID", "MenuCategory", "MenuItem"],
        ),
        ("txt", TfidfVectorizer(ngram_range=(1, 2)), "Ingredients"),
    ],
    remainder="drop",
)

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print("Data berhasil dimuat dan dipersiapkan dengan semua fitur.")


# 2. Mendefinisikan Model dan Hyperparameter Grid yang Lebih Kompleks
models_and_params = [
    {
        "name": "Logistic Regression",
        # --- PERBAIKAN: max_iter dinaikkan untuk mencegah ConvergenceWarning ---
        "model": LogisticRegression(max_iter=5000, random_state=42),
        "params": {
            "classifier__penalty": ["l1", "l2"],
            "classifier__C": [0.1, 1.0, 10.0, 50.0],
            "classifier__solver": ["liblinear", "saga"],
        },
    },
    {
        "name": "Random Forest",
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
        },
    },
    {
        "name": "Gradient Boosting",
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [3, 5, 8],
            "classifier__subsample": [0.8, 0.9, 1.0],
            "classifier__max_features": ["sqrt", "log2"],
        },
    },
]

# 3. Melakukan Tuning dan Mencari Model Terbaik
best_model_estimator = None
best_model_name = ""
best_accuracy = 0.0
start_time = time.time()

for config in models_and_params:
    model_name = config["name"]
    model = config["model"]
    params = config["params"]

    print(f"\n--- Sedang melakukan tuning untuk: {model_name} ---")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    grid_search = GridSearchCV(
        pipeline, params, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Hyperparameter terbaik untuk {model_name}: {grid_search.best_params_}")
    print(f"Skor akurasi cross-validation terbaik: {grid_search.best_score_:.4f}")

    if grid_search.best_score_ > best_accuracy:
        best_accuracy = grid_search.best_score_
        best_model_estimator = grid_search.best_estimator_
        best_model_name = model_name
        print(f"*** {model_name} menjadi model terbaik yang baru! ***")

total_time = time.time() - start_time
print(f"\n--- ⏳ Proses tuning semua model selesai dalam {total_time:.2f} detik ---")

# 4. Evaluasi Final Model Terbaik di Data Test
print(f"Model terbaik secara keseluruhan adalah: {best_model_name}")
y_pred = best_model_estimator.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(f"\nAkurasi model terbaik di data test: {final_accuracy:.4f}")
print("Laporan Klasifikasi Final:\n", report)

# 5. Menyimpan Model Terbaik dan Label Encoder
print("\nMenyimpan model terbaik hasil seleksi...")
joblib.dump(best_model_estimator, os.path.join(model_dir, "best_model.joblib"))
joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))

print(
    f"Model terbaik '{best_model_name}' dan komponennya berhasil disimpan di dalam folder '{model_dir}'."
)
