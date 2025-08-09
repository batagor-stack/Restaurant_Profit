# app.py

from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
import json

app = Flask(__name__)

# --- PATH DISESUAIKAN ---
model_path = os.path.join("model", "best_model.joblib")
label_encoder_path = os.path.join("model", "label_encoder.joblib")
data_path = os.path.join("data", "restaurant_profit.csv")  # Path ke data mentah

# Memuat semua komponen yang dibutuhkan
print("Memuat model dan komponen...")
try:
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    df = pd.read_csv(data_path, delimiter=";")

    # Membuat pemetaan 3 tingkat (Restoran -> Kategori -> Item)
    restaurant_menu_mapping = {}
    for restaurant_id in sorted(df["RestaurantID"].unique()):
        restaurant_df = df[df["RestaurantID"] == restaurant_id]
        category_mapping = (
            restaurant_df.groupby("MenuCategory")["MenuItem"]
            .apply(lambda x: sorted(list(set(x))))
            .to_dict()
        )
        restaurant_menu_mapping[restaurant_id] = category_mapping

    # Menambahkan pemetaan Item Menu -> Bahan
    menu_ingredients_mapping = (
        df.drop_duplicates(subset="MenuItem")
        .set_index("MenuItem")["Ingredients"]
        .to_dict()
    )

    print("Model dan semua pemetaan menu berhasil dimuat.")
except FileNotFoundError as e:
    print(
        f"Error: File tidak ditemukan - {e}. Pastikan semua file ada di direktori yang benar."
    )
    exit()


@app.route("/", methods=["GET", "POST"])
def index():
    # --- PERUBAHAN: Menambahkan variabel baru untuk hasil analisis ---
    prediction_text = ""
    prediction_color = "text-gray-800"
    analysis_text = ""
    recommendation_text = ""

    if request.method == "POST":
        try:
            data = {
                "RestaurantID": request.form["restaurant_id"],
                "MenuCategory": request.form["menu_category"],
                "MenuItem": request.form["menu_item"],
                "Ingredients": request.form["ingredients"],
                "Price": float(request.form["price"]),
            }

            input_df = pd.DataFrame([data])
            prediction_encoded = model.predict(input_df)
            prediction = label_encoder.inverse_transform(prediction_encoded)[0]

            prediction_text = f"Prediksi Profitabilitas: {prediction}"

            # --- PERUBAHAN: Logika untuk Analisis & Rekomendasi ---
            if prediction == "High":
                prediction_color = "text-green-500"
                analysis_text = "Menu ini memiliki potensi profitabilitas yang sangat baik. Biaya bahan baku dan harga jual sudah optimal."
                recommendation_text = "Prioritaskan item ini. Tonjolkan dalam promosi atau sebagai 'Chef's Recommendation' untuk meningkatkan volume penjualan."
            elif prediction == "Medium":
                prediction_color = "text-yellow-500"
                analysis_text = "Profitabilitas menu ini berada di tingkat standar. Merupakan tulang punggung menu yang stabil."
                recommendation_text = "Tinjau biaya bahan baku secara berkala untuk mencari peluang efisiensi. Pertimbangkan kenaikan harga minor jika pasar memungkinkan."
            else:  # Low
                prediction_color = "text-red-500"
                analysis_text = "Perhatian! Menu ini memiliki profitabilitas rendah dan berisiko mengurangi keuntungan restoran secara keseluruhan."
                recommendation_text = "Lakukan rekayasa menu (menu engineering): coba ganti bahan yang lebih murah, sesuaikan harga jual secara signifikan, atau pertimbangkan untuk menghapus item ini dari menu."

        except Exception as e:
            prediction_text = f"Terjadi error saat prediksi: {e}"

    return render_template(
        "index.html",
        restaurant_menu_mapping=restaurant_menu_mapping,
        menu_ingredients_mapping=menu_ingredients_mapping,
        prediction_text=prediction_text,
        prediction_color=prediction_color,
        # --- PERUBAHAN: Mengirim teks analisis ke template ---
        analysis_text=analysis_text,
        recommendation_text=recommendation_text,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
