import joblib
import pandas as pd
import numpy as np

# Muat model dan scaler yang telah disimpan
best_rf_model = joblib.load('best_rf_model.joblib')
scaler = joblib.load('scaler.joblib')

def predict_potability(input_data):
    """
    Memprediksi potabilitas air berdasarkan input data.

    Args:
        input_data (dict atau list of dict):
            Data masukan yang berisi fitur-fitur air.
            Contoh: {'ph': 7.0, 'Hardness': 180.0, 'Solids': 20000.0, 'Chloramines': 6.0,
                     'Sulfate': 300.0, 'Conductivity': 400.0, 'Organic_carbon': 12.0,
                     'Trihalomethanes': 70.0, 'Turbidity': 3.0}

    Returns:
        int: Hasil prediksi potabilitas air (0 = tidak layak minum, 1 = layak minum).
    """
    # Konversi input data menjadi DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_df = pd.DataFrame(input_data)
    else:
        raise ValueError("Input data harus berupa dictionary atau list of dictionaries.")

    # Pastikan urutan kolom sesuai dengan data pelatihan
    # (ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
    expected_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    if not all(col in input_df.columns for col in expected_columns):
        missing_cols = [col for col in expected_columns if col not in input_df.columns]
        raise ValueError(f"Input data harus mengandung semua kolom yang diharapkan. Kolom hilang: {missing_cols}")
    input_df = input_df[expected_columns]

    # Lakukan scaling pada input data
    input_scaled = scaler.transform(input_df)

    # Lakukan prediksi menggunakan model
    prediction = best_rf_model.predict(input_scaled)

    return int(prediction[0]) # Convert numpy.int64 to int for JSON serialization

# This part is for local testing and won't be used by the Flask app directly
if __name__ == "__main__":
    print("This script defines the predict_potability function.")
    print("It can be imported and used by other applications (like a Flask app).")
