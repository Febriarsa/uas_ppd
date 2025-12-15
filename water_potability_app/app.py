from flask import Flask, request, jsonify
import sys
import os

# Tambahkan direktori saat ini ke PATH agar dapat mengimpor predict_water_potability
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_water_potability import predict_potability

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Validasi input data
    expected_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    if not all(col in data for col in expected_columns):
        missing_cols = [col for col in expected_columns if col not in data]
        return jsonify({"error": f"Missing expected data fields: {', '.join(missing_cols)}"}), 400

    try:
        prediction = predict_potability(data)
        return jsonify({"potability": prediction}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Menjalankan Flask di port 5000
    # Host 0.0.0.0 agar dapat diakses dari luar container/lingkungan lokal
    app.run(host='0.0.0.0', port=5000, debug=True)
