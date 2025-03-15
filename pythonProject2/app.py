from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_path = r"V:\pythonProject2\Final_model.pkl"
scaler_path = r"V:\pythonProject2\Final_scaler.pkl"

try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or Scaler failed to load"}), 500

    try:
        data = request.form if request.form else request.json
        required_fields = ["step", "amount", "oldbalanceOrg", "newbalanceOrg", "oldbalanceDest", "newbalanceDest"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        input_data = np.array([[float(data[field]) for field in required_fields]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        result = "Fraudulent Transaction" if prediction == 1 else "Non-Fraudulent Transaction"

        return render_template('result.html', prediction=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
