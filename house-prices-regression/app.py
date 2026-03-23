from flask import Flask, request, jsonify
import joblib

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat mode yang telah disimpan
joblib_model = joblib.load('gbr_model.joblib') 

@app.route('/predict', methods=['POST'])
def predict():
  data = request.json['data']
  prediction = joblib_model.predict(data)
  return jsonify({
    'prediction': prediction.tolist()
  })