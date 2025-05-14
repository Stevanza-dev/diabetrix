from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'diabetes_ensemble_model.joblib'
SCALER_PATH = 'scaler.joblib'

# --- Load Model and Scaler ---
print("Loading trained model and scaler...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model or scaler file not found. Please run 'train_model.py' first.")
    model = None
    scaler = None
except Exception as e:
    print(f"An error occurred while loading the model or scaler: {e}")
    model = None
    scaler = None

# Define the expected feature order (must match training)
FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# --- Dummy Predict Function from appold.py ---
def dummy_predict(features):
    if np.mean(features) > 4:
        return "Positive (Diabetes Detected)"
    else:
        return "Negative (No Diabetes)"

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    prediction = ""
    if request.method == 'POST':
        try:
            required_fields = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'pedigree', 'age']
            
            # Periksa apakah semua kolom ada dan tidak kosong
            for field in required_fields:
                if field not in request.form or not request.form[field].strip():
                    prediction = f"Kolom kosong atau tidak ada: {field}"
                    return render_template('detect.html', prediction=prediction)
            
            # Ambil data dari form
            pregnancies = request.form['pregnancies']
            glucose = request.form['glucose']
            blood_pressure = request.form['blood_pressure']
            skin_thickness = request.form['skin_thickness']
            insulin = request.form['insulin']
            bmi = request.form['bmi']
            pedigree = request.form['pedigree']
            age = request.form['age']

            def safe_float(value, field_name):
                try:
                    # Ganti koma dengan titik untuk mendukung format desimal lokal
                    value = value.replace(',', '.')
                    return float(value)
                except ValueError:
                    raise ValueError(f"Input tidak valid untuk {field_name}. Harap masukkan nilai numerik.")
            
            # Konversi setiap kolom ke float dengan validasi
            pregnancies = safe_float(pregnancies, 'Jumlah Kehamilan')
            glucose = safe_float(glucose, 'Glukosa')
            blood_pressure = safe_float(blood_pressure, 'Tekanan Darah')
            skin_thickness = safe_float(skin_thickness, 'Ketebalan Kulit')
            insulin = safe_float(insulin, 'Insulin')
            bmi = safe_float(bmi, 'BMI')
            pedigree = safe_float(pedigree, 'Fungsi Pedigree Diabetes')
            age = safe_float(age, 'Usia')

            # Create a list of input features in the correct order
            input_features_list = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]

            if model is not None and scaler is not None:
                # Use the machine learning model
                # Create a DataFrame with correct feature names
                input_df = pd.DataFrame([input_features_list], columns=FEATURE_NAMES)
                # Scale the features
                scaled_features = scaler.transform(input_df)
                # Make prediction
                prediction = model.predict(scaled_features)
                probability = model.predict_proba(scaled_features)
                # Interpret prediction
                if prediction[0] == 1:
                    result = "Positive (Diabetes Detected)"
                else:
                    result = "Negative (No Diabetes)"
                prediction = f"Prediction using ML model: {result}. Confidence for Diabetes: {probability[0][1]*100:.2f}%. Confidence for No Diabetes: {probability[0][0]*100:.2f}%."
            else:
                # Use the dummy predict function
                result = dummy_predict(input_features_list)
                prediction = f"Prediction using dummy model: {result} (Note: ML model failed to load, using backup method.)"

        except ValueError as e:
            prediction = str(e)  # Tampilkan pesan kesalahan spesifik
        except Exception as e:
            prediction = f"Terjadi kesalahan saat prediksi: {str(e)}"
            print(f"Kesalahan saat prediksi: {e}")  # Log the error for debugging

        return render_template('detect.html', prediction = prediction)

    return render_template('detect.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)
