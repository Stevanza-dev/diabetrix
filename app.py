from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Setup database
def init_db():
    conn = sqlite3.connect('diabetes_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_ip TEXT,
                  pregnancies INTEGER,
                  glucose INTEGER,
                  blood_pressure INTEGER,
                  skin_thickness INTEGER,
                  insulin INTEGER,
                  bmi REAL,
                  pedigree REAL,
                  age INTEGER,
                  prediction TEXT,
                  check_date TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# --- Configuration ---
MODEL_PATH = 'model/diabetes_ensemble_model.joblib'
SCALER_PATH = 'model/scaler.joblib'

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
FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# --- Dummy Predict Function ---
def dummy_predict(features):
    if np.mean(features) > 4:
        return "Positive (Diabetes Detected)"
    else:
        return "Negative (No Diabetes)"
    
# --- Custom Filter ---
@app.template_filter('contains')
def contains_filter(s, substr):
    return substr in s if s else False

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def show_history():
    date_filter = request.args.get('date')
    conn = sqlite3.connect('diabetes_history.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if date_filter:
        # Filter berdasarkan tanggal (format: YYYY-MM-DD)
        c.execute('''SELECT * FROM history 
                     WHERE user_ip = ? AND DATE(check_date) = ?
                     ORDER BY check_date DESC''',
                  (request.remote_addr, date_filter))
    else:
        c.execute('''SELECT * FROM history 
                     WHERE user_ip = ? 
                     ORDER BY check_date DESC''',
                  (request.remote_addr,))
    rows = c.fetchall()
    history_data = []
    for row in rows:
        row_dict = dict(row)
        # Ubah check_date ke format tanggal yang diinginkan
        if row_dict['check_date']:
            try:
                dt = datetime.fromisoformat(row_dict['check_date'])
                row_dict['check_date'] = dt.strftime('%d-%m-%Y %H:%M')
            except Exception:
                pass  # biarkan string aslinya jika gagal
        history_data.append(row_dict)
    conn.close()
    
    # Hitung di backend
    high_risk = sum(1 for r in history_data if 'Positive' in r['prediction'])
    low_risk = sum(1 for r in history_data if 'Negative' in r['prediction'])
    
    return render_template('history.html',
                         history=history_data,
                         high_risk=high_risk,
                         low_risk=low_risk)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    prediction = ""
    # Dictionary untuk menyimpan semua nilai form
    form_data = {
        'pregnancies': '',
        'glucose': '',
        'blood_pressure': '',
        'skin_thickness': '',
        'insulin': '',
        'bmi': '',
        'pedigree': '',
        'age': ''
    }

    if request.method == 'POST':
        try:
            # Simpan semua nilai form yang di-submit
            for field in form_data.keys():
                form_data[field] = request.form.get(field, '')

            # Validasi: Pastikan semua field terisi
            for field, value in form_data.items():
                if not value.strip():
                    prediction = f"Error: Field {field.replace('_', ' ')} cannot be empty"
                    return render_template('detect.html', 
                                         prediction=prediction,
                                         form_data=form_data)

            # Konversi ke float dengan handling error
            def safe_float(value, field_name):
                try:
                    return float(value.replace(',', '.'))
                except ValueError:
                    raise ValueError(f"Invalid input for {field_name}. Please enter a number.")

            input_features = [
                safe_float(form_data['pregnancies'], 'Pregnancies'),
                safe_float(form_data['glucose'], 'Glucose'),
                safe_float(form_data['blood_pressure'], 'Blood Pressure'),
                safe_float(form_data['skin_thickness'], 'Skin Thickness'),
                safe_float(form_data['insulin'], 'Insulin'),
                safe_float(form_data['bmi'], 'BMI'),
                safe_float(form_data['pedigree'], 'Diabetes Pedigree'),
                safe_float(form_data['age'], 'Age')
            ]

            # Lakukan prediksi
            if model and scaler:
                input_df = pd.DataFrame([input_features], columns=FEATURE_NAMES)
                scaled_features = scaler.transform(input_df)
                pred = model.predict(scaled_features)[0]
                proba = model.predict_proba(scaled_features)[0]
                
                if pred == 1:
                    prediction = f"Positive (Diabetes Detected) - Confidence: {proba[1]*100:.1f}%"
                else:
                    prediction = f"Negative (No Diabetes) - Confidence: {proba[0]*100:.1f}%"
            else:
                prediction = dummy_predict(input_features) + " (Using dummy model)"

        except ValueError as e:
            prediction = str(e)
        except Exception as e:
            prediction = f"Prediction error: {str(e)}"
            print(f"Error during prediction: {e}")

        # Simpan ke database
        conn = sqlite3.connect('diabetes_history.db')
        c = conn.cursor()
        c.execute('''INSERT INTO history 
                     (user_ip, pregnancies, glucose, blood_pressure, 
                      skin_thickness, insulin, bmi, pedigree, age, 
                      prediction, check_date)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                     (request.remote_addr,
                      form_data['pregnancies'],
                      form_data['glucose'],
                      form_data['blood_pressure'],
                      form_data['skin_thickness'],
                      form_data['insulin'],
                      form_data['bmi'],
                      form_data['pedigree'],
                      form_data['age'],
                      prediction,
                      datetime.now()))
        conn.commit()
        conn.close()

        return render_template('detect.html', 
                             prediction=prediction,
                             form_data=form_data)

    return render_template('detect.html', 
                         prediction=prediction,
                         form_data=form_data)

@app.route('/delete-history/<int:id>', methods=['DELETE'])
def delete_history(id):
    conn = sqlite3.connect('diabetes_history.db')
    c = conn.cursor()
    c.execute('DELETE FROM history WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)