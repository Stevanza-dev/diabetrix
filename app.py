# from flask import Flask, render_template, request
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load model (pastikan kamu sudah punya model.pkl)
# model = joblib.load("model.pkl")

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     if request.method == "POST":
#         # Ambil 9 fitur dari form
#         features = [float(request.form[f"feature{i}"]) for i in range(1, 10)]
#         result = model.predict([features])
#         prediction = "Diabetes Detected" if result[0] == 1 else "No Diabetes"

#     return render_template("index.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Dummy model function (untuk detect.html)
def dummy_predict(features):
    if np.mean(features) > 4:
        return "Positive (Diabetes Detected)"
    else:
        return "Negative (No Diabetes)"

# Route untuk Home (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk About (about.html)
@app.route('/about')
def about():
    return render_template('about.html')

# Route untuk Diabetes Detection (detect.html)
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    result = None
    if request.method == 'POST':
        input_data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['pedigree']),
            float(request.form['age'])
        ]
        result = dummy_predict(input_data)
    return render_template('detect.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)