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

# Dummy model function
def dummy_predict(features):
    # Contoh: jika rata-rata nilai input lebih besar dari 4, prediksi "diabetes"
    if np.mean(features) > 4:
        return "Positive (Diabetes Detected)"
    else:
        return "Negative (No Diabetes)"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Get each field by its correct name
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
