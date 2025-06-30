from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from sklearn.naive_bayes import GaussianNB
import joblib

app = Flask(__name__)

try:
    model = joblib.load("C:/Users/anupa/Downloads/model.pkl")
except Exception as e:
    print("Error loading the model:", e)
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Error: Model not found. Please make sure the model is properly trained and saved."
    
    # Get the form data
    age = float(request.form["age"])
    sex = float(request.form["sex"])
    cp = float(request.form["cp"])
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    fbs = float(request.form["fbs"])
    restecg = float(request.form["restecg"])
    thalach = float(request.form["thalach"])
    exang = float(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = float(request.form["slope"])
    ca = float(request.form["ca"])
    thal = float(request.form["thal"])
    
    # Create numpy array for prediction
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Get model prediction
    prediction = model.predict(input_data)
    
    # Format prediction result
    result = "Heart disease is present" if prediction == 1 else "No heart disease"

    # Redirect to result page with the prediction
    return redirect(url_for('result', prediction=result))

@app.route("/result")
def result():
    # Render the result template with the prediction
    return render_template("result.html", prediction=request.args.get('prediction'))

if __name__ == "__main__":
    app.run(debug=True)
