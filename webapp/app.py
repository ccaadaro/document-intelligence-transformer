from flask import Flask, render_template, request
from prometheus_flask_exporter import PrometheusMetrics

import requests

app = Flask(__name__)
metrics = PrometheusMetrics(app)
metrics.info("app_info", "Document Type Prediction API", version="1.0")


API_URL = "http://api:5000/predict"  # usa 'localhost' si no usas Docker

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        image = request.files["image"]
        text = request.form["text"]

        files = {"image": image}
        data = {"text": text}
        try:
            response = requests.post(API_URL, files=files, data=data)
            if response.status_code == 200:
                prediction = response.json()
            else:
                prediction = {"error": f"API Error {response.status_code}"}
        except Exception as e:
            prediction = {"error": str(e)}

    return render_template("index.html", prediction=prediction)
