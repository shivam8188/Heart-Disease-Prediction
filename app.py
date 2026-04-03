import os
import json
from flask import Flask, request, render_template, jsonify
from src.heart.pipeline.prediction_pipeline import PredictPipeline, PredictionInput
from src.heart.logger import logger

app = Flask(__name__)
predict_pipeline = PredictPipeline()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form
        input_data = PredictionInput(
            age=float(form["age"]),
            sex=int(form["sex"]),
            cp=int(form["cp"]),
            trestbps=float(form["trestbps"]),
            chol=float(form["chol"]),
            fbs=int(form["fbs"]),
            restecg=int(form["restecg"]),
            thalach=float(form["thalach"]),
            exang=int(form["exang"]),
            oldpeak=float(form["oldpeak"]),
            slope=int(form["slope"]),
            ca=int(form["ca"]),
            thal=int(form["thal"]),
        )
        result = predict_pipeline.predict(input_data)
        return render_template("result.html", result=result, form_data=form)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("error.html", error_message=str(e))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for predictions."""
    try:
        data = request.get_json(force=True)
        input_data = PredictionInput(**data)
        result = predict_pipeline.predict(input_data)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/model-info", methods=["GET"])
def model_info():
    """Returns training report if available."""
    report_path = os.path.join("Artifacts", "model_report.json")
    eval_path = os.path.join("Artifacts", "eval_report.json")
    data = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            data["training"] = json.load(f)
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            data["evaluation"] = json.load(f)
    return jsonify(data)


if __name__ == "__main__":
    print("Your app is running at http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)
