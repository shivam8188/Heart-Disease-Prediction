# ❤️ Heart Disease Prediction — v2.0

An end-to-end machine learning web application that predicts heart disease risk based on 13 clinical indicators, built with Flask and scikit-learn.

---

## 🗂️ Project Structure

```
HeartDiseasePrediction/
├── app.py                          # Flask web app entry point
├── train.py                        # Training pipeline entry point
├── setup.py
├── requirements.txt
├── Dockerfile
│
├── src/heart/
│   ├── components/
│   │   ├── data_ingestion.py       # Load & split dataset
│   │   ├── data_transformation.py  # Preprocessing pipeline
│   │   ├── model_trainer.py        # Train & select best model
│   │   └── model_evaluation.py     # Evaluate & report metrics
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates full training
│   │   └── prediction_pipeline.py  # Inference logic
│   ├── utils/utils.py              # Save/load/evaluate helpers
│   ├── logger.py
│   └── exception.py
│
├── templates/
│   ├── index.html                  # Main prediction form
│   ├── result.html                 # Prediction result page
│   └── error.html                  # Error display
│
├── static/
│   ├── css/styles.css
│   └── js/main.js
│
├── Artifacts/                      # Auto-generated: model, preprocessor, reports
├── logs/                           # Auto-generated: log files
└── Notebook_Experiments/
    └── Data/
        └── heart.csv               # ← Place your dataset here
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add dataset
Download the [Cleveland Heart Disease dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) and place it at:
```
Notebook_Experiments/Data/heart.csv
```

### 3. Train the model
```bash
python train.py
```
This will:
- Ingest and split the data
- Preprocess features
- Train 8 classifiers and select the best by F1 score
- Save `Artifacts/model.pkl`, `Artifacts/preprocessor.pkl`
- Write training + evaluation reports to `Artifacts/`

### 4. Run the web app
```bash
python app.py
```
Visit: [http://localhost:8080](http://localhost:8080)

---

## 🐳 Docker

```bash
docker build -t cardiopredict .
docker run -p 8080:8080 cardiopredict
```

---

## 🔌 API

### `POST /api/predict`
**Request body (JSON):**
```json
{
  "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
  "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
  "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}
```
**Response:**
```json
{
  "status": "success",
  "result": {
    "prediction": 1,
    "probability": 87.4,
    "label": "Heart Disease Detected",
    "risk_level": "High"
  }
}
```

### `GET /model-info`
Returns JSON report of all trained models + evaluation metrics.

---

## 📊 Models Evaluated

| Model                  | Tuned |
|------------------------|-------|
| Logistic Regression    | ✓     |
| Naive Bayes            |       |
| Random Forest          | ✓     |
| XGBoost                | ✓     |
| Gradient Boosting      | ✓     |
| K-Nearest Neighbors    | ✓     |
| Decision Tree          | ✓     |
| Support Vector Machine | ✓     |

Best model is selected automatically by **F1 score**.

---

## ⚠️ Disclaimer
This tool is for **educational purposes only**. It is not a substitute for professional medical diagnosis or advice.
