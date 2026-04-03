import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.heart.logger import logger
from src.heart.exception import HeartException
from src.heart.utils.utils import load_object

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]


@dataclass
class PredictionInput:
    age: float
    sex: int           # 0=Female, 1=Male
    cp: int            # 0-3 chest pain type
    trestbps: float    # resting blood pressure
    chol: float        # serum cholesterol
    fbs: int           # fasting blood sugar >120 mg/dl
    restecg: int       # resting ECG results
    thalach: float     # max heart rate achieved
    exang: int         # exercise induced angina
    oldpeak: float     # ST depression induced by exercise
    slope: int         # slope of peak exercise ST segment
    ca: int            # number of major vessels
    thal: int          # thalassemia type

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "age": self.age,
            "sex": self.sex,
            "cp": self.cp,
            "trestbps": self.trestbps,
            "chol": self.chol,
            "fbs": self.fbs,
            "restecg": self.restecg,
            "thalach": self.thalach,
            "exang": self.exang,
            "oldpeak": self.oldpeak,
            "slope": self.slope,
            "ca": self.ca,
            "thal": self.thal,
        }])[FEATURE_COLUMNS]


class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("Artifacts", "model.pkl")
        self._preprocessor = None
        self._model = None

    def _load_artifacts(self):
        if self._preprocessor is None:
            self._preprocessor = load_object(self.preprocessor_path)
        if self._model is None:
            self._model = load_object(self.model_path)

    def predict(self, input_data: PredictionInput) -> dict:
        """
        Returns a dict with:
          - prediction: 0 or 1
          - probability: float (confidence %)
          - label: human-readable result
          - risk_level: Low / Medium / High
        """
        try:
            self._load_artifacts()
            df = input_data.to_dataframe()
            scaled = self._preprocessor.transform(df)
            pred = int(self._model.predict(scaled)[0])

            prob = None
            if hasattr(self._model, "predict_proba"):
                prob = round(float(self._model.predict_proba(scaled)[0][pred]) * 100, 1)

            if pred == 1:
                label = "Heart Disease Detected"
                risk_level = "High" if (prob or 80) >= 75 else "Medium"
            else:
                label = "No Heart Disease Detected"
                risk_level = "Low"

            logger.info(f"Prediction: {pred}, Probability: {prob}%, Risk: {risk_level}")
            return {
                "prediction": pred,
                "probability": prob,
                "label": label,
                "risk_level": risk_level,
            }

        except Exception as e:
            raise HeartException(e, sys)
