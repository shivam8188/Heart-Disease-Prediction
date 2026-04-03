import os
import sys
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from src.heart.logger import logger
from src.heart.exception import HeartException
from src.heart.utils.utils import load_object


class ModelEvaluation:
    def __init__(self):
        self.model_path = os.path.join("Artifacts", "model.pkl")
        self.eval_report_path = os.path.join("Artifacts", "eval_report.json")

    def evaluate(self, test_arr) -> dict:
        """Run full evaluation on the saved model and return metrics."""
        logger.info("Model evaluation started")
        try:
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            model = load_object(self.model_path)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            metrics = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
            }

            os.makedirs(os.path.dirname(self.eval_report_path), exist_ok=True)
            with open(self.eval_report_path, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            raise HeartException(e, sys)
