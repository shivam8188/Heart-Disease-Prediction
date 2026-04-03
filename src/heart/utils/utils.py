import os
import sys
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.heart.logger import logger
from src.heart.exception import HeartException


def save_object(file_path: str, obj) -> None:
    """Serialize and save a Python object to disk."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to: {file_path}")
    except Exception as e:
        raise HeartException(e, sys)


def load_object(file_path: str):
    """Load a serialized Python object from disk."""
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from: {file_path}")
        return obj
    except Exception as e:
        raise HeartException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict) -> dict:
    """
    Train and evaluate multiple classifiers.
    Returns a dict of model_name -> metrics dict.
    """
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report[name] = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            }
            logger.info(f"{name}: {report[name]}")
        return report
    except Exception as e:
        raise HeartException(e, sys)
