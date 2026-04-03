import os
import sys
from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from src.heart.logger import logger
from src.heart.exception import HeartException
from src.heart.utils.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("Artifacts", "model.pkl")
    model_report_path: str = os.path.join("Artifacts", "model_report.json")
    min_accuracy_threshold: float = 0.75


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8),
    "XGBoost": XGBClassifier(
        learning_rate=0.05, n_estimators=100, max_depth=6,
        gamma=0.1, subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", C=2, probability=True, random_state=42),
}


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config or ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        logger.info("Model training started")
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            report = evaluate_models(X_train, y_train, X_test, y_test, MODELS)

            # Rank by F1 score for a better clinical metric
            best_name = max(report, key=lambda k: report[k]["f1"])
            best_metrics = report[best_name]
            best_model = MODELS[best_name]

            logger.info(f"Best model: {best_name} | {best_metrics}")

            if best_metrics["accuracy"] < self.config.min_accuracy_threshold:
                raise ValueError(
                    f"Best model accuracy {best_metrics['accuracy']} below threshold "
                    f"{self.config.min_accuracy_threshold}"
                )

            # Re-fit on full training data
            best_model.fit(X_train, y_train)
            save_object(self.config.model_path, best_model)

            # Save full report as JSON
            import json, os
            os.makedirs(os.path.dirname(self.config.model_report_path), exist_ok=True)
            with open(self.config.model_report_path, "w") as f:
                json.dump({"best_model": best_name, "all_models": report}, f, indent=2)

            logger.info("Model training complete")
            return best_name, best_metrics, report

        except Exception as e:
            raise HeartException(e, sys)
