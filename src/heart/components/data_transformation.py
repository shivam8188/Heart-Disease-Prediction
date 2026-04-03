import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from src.heart.logger import logger
from src.heart.exception import HeartException
from src.heart.utils.utils import save_object

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]
TARGET_COLUMN = "condition"


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("Artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = None):
        self.config = config or DataTransformationConfig()

    def build_preprocessor(self) -> ColumnTransformer:
        """Build and return the sklearn preprocessing pipeline."""
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, FEATURE_COLUMNS)
        ])
        return preprocessor

    def initialize_data_transformation(self, train_path: str, test_path: str):
        logger.info("Data transformation started")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df[FEATURE_COLUMNS]
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df[FEATURE_COLUMNS]
            y_test = test_df[TARGET_COLUMN]

            preprocessor = self.build_preprocessor()
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_arr, y_train.values]
            test_arr = np.c_[X_test_arr, y_test.values]

            save_object(self.config.preprocessor_path, preprocessor)
            logger.info("Data transformation complete")
            return train_arr, test_arr

        except Exception as e:
            raise HeartException(e, sys)
