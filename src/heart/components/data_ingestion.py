import os
import sys
import pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from src.heart.logger import logger
from src.heart.exception import HeartException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("Artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("Artifacts", "train_data.csv")
    test_data_path: str = os.path.join("Artifacts", "test_data.csv")
    source_data_path: str = os.path.join("Notebook_Experiments", "Data", "heart.csv")
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Data ingestion started")
        try:
            df = pd.read_csv(self.config.source_data_path)
            logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            # Basic validation
            required_cols = [
                "age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "condition"
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info("Raw data saved")

            train_df, test_df = train_test_split(
                df, test_size=self.config.test_size, random_state=self.config.random_state, stratify=df["condition"]
            )
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logger.info(f"Train/test split saved — train: {len(train_df)}, test: {len(test_df)}")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise HeartException(e, sys)
