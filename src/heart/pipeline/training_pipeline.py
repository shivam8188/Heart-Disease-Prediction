import sys
from src.heart.logger import logger
from src.heart.exception import HeartException
from src.heart.components.data_ingestion import DataIngestion
from src.heart.components.data_transformation import DataTransformation
from src.heart.components.model_trainer import ModelTrainer
from src.heart.components.model_evaluation import ModelEvaluation


def run_training_pipeline():
    """Execute the full ML training pipeline."""
    logger.info("=" * 60)
    logger.info("Training pipeline started")
    try:
        # 1. Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # 2. Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr = transformation.initialize_data_transformation(train_path, test_path)

        # 3. Model Training
        trainer = ModelTrainer()
        best_model_name, best_metrics, all_reports = trainer.initiate_model_training(train_arr, test_arr)

        # 4. Model Evaluation
        evaluator = ModelEvaluation()
        eval_metrics = evaluator.evaluate(test_arr)

        print("\n" + "=" * 60)
        print(f"  Best Model   : {best_model_name}")
        print(f"  Accuracy     : {eval_metrics['accuracy']:.4f}")
        print(f"  Precision    : {eval_metrics['precision']:.4f}")
        print(f"  Recall       : {eval_metrics['recall']:.4f}")
        print(f"  F1 Score     : {eval_metrics['f1_score']:.4f}")
        if eval_metrics.get("roc_auc"):
            print(f"  ROC-AUC      : {eval_metrics['roc_auc']:.4f}")
        print("=" * 60)

        logger.info("Training pipeline completed successfully")
        return best_model_name, eval_metrics

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise HeartException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
