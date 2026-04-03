"""
train.py
--------
Run this script to execute the full training pipeline.

Usage:
    python train.py

Make sure heart.csv is placed at:
    Notebook_Experiments/Data/heart.csv
"""

from src.heart.pipeline.training_pipeline import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline()
