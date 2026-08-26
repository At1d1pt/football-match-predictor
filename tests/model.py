'''
Test the model for every match in the past.
Not really useful since it has already seen the data during training.
'''

import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from models.train import evaluate_model

def test_model():
    model = joblib.load("models/model_xgboost.pkl")
    df = pd.read_csv("data/processed/training_data.csv")

    feature_columns = [col for col in df.columns if col not in ["date", "home_team", "away_team", "home_goals", "away_goals", "result", "home_elo_after", "away_elo_after", "home_xg", "away_xg"]]

    X = df[feature_columns]
    y = df["result"].map({"H": 0, "D": 1, "A": 2})

    evaluate_model(model, X, y, feature_columns)

if __name__ == "__main__":
    test_model()
    