"""
This is version is built by me.

"""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

def load_data(path="data/processed/training_data.csv"):
    df = pd.read_csv(path)
    return df

def prepare_data(df):
    feature_columns = [col for col in df.columns if col not in ["date", "home_team", "away_team", "home_goals", "away_goals", "result", "home_elo_after", "away_elo_after", "home_xg", "away_xg"]]

    X = df[feature_columns]
    y = df["result"]

    return X, y, feature_columns

def split_data(X, y, train_size=0.8):
    split = int(len(X) * train_size)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, feature_columns):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}\n")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print(importance.to_string(index=False))

    return predictions, probabilities

def save_model(model, name):
    joblib.dump(model, f"models/model_{name}.pkl")
    print("\nModel saved successfully!")