"""

This program predicts expected goals scored by both teams using XGBoost Poisson Regressors,
and evaluates match outcome accuracy using both simple goal difference and Poisson probability matrices.

"""

import sys
import os

from train import load_data, split_data, save_model
from poisson import PoissonModel
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import pandas as pd

def goals_to_result(home, away, threshold=0.25):
    diff = home - away

    if abs(diff) < threshold:
        return "D"
    elif diff > 0:
        return "H"
    else:
        return "A"

def prepare_data(df):
    feature_columns = [col for col in df.columns if col not in ["date", "home_team", "away_team", "result", "home_goals", "away_goals", "home_elo_after", "away_elo_after", "home_xg", "away_xg"]]

    X = df[feature_columns]    
    y_home = df["home_goals"]
    y_away = df["away_goals"]

    return X, y_home, y_away, feature_columns

if __name__ == "__main__":
    df = load_data()
    X, y_home, y_away, feature_columns = prepare_data(df)

    X_train, X_test, y_home_train, y_home_test = split_data(X, y_home)
    _, _, y_away_train, y_away_test = split_data(X, y_away)

    home_model = XGBRegressor(
        objective="count:poisson",
        random_state=42,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=500
    )

    home_model.fit(X_train, y_home_train)

    away_model = XGBRegressor(
        objective="count:poisson",
        random_state=42,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=500
    )

    away_model.fit(X_train, y_away_train)

    home_predictions = home_model.predict(X_test)
    away_predictions = away_model.predict(X_test)

    poisson_model = PoissonModel(max_goals=6)

    threshold_predictions = []
    poisson_predictions = []
    actual_results = []

    for home_lam, away_lam in zip(home_predictions, away_predictions):
        threshold_predictions.append(goals_to_result(home_lam, away_lam))
        poisson_res = poisson_model.predict(home_lam, away_lam)
        poisson_predictions.append(poisson_res["predicted_outcome"])

    for home, away in zip(y_home_test, y_away_test):
        if home > away:
            actual_results.append("H")
        elif away > home:
            actual_results.append("A")
        else:
            actual_results.append("D")

    print("Goal Model MAE Home:", mean_absolute_error(y_home_test, home_predictions))
    print("Goal Model MAE Away:", mean_absolute_error(y_away_test, away_predictions))
    print("Goal Model Accuracy (Threshold method):", accuracy_score(actual_results, threshold_predictions))
    print("Goal Model Accuracy (Poisson matrix method):", accuracy_score(actual_results, poisson_predictions))

    save_model(home_model, "home_goals")
    save_model(away_model, "away_goals")
