from train import load_data, prepare_data, split_data
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pandas as pd

classifier = load("models/model_xgboost.pkl")
home_model = load("models/model_home_goals.pkl")
away_model = load("models/model_away_goals.pkl")

label_map = {
    "H": 0,
    "D": 1,
    "A": 2
}

reverse_map = {
    0: "H",
    1: "D",
    2: "A"
}

df = load_data()

X, y, feature_columns = prepare_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

classifier_predictions = classifier.predict(X_test)

home_predictions = home_model.predict(X_test)
away_predictions = away_model.predict(X_test)

def goals_to_result(home, away, threshold=0.25):
    if abs(home - away) <= threshold:
        return "D"

    elif home > away:
        return "H"

    return "A"

goal_predictions = [goals_to_result(h, a) for h, a in zip(home_predictions, away_predictions)]

classifier_accuracy = accuracy_score(y_test, [reverse_map[p] for p in classifier_predictions])
goal_accuracy = accuracy_score(y_test, goal_predictions)

classifier_predictions = [reverse_map[p] for p in classifier_predictions]

print("=" * 60)
print("XGBOOST CLASSIFIER")
print("=" * 60)

print(f"Accuracy: {accuracy_score(y_test, classifier_predictions):.4f}\n")

print(classification_report(
    y_test,
    classifier_predictions,
    target_names=["H", "D", "A"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, classifier_predictions))

print("\n" + "=" * 60)
print("GOAL REGRESSION MODEL")
print("=" * 60)

print(f"Accuracy: {accuracy_score(y_test, goal_predictions):.4f}\n")

print(classification_report(
    y_test,
    goal_predictions,
    target_names=["H", "D", "A"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, goal_predictions))

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Classifier": classifier_predictions,
    "Goal Model": goal_predictions,
    "Pred Home Goals": home_predictions.round(2),
    "Pred Away Goals": away_predictions.round(2)
})

print("\nSample Predictions")
print(comparison.head(20))

disagreement = comparison[
    comparison["Classifier"] != comparison["Goal Model"]
]

print(f"\nModels disagree on {len(disagreement)} matches.\n")
print(disagreement.head(20))