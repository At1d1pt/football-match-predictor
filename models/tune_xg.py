from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from train import load_data, prepare_data, split_data, evaluate_model, save_model

df = load_data()
X, y, feature_columns = prepare_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

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

y_train = y_train.map(label_map)
y_test = y_test.map(label_map)

model = XGBClassifier(
    objective="multiclass",
    num_class=3,
    random_state=42
)

param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_alpha": [0, 0.1, 0.5, 1],
    "reg_lambda": [0.5, 1, 2, 5]
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1,
    verbose=2
)

search.fit(X_train, y_train)
print(search.best_params_)
print(search.best_score_)
best_model = search.best_estimator_

predictions = best_model.predict(X_test)

evaluate_model(best_model, X_test, y_test, feature_columns)

