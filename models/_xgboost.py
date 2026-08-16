"""
This module trains a model using XGBoost. XGBoost is relatively better for tabular data than Random Forest.
"""

from train import load_data, prepare_data, split_data, evaluate_model, save_model
from xgboost import XGBClassifier

from joblib import dump

df = load_data()
X, y, feature_columns = prepare_data(df)
dump(feature_columns, "models/feature_columns.pkl")
X_train, X_test, y_train, y_test = split_data(X, y)

label_map = {
    "H": 0,
    "D": 1,
    "A": 2
}

reverse_map = {v: k for k, v in label_map.items()}

y_train = y_train.map(label_map)
y_test = y_test.map(label_map)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    eval_metric="mlogloss",
    subsample=0.7,
    reg_lambda=5,
    reg_alpha=1,
    n_estimators=300,
    min_child_weight=5,
    max_depth=5,
    learning_rate=0.03,
    gamma=0.1,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

evaluate_model(model, X_test, y_test, feature_columns)
save_model(model, name="xgboost")


# WORK IN PROGRESS

class ResultPredictionModel(XGBClassifier):
    def __init__(self):
        self._objective = "multi:softprob"
        self._num_class = 3
        self._random_state = 42
        self._eval_metric = "mlogloss"
        self._subsample = 0.7
        self._reg_lambda = 5
        self._reg_alpha = 5
        self._n_estimators = 300
        self._min_child_weight = 5
        self._max_depth = 5
        self._learning_rate = 0.03
        self._gamma = 0.1
        self._colsample_bytree = 0.8

        self.tune()

        super().__init__(    
            objective=self._objective,
            num_class=self._num_class,
            random_state=self._random_state,
            eval_metric=self._eval_metric,
            subsample=self._subsample,
            reg_lambda=self._reg_lambda,
            reg_alpha=self._reg_alpha,
            n_estimators=self._n_estimators,
            min_child_weight=self._min_child_weight,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            gamma=self._gamma,
            colsample_bytree=self._colsample_bytree
        )

        def tune():
            """Tune the parameters"""
