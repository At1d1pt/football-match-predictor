"""
Plots metric curves for both models:

Window 1 — XGBoost Classifier (result prediction)
  - Confusion Matrix
  - ROC Curves (per-class, with AUC)
  - Precision-Recall Curves (per-class, with AP)
  - Training Loss Curve (mlogloss)

Window 2 — XGBoost Regressor (goals prediction)
  - Actual vs Predicted scatter (Home goals)
  - Actual vs Predicted scatter (Away goals)
  - Residual distributions (Home & Away)
  - Training Loss Curves (Poisson loss, Home & Away)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier, XGBRegressor

from train import load_data, prepare_data, split_data
from goals_model import prepare_data as prepare_goals_data

# ══════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════
df = load_data()

# Classifier data
X_cls, y_cls, feature_columns = prepare_data(df)
X_cls_train, X_cls_test, y_cls_train, y_cls_test = split_data(X_cls, y_cls)

label_map   = {"H": 0, "D": 1, "A": 2}
reverse_map = {v: k for k, v in label_map.items()}
class_names = ["H", "D", "A"]

y_cls_train_enc = y_cls_train.map(label_map)
y_cls_test_enc  = y_cls_test.map(label_map)

# Regressor data
X_reg, y_home, y_away, _ = prepare_goals_data(df)
X_reg_train, X_reg_test, y_home_train, y_home_test = split_data(X_reg, y_home)
_, _, y_away_train, y_away_test = split_data(X_reg, y_away)

# ══════════════════════════════════════════════════════════════
#  TRAIN CLASSIFIER
# ══════════════════════════════════════════════════════════════
print("Training classifier...")
classifier = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    eval_metric="mlogloss",
    subsample=1.0,
    reg_lambda=1,
    reg_alpha=0,
    n_estimators=200,
    min_child_weight=1,
    max_depth=5,
    learning_rate=0.1,
    gamma=0.1,
    colsample_bytree=0.8,
)

classifier.fit(
    X_cls_train, y_cls_train_enc,
    eval_set=[(X_cls_train, y_cls_train_enc), (X_cls_test, y_cls_test_enc)],
    verbose=False,
)

y_pred = classifier.predict(X_cls_test)
y_prob = classifier.predict_proba(X_cls_test)
y_test_bin = label_binarize(y_cls_test_enc, classes=[0, 1, 2])

# ══════════════════════════════════════════════════════════════
#  TRAIN REGRESSORS
# ══════════════════════════════════════════════════════════════
print("Training home goals regressor...")
home_model = XGBRegressor(
    objective="count:poisson",
    random_state=42,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=500,
    eval_metric="poisson-nloglik",
)

home_model.fit(
    X_reg_train, y_home_train,
    eval_set=[(X_reg_train, y_home_train), (X_reg_test, y_home_test)],
    verbose=False,
)

print("Training away goals regressor...")
away_model = XGBRegressor(
    objective="count:poisson",
    random_state=42,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=500,
    eval_metric="poisson-nloglik",
)

away_model.fit(
    X_reg_train, y_away_train,
    eval_set=[(X_reg_train, y_away_train), (X_reg_test, y_away_test)],
    verbose=False,
)

home_preds = home_model.predict(X_reg_test)
away_preds = away_model.predict(X_reg_test)

# ══════════════════════════════════════════════════════════════
#  WINDOW 1 — CLASSIFIER METRICS
# ══════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 11))
fig1.suptitle("XGBoost Classifier — Metric Curves", fontsize=16, fontweight="bold")

colors = {"H": "#2ecc71", "D": "#f1c40f", "A": "#e74c3c"}

# 1. Confusion Matrix
ax = axes1[0, 0]
cm = confusion_matrix(y_cls_test_enc, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix")

# 2. ROC Curves
ax = axes1[0, 1]
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[cls], lw=2, label=f"{cls}  (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (One-vs-Rest)")
ax.legend(loc="lower right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

# 3. Precision-Recall Curves
ax = axes1[1, 0]
for i, cls in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_prob[:, i])
    ax.plot(recall, precision, color=colors[cls], lw=2, label=f"{cls}  (AP = {ap:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves")
ax.legend(loc="lower left")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

# 4. Loss Curve
ax = axes1[1, 1]
cls_results = classifier.evals_result()
cls_train_loss = cls_results["validation_0"]["mlogloss"]
cls_val_loss   = cls_results["validation_1"]["mlogloss"]
epochs = range(1, len(cls_train_loss) + 1)
ax.plot(epochs, cls_train_loss, color="#3498db", lw=2, label="Train")
ax.plot(epochs, cls_val_loss,   color="#e74c3c", lw=2, label="Validation")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("Log Loss (mlogloss)")
ax.set_title("Training & Validation Loss")
ax.legend()

fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig("models/classifier_metrics.png", dpi=150)
print("Saved: models/classifier_metrics.png")

# ══════════════════════════════════════════════════════════════
#  WINDOW 2 — REGRESSOR METRICS
# ══════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 3, figsize=(20, 11))
fig2.suptitle("XGBoost Goals Regressor — Metric Curves", fontsize=16, fontweight="bold")

home_color = "#3498db"
away_color = "#e74c3c"

home_mae = mean_absolute_error(y_home_test, home_preds)
away_mae = mean_absolute_error(y_away_test, away_preds)
home_rmse = np.sqrt(mean_squared_error(y_home_test, home_preds))
away_rmse = np.sqrt(mean_squared_error(y_away_test, away_preds))

# 1. Actual vs Predicted — Home
ax = axes2[0, 0]
ax.scatter(y_home_test, home_preds, alpha=0.3, s=15, color=home_color, edgecolors="none")
lims = [0, max(y_home_test.max(), home_preds.max()) + 0.5]
ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
ax.set_xlabel("Actual Home Goals")
ax.set_ylabel("Predicted Home Goals")
ax.set_title(f"Home Goals — Actual vs Predicted\nMAE={home_mae:.3f}  RMSE={home_rmse:.3f}")
ax.set_xlim(lims); ax.set_ylim(lims)

# 2. Actual vs Predicted — Away
ax = axes2[0, 1]
ax.scatter(y_away_test, away_preds, alpha=0.3, s=15, color=away_color, edgecolors="none")
lims = [0, max(y_away_test.max(), away_preds.max()) + 0.5]
ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
ax.set_xlabel("Actual Away Goals")
ax.set_ylabel("Predicted Away Goals")
ax.set_title(f"Away Goals — Actual vs Predicted\nMAE={away_mae:.3f}  RMSE={away_rmse:.3f}")
ax.set_xlim(lims); ax.set_ylim(lims)

# 3. Prediction Error by Actual Goal Count
ax = axes2[0, 2]
home_residuals = home_preds - y_home_test.values
away_residuals = away_preds - y_away_test.values
goal_vals = sorted(y_home_test.unique())
home_means = [home_residuals[y_home_test.values == g].mean() for g in goal_vals]
away_means = [away_residuals[y_away_test.values == g].mean() for g in goal_vals if g in y_away_test.values]
away_goal_vals = [g for g in goal_vals if g in y_away_test.values]
bar_width = 0.35
x = np.arange(len(goal_vals))
ax.bar(x - bar_width/2, home_means, bar_width, color=home_color, alpha=0.8, label="Home")
x_away = np.arange(len(away_goal_vals))
ax.bar(x_away + bar_width/2, away_means, bar_width, color=away_color, alpha=0.8, label="Away")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels([int(g) for g in goal_vals])
ax.set_xlabel("Actual Goals")
ax.set_ylabel("Mean Prediction Error")
ax.set_title("Mean Error by Actual Goal Count")
ax.legend()

# 4. Residual Distribution — Home
ax = axes2[1, 0]
ax.hist(home_residuals, bins=40, color=home_color, alpha=0.7, edgecolor="white")
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_xlabel("Residual (Predicted − Actual)")
ax.set_ylabel("Frequency")
ax.set_title(f"Home Goals — Residual Distribution\nMean={home_residuals.mean():.3f}  Std={home_residuals.std():.3f}")

# 5. Residual Distribution — Away
ax = axes2[1, 1]
ax.hist(away_residuals, bins=40, color=away_color, alpha=0.7, edgecolor="white")
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_xlabel("Residual (Predicted − Actual)")
ax.set_ylabel("Frequency")
ax.set_title(f"Away Goals — Residual Distribution\nMean={away_residuals.mean():.3f}  Std={away_residuals.std():.3f}")

# 6. Training Loss Curves (Home & Away)
ax = axes2[1, 2]
home_results = home_model.evals_result()
away_results = away_model.evals_result()

metric_key = "poisson-nloglik"
home_train_loss = home_results["validation_0"][metric_key]
home_val_loss   = home_results["validation_1"][metric_key]
away_train_loss = away_results["validation_0"][metric_key]
away_val_loss   = away_results["validation_1"][metric_key]
epochs = range(1, len(home_train_loss) + 1)

ax.plot(epochs, home_train_loss, color=home_color, lw=1.5, ls="-",  label="Home Train")
ax.plot(epochs, home_val_loss,   color=home_color, lw=1.5, ls="--", label="Home Val")
ax.plot(epochs, away_train_loss, color=away_color, lw=1.5, ls="-",  label="Away Train")
ax.plot(epochs, away_val_loss,   color=away_color, lw=1.5, ls="--", label="Away Val")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("Poisson NLL")
ax.set_title("Training & Validation Loss")
ax.legend(fontsize=9)

fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig("models/regressor_metrics.png", dpi=150)
print("Saved: models/regressor_metrics.png")

plt.show()
