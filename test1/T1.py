from pathlib import Path
import sys
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, make_scorer)

# Read data.
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

if "label" not in train.columns or "label" not in test.columns:
    raise ValueError("Both files must contain the target column named 'label'.")
if set(train.columns) != set(test.columns):
    raise ValueError("Train and test files must have the same columns.")

# Exclude a row with no label.
train = train.dropna(subset=["label"]).copy()
test_labeled = test.dropna(subset=["label"]).copy()
X_train = train.drop(columns="label")
y_train = train["label"].astype(str).str.strip().str.lower() #Just save label.
X_test = test_labeled.drop(columns="label")
y_test = test_labeled["label"].astype(str).str.strip().str.lower()
if set(y_train.unique()) != {"no", "yes"}:
    raise ValueError("Expected label values 'no' and 'yes'.")

# Extract data type
numeric_cols = [c for c in X_train.columns if is_numeric_dtype(X_train[c])]
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

# Data cleaning
def make_preprocessor(scale_numeric):
    num_steps = [("fill", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)
    categorical_pipe = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

# Model and hyperparameter setup
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=10) # make sure it is reproducible
f1_yes = make_scorer(f1_score, pos_label="yes")

logistic = Pipeline([
    ("prepare", make_preprocessor(scale_numeric=True)),
    ("classifier", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)),
])
logistic_grid = {
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__class_weight": [None, "balanced"],
}

forest = Pipeline([
    ("prepare", make_preprocessor(scale_numeric=False)),
    ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
])
forest_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 16],
    "classifier__min_samples_leaf": [1, 3],
    "classifier__class_weight": [None, "balanced"],
}

# Model training
searches = {
    "Logistic Regression": GridSearchCV(logistic, logistic_grid, scoring=f1_yes,
                                         cv=cv, n_jobs=1, refit=True),
    "Random Forest": GridSearchCV(forest, forest_grid, scoring=f1_yes,
                                  cv=cv, n_jobs=1, refit=True),
}

print(f"Training records used: {len(train):,}; test records with labels: {len(test_labeled):,}")
print(f"Train label distribution: {y_train.value_counts().to_dict()}")
print(f"Test label distribution: {y_test.value_counts().to_dict()}")
print(f"Train missing feature cells: {int(X_train.isna().sum().sum()):,}")
print(f"Test missing feature cells: {int(X_test.isna().sum().sum()):,}")

rows = []
for name, search in searches.items():
    print(f"\nTuning {name}...")
    search.fit(X_train, y_train)
    model = search.best_estimator_
    pred = model.predict(X_test)
    yes_index = list(model.named_steps["classifier"].classes_).index("yes")#?????
    prob_yes = model.predict_proba(X_test)[:, yes_index]
    cm = confusion_matrix(y_test, pred, labels=["no", "yes"])
    metrics = {
        "Model": name,
        "CV best F1 (yes)": search.best_score_,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (yes)": precision_score(y_test, pred, pos_label="yes", zero_division=0),
        "Recall (yes)": recall_score(y_test, pred, pos_label="yes", zero_division=0),
        "F1 (yes)": f1_score(y_test, pred, pos_label="yes", zero_division=0),
        "ROC-AUC": roc_auc_score((y_test == "yes").astype(int), prob_yes),
        "Best parameters": search.best_params_,
        "Confusion matrix [no, yes]": cm.tolist(),
    }
    rows.append(metrics)
    print(metrics)

results = pd.DataFrame(rows)
results.to_csv("model_comparison.csv", index=False, encoding="utf-8-sig")

# Predict unlabeled test row
if test["label"].isna().any():
    missing_mask = test["label"].isna()
    test_features = test.loc[missing_mask].drop(columns="label")
    winner_name = results.sort_values("CV best F1 (yes)", ascending=False).iloc[0]["Model"]
    winner = searches[winner_name].best_estimator_
    predicted = winner.predict(test_features)
    pd.DataFrame({"row_index": test.index[missing_mask], "predicted_label": predicted}).to_csv(
        "unlabeled_test_predictions.csv", index=False, encoding="utf-8-sig")

print("\nSaved model_comparison.csv")
if Path("unlabeled_test_predictions.csv").exists():
    print("Saved unlabeled_test_predictions.csv")

