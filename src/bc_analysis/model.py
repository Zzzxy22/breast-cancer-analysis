"""分类建模：交叉验证比较多个模型，并在留出测试集上评估最佳模型。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SCORING = ["roc_auc", "accuracy", "recall", "precision", "f1"]


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, C=1.0))]
        ),
        "SVM (RBF)": Pipeline(
            [("scale", StandardScaler()), ("clf", CalibratedClassifierCV(SVC(random_state=random_state), ensemble=False))]
        ),
        "Random Forest": Pipeline(
            [("clf", RandomForestClassifier(n_estimators=300, random_state=random_state))]
        ),
    }


@dataclass
class ModelResults:
    cv_scores: pd.DataFrame
    best_name: str
    test_metrics: dict[str, float]
    confusion: np.ndarray
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = field(default_factory=dict)
    coefficients: pd.Series | None = None


def evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> ModelResults:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    models = build_models(random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING)
        row = {"model": name}
        for m in SCORING:
            vals = scores[f"test_{m}"]
            row[f"{m}_mean"] = vals.mean()
            row[f"{m}_std"] = vals.std()
        rows.append(row)
    cv_scores = (
        pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    )
    best_name = cv_scores.loc[0, "model"]

    roc = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc[name] = (fpr, tpr, roc_auc_score(y_test, proba))

    best = models[best_name]
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    test_metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "n_test": int(len(y_test)),
    }

    # 逻辑回归的标准化系数可直接解释特征方向与强度
    lr = models["Logistic Regression"]
    coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=X.columns, name="coefficient")

    return ModelResults(
        cv_scores=cv_scores,
        best_name=best_name,
        test_metrics=test_metrics,
        confusion=confusion_matrix(y_test, pred, labels=[0, 1]),
        roc_curves=roc,
        coefficients=coefs,
    )
