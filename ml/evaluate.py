"""
Evaluation and metrics computation module for Ban Quality ML.
Calculates classification metrics, confusion matrix, error analysis, and baseline comparison.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression


def find_best_threshold_on_validation(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    min_recall_class_1: Optional[float] = None,
) -> float:
    """
    Select optimal classification threshold using validation predictions.
    Defaults to maximizing F1-score on class 1, optionally constrained by min recall.
    """
    thresholds = np.linspace(0.20, 0.80, 61)
    best_thresh = 0.50
    best_score = -1.0

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        rec = recall_score(y_true, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        
        if min_recall_class_1 is not None and rec < min_recall_class_1:
            continue
            
        if f1 > best_score:
            best_score = f1
            best_thresh = float(round(t, 2))

    return best_thresh


def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Calculate ROC-AUC if both classes exist in y_true
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    else:
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    
    # Specificity: TN / (TN + FP)
    tn = cm[0][0]
    fp = cm[0][1]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision_class_1": prec1,
        "recall_class_1": rec1,
        "f1_class_1": f1,
        "roc_auc": roc_auc,
        "balanced_accuracy": bal_acc,
        "specificity": specificity,
        "confusion_matrix": cm,
    }


def perform_error_analysis(
    df_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> Dict[str, Any]:
    """
    Extract False Positives (actual=0, pred=1) and False Negatives (actual=1, pred=0).
    """
    records = df_raw.copy()
    records["actual"] = y_true
    records["predicted"] = y_pred
    records["probability_bad"] = [round(float(p), 4) for p in y_probs]

    false_positives = records[(records["actual"] == 0) & (records["predicted"] == 1)]
    false_negatives = records[(records["actual"] == 1) & (records["predicted"] == 0)]

    return {
        "false_positives_count": len(false_positives),
        "false_negatives_count": len(false_negatives),
        "false_positives_sample": false_positives.head(5).to_dict(orient="records"),
        "false_negatives_sample": false_negatives.head(5).to_dict(orient="records"),
    }


def evaluate_baseline_logistic_regression(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit and evaluate Logistic Regression baseline model for sanity comparison.
    """
    clf = LogisticRegression(random_state=42, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)
    y_probs = clf.predict_proba(X_test_scaled)[:, 1]
    metrics = compute_metrics(y_test, y_probs, threshold=0.50)
    return {
        "model": "LogisticRegression (baseline)",
        "metrics": metrics,
    }
