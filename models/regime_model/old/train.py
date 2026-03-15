"""Training and export utilities for the regime model."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_UTILS_SRC = PROJECT_ROOT / "notebook_utils" / "src"
if str(NOTEBOOK_UTILS_SRC) not in sys.path:
    sys.path.append(str(NOTEBOOK_UTILS_SRC))

try:
    from notebook_utils._optimization_functions_37_v2 import plot_features_vs_cvscore_rfecv_020
except Exception:
    plot_features_vs_cvscore_rfecv_020 = None


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Object of type %s is not JSON serializable" % type(value).__name__)


def _split_on_dates(df: pd.DataFrame, fraction: float, date_col: str = "normed_date") -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    dates = np.sort(pd.to_datetime(df[date_col]).dropna().dt.normalize().unique())
    if len(dates) < 3:
        raise ValueError("Need at least three distinct dates for chronological splitting.")

    split_idx = int(round(len(dates) * fraction))
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    split_date = pd.Timestamp(dates[split_idx])

    left = df[pd.to_datetime(df[date_col]).dt.normalize() < split_date].copy()
    right = df[pd.to_datetime(df[date_col]).dt.normalize() >= split_date].copy()
    return left, right, split_date.strftime("%Y-%m-%d")


def create_chronological_splits(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Replicate the notebook's 85/15 then 80/20 chronological split pattern."""
    ins_df, oos_df, oos_break = _split_on_dates(df, 0.85)
    train_df, test_df, test_break = _split_on_dates(ins_df, 0.80)
    split_info = {
        "oos_break_date": oos_break,
        "validation_break_date": test_break,
    }
    return {
        "train": train_df,
        "test": test_df,
        "oos": oos_df,
    }, split_info


def build_xgb_classifier(params: Dict[str, float], random_state: int = 42) -> xgb.XGBClassifier:
    """Build an xgboost 0.80-compatible classifier."""
    model_params = {
        "objective": "binary:logistic",
        "learning_rate": 0.1,
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "gamma": float(params["gamma"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "min_child_weight": float(params["min_child_weight"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "random_state": int(random_state),
        "nthread": -1,
        "tree_method": "hist",
    }
    return xgb.XGBClassifier(**model_params)


def _sample_param_grid(n_iter: int, random_state: int = 42) -> List[Dict[str, float]]:
    rng = np.random.RandomState(random_state)
    samples = []
    max_depth_choices = [3, 4, 5, 6, 8, 10]
    for _ in range(n_iter):
        sample = {
            "n_estimators": 400,
            "max_depth": int(rng.choice(max_depth_choices)),
            "gamma": float(rng.uniform(0.0, 0.6)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "min_child_weight": float(10 ** rng.uniform(-2, 2)),
            "reg_alpha": float(10 ** rng.uniform(-4, 1)),
            "reg_lambda": float(10 ** rng.uniform(-2, 2)),
        }
        samples.append(sample)
    return samples


def _time_series_cv_recall(X: pd.DataFrame, y: pd.Series, params: Dict[str, float], cv_splits: int, random_state: int) -> float:
    cv = TimeSeriesSplit(n_splits=cv_splits)
    scores = []
    for train_idx, val_idx in cv.split(X):
        model = build_xgb_classifier(params, random_state=random_state)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], eval_metric="logloss", verbose=False)
        pred = model.predict(X.iloc[val_idx])
        scores.append(recall_score(y.iloc[val_idx], pred))
    return float(np.mean(scores))


def tune_xgb_time_series(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_iter: int = 20,
    cv_splits: int = 3,
    random_state: int = 42,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Tune XGBoost hyperparameters with chronological cross-validation only."""
    trials = []
    for params in _sample_param_grid(n_iter=n_iter, random_state=random_state):
        mean_recall = _time_series_cv_recall(
            X_train,
            y_train,
            params=params,
            cv_splits=cv_splits,
            random_state=random_state,
        )
        trial = dict(params)
        trial["mean_recall"] = mean_recall
        trials.append(trial)

    trials_df = pd.DataFrame(trials).sort_values("mean_recall", ascending=False).reset_index(drop=True)
    best_params = {}
    for key, value in trials_df.iloc[0].drop("mean_recall").to_dict().items():
        if key in ("n_estimators", "max_depth"):
            best_params[key] = int(value)
        else:
            best_params[key] = float(value)
    return best_params, trials_df


def fit_with_early_stopping(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, float],
    *,
    random_state: int = 42,
    validation_fraction: float = 0.10,
    early_stopping_rounds: int = 50,
) -> xgb.XGBClassifier:
    """Fit the model using a trailing chronological validation slice for early stopping."""
    if len(X_train) < 20:
        model = build_xgb_classifier(params, random_state=random_state)
        model.fit(X_train, y_train, eval_metric="logloss", verbose=False)
        return model

    split_idx = int(round(len(X_train) * (1.0 - validation_fraction)))
    split_idx = min(max(split_idx, 5), len(X_train) - 5)

    X_fit = X_train.iloc[:split_idx]
    y_fit = y_train.iloc[:split_idx]
    X_eval = X_train.iloc[split_idx:]
    y_eval = y_train.iloc[split_idx:]

    model = build_xgb_classifier(params, random_state=random_state)
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_eval, y_eval)],
        eval_metric="logloss",
        verbose=False,
        early_stopping_rounds=early_stopping_rounds,
    )
    return model


def run_rfecv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, float],
    *,
    cv_splits: int = 3,
    step: int = 1,
    min_features_to_select: int = 2,
    random_state: int = 42,
) -> Tuple[List[str], RFECV]:
    """Select features using RFECV with TimeSeriesSplit."""
    estimator = build_xgb_classifier(params, random_state=random_state)
    rfecv = RFECV(
        estimator=estimator,
        step=step,
        min_features_to_select=min_features_to_select,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring="recall",
        n_jobs=1,
    )
    rfecv.fit(X_train, y_train)
    selected_features = X_train.columns[rfecv.support_].tolist()
    return selected_features, rfecv


def score_split(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    """Generate metrics and probabilities for a single data split."""
    pred = model.predict(X)
    pred_proba = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, pred_proba)),
        "classification_report": classification_report(y, pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "probability_mean": float(np.mean(pred_proba)),
    }
    return {
        "metrics": metrics,
        "predictions": pred,
        "probabilities": pred_proba,
    }


def render_roc_curve(
    y_train: pd.Series,
    p_train: np.ndarray,
    y_test: pd.Series,
    p_test: np.ndarray,
    output_path: Path,
) -> None:
    """Save the train-vs-test ROC curve."""
    fpr_train, tpr_train, _ = roc_curve(y_train, p_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, p_test)
    auc_train = roc_auc_score(y_train, p_train)
    auc_test = roc_auc_score(y_test, p_test)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_train, tpr_train, label="Train ROC (AUC = %.2f)" % auc_train)
    plt.plot(fpr_test, tpr_test, label="Test ROC (AUC = %.2f)" % auc_test)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Regime Model")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def export_rfecv_plot(rfecv: RFECV, X_train: pd.DataFrame, output_path: Path) -> None:
    """Persist the sklearn 0.20.x-compatible RFECV plot."""
    if plot_features_vs_cvscore_rfecv_020 is not None:
        try:
            plot_features_vs_cvscore_rfecv_020(rfecv, X_train, scoring_label="recall", increasing_x=True)
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            plt.close("all")
            return
        except Exception:
            plt.close("all")

    if hasattr(rfecv, "cv_results_"):
        scores = np.asarray(rfecv.cv_results_["mean_test_score"], dtype=float)
    else:
        scores = np.asarray(getattr(rfecv, "grid_scores_", []), dtype=float)
    x_axis = np.arange(1, len(scores) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, scores)
    plt.xlabel("RFECV Iteration")
    plt.ylabel("Mean CV Recall")
    plt.title("RFECV - Recall vs Iteration")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def export_feature_importance_plot(model: xgb.XGBClassifier, feature_columns: Iterable[str], output_path: Path) -> None:
    """Save feature importance for the selected model."""
    importances = pd.Series(model.feature_importances_, index=list(feature_columns)).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    importances.head(20).sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Regime Model Feature Importance")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def export_model_artifacts(
    *,
    model: xgb.XGBClassifier,
    feature_columns: List[str],
    metadata: Dict[str, object],
    model_dir: Path,
) -> None:
    """Save the portable model artifact bundle."""
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(str(model_dir / "model.pkl"), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(str(model_dir / "feature_columns.pkl"), "wb") as handle:
        pickle.dump(feature_columns, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(str(model_dir / "model_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=_json_default)
