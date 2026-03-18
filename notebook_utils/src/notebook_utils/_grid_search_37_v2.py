from __future__ import annotations
import inspect
import numpy as np
from typing import Dict, Any, Tuple, Optional

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import recall_score, make_scorer, classification_report
from scipy.stats import loguniform, uniform
import xgboost as xgb

def build_base_xgb(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.1,
        n_estimators=300,      # search-time cap; final fit overrides
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        nthread=-1,           # xgboost 0.80
        random_state=random_state,
        tree_method="hist",
    )

def param_distributions(search_n_estimators: int, seed: int) -> Dict[str, Any]:
    return {
        "n_estimators": [search_n_estimators],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "gamma": uniform(0.0, 0.6),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": loguniform(1e-2, 1e2),
        "reg_alpha": loguniform(1e-4, 1e1),
        "reg_lambda": loguniform(1e-2, 1e2),
    }

def randomized_search_on_sample(
    X, y,
    *,
    search_rows: int = 200_000,
    n_iter: int = 60,
    cv_splits: int = 3,
    scoring: str = "recall",
    random_state: int = 42,
    search_n_estimators: int = 400,
    n_jobs: int = 1,
) -> Tuple[Dict[str, Any], RandomizedSearchCV]:
    """
    RandomizedSearchCV on a stratified subset drawn via train_test_split.
    """
    n_samples = len(X)
    n_classes = np.unique(np.asarray(y)).size

    # If asked rows >= available, just use all data (no split).
    if (search_rows is None) or (search_rows >= n_samples):
        Xs, ys = X, y
    else:
        # Make sure we leave at least one sample per class for the test split.
        max_train_allowed = max(1, n_samples - n_classes)
        n_train = int(min(search_rows, max_train_allowed))
        if n_train < 1:
            # Fallback: if dataset is tiny or classes are too many, skip sampling
            Xs, ys = X, y
        else:
            try:
                Xs, _, ys, _ = train_test_split(
                    X, y,
                    train_size=n_train,      # integer count
                    stratify=y,
                    shuffle=True,
                    random_state=random_state,
                )
            except ValueError:
                # Fallback when stratify can't satisfy class constraints
                Xs, _, ys, _ = train_test_split(
                    X, y,
                    train_size=n_train,
                    shuffle=True,
                    random_state=random_state,
            )

    base = build_base_xgb(random_state=random_state)
    dists = param_distributions(search_n_estimators, random_state)

    scorer = make_scorer(recall_score) if scoring == "recall" else scoring
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    search_kwargs = {
        "estimator": base,
        "param_distributions": dists,
        "n_iter": n_iter,
        "scoring": scorer,
        "cv": cv,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbose": 2,
        "refit": False,   # final refit with ES below
    }
    if "iid" in inspect.signature(RandomizedSearchCV).parameters:
        search_kwargs["iid"] = True  # sklearn 0.20.3 compatibility

    search = RandomizedSearchCV(**search_kwargs)
    search.fit(Xs, ys)
    return search.best_params_, search

def final_refit_with_early_stopping(
    X, y,
    best_params: Dict[str, Any],
    *,
    final_n_estimators_cap: int = 4000,
    early_stopping_rounds: int = 100,
    val_size: float = 0.1,
    random_state: int = 42,
    eval_metric: str = "logloss",
) -> xgb.XGBClassifier:
    """
    One full-data fit with Early Stopping to pick the best iteration.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )

    final_params = dict(best_params)
    final_params.update({
        "n_estimators": final_n_estimators_cap,
        "learning_rate": 0.1,
        "objective": "binary:logistic",
        "tree_method": "hist",
        "nthread": -1,
        "random_state": random_state,
    })

    model = xgb.XGBClassifier(**final_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=eval_metric,
        verbose=False,
        early_stopping_rounds=early_stopping_rounds,
    )
    return model

def tune_xgb_fast_compatible(
    X_train, y_train,
    X_test=None, y_test=None,
    *,
    search_rows: int = 200_000,
    n_iter: int = 60,
    cv_splits: int = 3,
    search_n_estimators: int = 400,
    final_n_estimators_cap: int = 4000,
    early_stopping_rounds: int = 100,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    End-to-end: randomized search on a stratified subset -> single ES refit on all data.
    """
    best_params, _ = randomized_search_on_sample(
        X_train, y_train,
        search_rows=search_rows,
        n_iter=n_iter,
        cv_splits=cv_splits,
        search_n_estimators=search_n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model = final_refit_with_early_stopping(
        X_train, y_train,
        best_params=best_params,
        final_n_estimators_cap=final_n_estimators_cap,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )

    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        print("Test recall:", recall_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    return model, best_params

# Example:
# model, best = tune_xgb_fast_compatible(X_train, y_train, X_test, y_test, search_rows=200_000, n_iter=60, cv_splits=5)
# print("Best params:", best)
# print("Best ntree limit:", getattr(model, "best_ntree_limit", None))
# print("Best iteration:", getattr(model, "best_iteration", None))
