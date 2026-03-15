
from __future__ import annotations

from IPython.display import display
from pprint import pprint
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detect_datetime_col(df: pd.DataFrame, candidates: Optional[Iterable[str]] = None) -> Optional[str]:
    if candidates is None:
        candidates = ["timestamp", "datetime", "date", "normed_date", "day"]
    return next((c for c in candidates if c in df.columns), None)


def detect_target_col(df: pd.DataFrame, candidates: Optional[Iterable[str]] = None) -> Optional[str]:
    if candidates is None:
        candidates = ["target", "y", "label", "market_down", "target_downside"]
    return next((c for c in candidates if c in df.columns), None)


def detect_probability_col(df: pd.DataFrame, candidates: Optional[Iterable[str]] = None) -> Optional[str]:
    if candidates is None:
        candidates = [
            "pred_proba",
            "pred_prob",
            "probability",
            "score",
            "proba",
            "downside_proba",
            "predicted_probability",
        ]
    exact = next((c for c in candidates if c in df.columns), None)
    if exact is not None:
        return exact

    fuzzy = [
        c for c in df.columns
        if ("prob" in c.lower() or "proba" in c.lower() or "score" in c.lower())
    ]
    return fuzzy[0] if fuzzy else None


def detect_prediction_col(df: pd.DataFrame, candidates: Optional[Iterable[str]] = None) -> Optional[str]:
    if candidates is None:
        candidates = ["pred_class", "prediction", "pred", "y_pred"]
    exact = next((c for c in candidates if c in df.columns), None)
    if exact is not None:
        return exact

    fuzzy = [
        c for c in df.columns
        if "pred" in c.lower() and "prob" not in c.lower() and "proba" not in c.lower()
    ]
    return fuzzy[0] if fuzzy else None


def prepare_df_for_diagnostics(
    df: pd.DataFrame,
    dt_col: Optional[str] = None,
    sort: bool = True,
) -> tuple[pd.DataFrame, Optional[str]]:
    out = df.copy()
    if dt_col is None:
        dt_col = detect_datetime_col(out)

    if dt_col is not None:
        out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
        if sort:
            out = out.sort_values(dt_col).reset_index(drop=True)

    return out, dt_col


def print_dataset_overview(df: pd.DataFrame, name: str = "dataset") -> None:
    print(f"\n=== OVERVIEW: {name} ===")
    print("Shape:", df.shape)
    print("Columns:")
    pprint(df.columns.tolist())

    dt_col = detect_datetime_col(df)
    target_col = detect_target_col(df)
    prob_col = detect_probability_col(df)
    pred_col = detect_prediction_col(df)

    print("\nDetected columns:")
    print("Datetime:", dt_col)
    print("Target:", target_col)
    print("Probability:", prob_col)
    print("Prediction:", pred_col)

    if dt_col is not None:
        print("\nDate range:")
        print(df[dt_col].min(), "->", df[dt_col].max())


def plot_target_balance(df: pd.DataFrame, target_col: Optional[str] = None, name: str = "dataset") -> None:
    if target_col is None:
        target_col = detect_target_col(df)
    if target_col is None:
        print(f"[{name}] No target column detected.")
        return

    counts = df[target_col].value_counts(dropna=False).sort_index()
    display(counts.to_frame("count"))

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title(f"Target Balance - {name}")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_columns: list[str], name: str = "dataset") -> None:
    fi = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    display(fi.to_frame("importance"))

    plt.figure(figsize=(10, 5))
    fi.sort_values().plot(kind="barh")
    plt.title(f"XGBoost Feature Importance - {name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_feature_distributions_by_target(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: Optional[str] = None,
    name: str = "dataset",
    max_features: Optional[int] = None,
) -> None:
    if target_col is None:
        target_col = detect_target_col(df)
    if target_col is None:
        print(f"[{name}] No target column detected.")
        return

    cols = [c for c in feature_columns if c in df.columns]
    if max_features is not None:
        cols = cols[:max_features]

    for col in cols:
        plt.figure(figsize=(8, 4))
        df.loc[df[target_col] == 0, col].dropna().hist(bins=40, alpha=0.5, label="Target 0")
        df.loc[df[target_col] == 1, col].dropna().hist(bins=40, alpha=0.5, label="Target 1")
        plt.title(f"{name} - Distribution of {col} by Target")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()


def feature_summary_by_target(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    if target_col is None:
        target_col = detect_target_col(df)
    if target_col is None:
        print("No target column detected.")
        return None

    cols = [c for c in feature_columns if c in df.columns]
    summary = df.groupby(target_col)[cols].agg(["mean", "std", "median"])
    display(summary)
    return summary


def plot_feature_correlation(df: pd.DataFrame, feature_columns: list[str], name: str = "dataset") -> None:
    cols = [c for c in feature_columns if c in df.columns]
    corr = df[cols].corr()
    display(corr)

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title(f"Feature Correlation Matrix - {name}")
    plt.tight_layout()
    plt.show()


def plot_probability_distribution(df: pd.DataFrame, prob_col: Optional[str] = None, name: str = "dataset") -> None:
    if prob_col is None:
        prob_col = detect_probability_col(df)
    if prob_col is None:
        print(f"[{name}] No probability column detected.")
        return

    plt.figure(figsize=(8, 4))
    df[prob_col].dropna().hist(bins=50)
    plt.title(f"Predicted Probability Distribution - {name}")
    plt.xlabel(prob_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_probability_by_target(
    df: pd.DataFrame,
    prob_col: Optional[str] = None,
    target_col: Optional[str] = None,
    name: str = "dataset",
) -> None:
    if prob_col is None:
        prob_col = detect_probability_col(df)
    if target_col is None:
        target_col = detect_target_col(df)

    if prob_col is None or target_col is None:
        print(f"[{name}] Missing probability or target column.")
        return

    plt.figure(figsize=(8, 4))
    df.loc[df[target_col] == 0, prob_col].dropna().hist(bins=40, alpha=0.5, label="Target 0")
    df.loc[df[target_col] == 1, prob_col].dropna().hist(bins=40, alpha=0.5, label="Target 1")
    plt.title(f"Predicted Probability by Target - {name}")
    plt.xlabel(prob_col)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_probability_through_time(
    df: pd.DataFrame,
    prob_col: Optional[str] = None,
    dt_col: Optional[str] = None,
    name: str = "dataset",
) -> None:
    if prob_col is None:
        prob_col = detect_probability_col(df)
    if dt_col is None:
        dt_col = detect_datetime_col(df)

    if prob_col is None or dt_col is None:
        print(f"[{name}] Missing probability or datetime column.")
        return

    tmp = df[[dt_col, prob_col]].dropna().sort_values(dt_col)

    plt.figure(figsize=(12, 5))
    plt.plot(tmp[dt_col], tmp[prob_col])
    plt.title(f"Predicted Regime Probability Through Time - {name}")
    plt.xlabel(dt_col)
    plt.ylabel(prob_col)
    plt.tight_layout()
    plt.show()


def plot_daily_average_probability(
    df: pd.DataFrame,
    prob_col: Optional[str] = None,
    dt_col: Optional[str] = None,
    name: str = "dataset",
) -> None:
    if prob_col is None:
        prob_col = detect_probability_col(df)
    if dt_col is None:
        dt_col = detect_datetime_col(df)

    if prob_col is None or dt_col is None:
        print(f"[{name}] Missing probability or datetime column.")
        return

    tmp = df[[dt_col, prob_col]].dropna().copy()
    tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
    tmp["day"] = tmp[dt_col].dt.normalize()

    daily_prob = tmp.groupby("day")[prob_col].mean()
    display(daily_prob.to_frame("avg_pred_proba"))

    plt.figure(figsize=(12, 5))
    plt.plot(daily_prob.index, daily_prob.values)
    plt.title(f"Daily Average Regime Probability - {name}")
    plt.xlabel("Day")
    plt.ylabel("Average probability")
    plt.tight_layout()
    plt.show()


def calibration_table(
    df: pd.DataFrame,
    prob_col: Optional[str] = None,
    target_col: Optional[str] = None,
    q: int = 10,
    name: str = "dataset",
) -> Optional[pd.DataFrame]:
    if prob_col is None:
        prob_col = detect_probability_col(df)
    if target_col is None:
        target_col = detect_target_col(df)

    if prob_col is None or target_col is None:
        print(f"[{name}] Missing probability or target column.")
        return None

    tmp = df[[prob_col, target_col]].dropna().copy()
    tmp["proba_bucket"] = pd.qcut(tmp[prob_col], q=q, duplicates="drop")

    bucket_stats = tmp.groupby("proba_bucket").agg(
        avg_proba=(prob_col, "mean"),
        realized_rate=(target_col, "mean"),
        count=(target_col, "size"),
    )
    display(bucket_stats)

    plt.figure(figsize=(8, 5))
    plt.plot(bucket_stats["avg_proba"].values, label="Average predicted probability")
    plt.plot(bucket_stats["realized_rate"].values, label="Realized target rate")
    plt.title(f"Calibration by Probability Bucket - {name}")
    plt.xlabel("Bucket")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return bucket_stats


def feature_audit_table(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    cols = [c for c in feature_columns if c in df.columns]
    audit = pd.DataFrame({
        "feature": cols,
        "null_count": [df[c].isna().sum() for c in cols],
        "mean": [df[c].mean() for c in cols],
        "std": [df[c].std() for c in cols],
        "min": [df[c].min() for c in cols],
        "max": [df[c].max() for c in cols],
    })
    display(audit)
    return audit


def run_dataset_diagnostics(
    df: pd.DataFrame,
    name: str = "dataset",
    feature_columns: Optional[list[str]] = None,
    model=None,
    max_features_for_hist: int = 10,
    show_feature_importance: bool = True,
) -> None:
    df, dt_col = prepare_df_for_diagnostics(df)

    print_dataset_overview(df, name=name)

    target_col = detect_target_col(df)
    prob_col = detect_probability_col(df)

    plot_target_balance(df, target_col=target_col, name=name)

    if feature_columns is not None:
        feature_audit_table(df, feature_columns)
        plot_feature_correlation(df, feature_columns, name=name)
        plot_feature_distributions_by_target(
            df=df,
            feature_columns=feature_columns,
            target_col=target_col,
            name=name,
            max_features=max_features_for_hist,
        )
        feature_summary_by_target(df, feature_columns, target_col=target_col)

    plot_probability_distribution(df, prob_col=prob_col, name=name)
    plot_probability_by_target(df, prob_col=prob_col, target_col=target_col, name=name)
    plot_probability_through_time(df, prob_col=prob_col, dt_col=dt_col, name=name)
    plot_daily_average_probability(df, prob_col=prob_col, dt_col=dt_col, name=name)
    calibration_table(df, prob_col=prob_col, target_col=target_col, name=name)

    if model is not None and feature_columns is not None and show_feature_importance:
        plot_feature_importance(model, feature_columns, name=name)


def compare_probability_summary(dfs: list[pd.DataFrame], names: list[str]) -> pd.DataFrame:
    rows = []
    for df, name in zip(dfs, names):
        prob_col = detect_probability_col(df)
        target_col = detect_target_col(df)

        row = {
            "dataset": name,
            "rows": len(df),
            "prob_col": prob_col,
            "target_col": target_col,
        }

        if prob_col is not None:
            row["prob_mean"] = df[prob_col].mean()
            row["prob_std"] = df[prob_col].std()
            row["prob_min"] = df[prob_col].min()
            row["prob_max"] = df[prob_col].max()

        if target_col is not None:
            row["target_rate"] = df[target_col].mean()

        rows.append(row)

    out = pd.DataFrame(rows)
    display(out)
    return out