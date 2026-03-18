from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class RunConfig:
    """Notebook-friendly run configuration for the regime_model workflow."""

    model_name: str = "regime_model"
    round_name: str = "round_1"
    long_short: int = -1
    debug_rows: Optional[int] = 50000

    ins_start_date: Optional[str] = None
    oos_start_date: Optional[str] = None
    insample_fraction: float = 0.80
    train_fraction: float = 0.80

    target_column: str = "wins"
    probability_column: str = "ml_proba_1"

    save_intermediate: bool = True
    save_model: bool = True
    save_tables: bool = True
    save_figures: bool = False
    save_feature_columns: bool = True
    use_existing_intermediate: bool = True
    overwrite_outputs: bool = True

    use_selected_features: bool = True
    tuning_mode: str = "fast"
    random_state: int = 42
    search_rows: int = 1000
    n_iter: int = 60
    cv_splits: int = 5
    search_n_jobs: int = 1
    search_n_estimators: int = 400
    final_n_estimators_cap: int = 4000
    early_stopping_rounds: int = 100
    target_lookback_months: int = 9
    # ADDED
    classification_threshold: float = 0.5
    #
    rfecv_step: int = 1
    rfecv_min_features_to_select: int = 5
    rfecv_scoring: str = "precision"
    rfecv_n_jobs: int = 1
    entry_fee: float = 3.5

    master_data_filename: str = "master_variable_inventory 20251223.xlsx"
    liquidity_filename: str = "liquidity_indices_updated_20260309.xlsx"
    fear_greed_filename: str = "fear_and_greed_full.csv"
    raw_event_filenames: Tuple[str, ...] = field(
        default_factory=lambda: (
            "BoMoS_v2_1min_a 02-24 20_41_30_010220_022026.csv",
            "BoMoS_v2_1min_b 02-25 16_13_11_010213_123119.csv",
        )
    )
    intermediate_partition_names: Tuple[str, ...] = field(
        default_factory=lambda: (
            "partition_ins_80",
            "partition_ins_20",
            "partition_oos",
            "partition_ins",
        )
    )
    scored_partition_names: Tuple[str, ...] = field(
        default_factory=lambda: (
            "partition_ins_80_001",
            "partition_ins_20_001",
            "partition_oos_001",
        )
    )
    artifact_prefix: str = "xgb"

    def __post_init__(self) -> None:
        if self.long_short not in (-1, 1):
            raise ValueError("long_short must be -1 or 1.")
        for name, value in (
            ("insample_fraction", self.insample_fraction),
            ("train_fraction", self.train_fraction),
        ):
            if not (0 < float(value) < 1):
                raise ValueError("%s must be between 0 and 1." % name)
        if self.tuning_mode not in ("fast",):
            raise ValueError("Unsupported tuning_mode: %s" % self.tuning_mode)

    @property
    def side_label(self) -> str:
        return "long" if self.long_short == 1 else "short"

    @property
    def model_artifact_stem(self) -> str:
        return "%s_%s_%s" % (self.artifact_prefix, self.model_name, self.side_label)
