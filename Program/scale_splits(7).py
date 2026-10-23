from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(r"C:\CODE\favorita_inventory_pipeline\outputs")
SOURCE_FILES = {
    "train": OUTPUT_DIR / "train_split.csv",
    "validation": OUTPUT_DIR / "validation_split.csv",
    "test": OUTPUT_DIR / "test_split.csv",
}
TARGET_COLUMN = "unit_sales"
SCALED_TARGET_COLUMN = "unit_sales_scaled"
EXPECTED_RANGES = {
    "train": ("2013-01-01", "2017-07-16"),
    "validation": ("2017-07-17", "2017-07-31"),
    "test": ("2017-08-01", "2017-08-15"),
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source_split(name: str, frame: pd.DataFrame) -> None:
    start_date, end_date = EXPECTED_RANGES[name]
    dates = pd.to_datetime(frame["date"])
    if dates.min() < pd.Timestamp(start_date) or dates.max() > pd.Timestamp(end_date):
        raise ValueError(f"{name} contains dates outside its approved range.")
    if frame.duplicated(["store_nbr", "item_nbr", "date"]).any():
        raise ValueError(f"{name} contains duplicate store-item-date rows.")
    if (frame[TARGET_COLUMN] < 0).any():
        raise ValueError(f"{name} contains negative sales values.")


def transform_target(frame: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    scaled = frame.copy()
    log_sales = np.log1p(scaled[TARGET_COLUMN].to_numpy(dtype="float64"))
    # Keep float64 so saved target values round-trip through the inverse transform.
    scaled[SCALED_TARGET_COLUMN] = scaler.transform(log_sales.reshape(-1, 1))
    return scaled


def split_summary(name: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "split": name,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "rows": int(len(frame)),
        "date_min": str(pd.to_datetime(frame["date"]).min().date()),
        "date_max": str(pd.to_datetime(frame["date"]).max().date()),
        "scaled_target_min": float(frame[SCALED_TARGET_COLUMN].min()),
        "scaled_target_max": float(frame[SCALED_TARGET_COLUMN].max()),
        "scaled_target_mean": float(frame[SCALED_TARGET_COLUMN].mean()),
        "scaled_target_std": float(frame[SCALED_TARGET_COLUMN].std(ddof=0)),
    }


def main() -> None:
    source_hashes_before = {name: file_sha256(path) for name, path in SOURCE_FILES.items()}
    splits = {name: pd.read_csv(path) for name, path in SOURCE_FILES.items()}
    for name, frame in splits.items():
        validate_source_split(name, frame)

    # Fit exclusively on log-transformed training demand; validation/test are transform-only.
    train_log_sales = np.log1p(splits["train"][TARGET_COLUMN].to_numpy(dtype="float64"))
    target_scaler = StandardScaler().fit(train_log_sales.reshape(-1, 1))

    scaled_splits = {
        name: transform_target(frame, target_scaler) for name, frame in splits.items()
    }
    output_files = {
        name: OUTPUT_DIR / f"{name}_split_scaled.csv" for name in SOURCE_FILES
    }
    for name, frame in scaled_splits.items():
        frame.to_csv(output_files[name], index=False)

    artifact = {
        "target_column": TARGET_COLUMN,
        "scaled_target_column": SCALED_TARGET_COLUMN,
        "pre_transform": "log1p",
        "post_transform": "StandardScaler",
        "inverse_transform": "expm1(target_scaler.inverse_transform(predictions))",
        "target_scaler": target_scaler,
        "fit_split": "train",
    }
    scaler_path = OUTPUT_DIR / "target_sales_scaler.joblib"
    joblib.dump(artifact, scaler_path)

    recovered = np.expm1(
        target_scaler.inverse_transform(
            scaled_splits["train"][SCALED_TARGET_COLUMN].to_numpy(dtype="float64").reshape(-1, 1)
        ).ravel()
    )
    inverse_max_abs_error = float(
        np.max(np.abs(recovered - splits["train"][TARGET_COLUMN].to_numpy(dtype="float64")))
    )

    source_hashes_after = {name: file_sha256(path) for name, path in SOURCE_FILES.items()}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A source split changed during scaling.")

    report = {
        "fit_split": "train",
        "target_scaling": {
            "raw_column": TARGET_COLUMN,
            "scaled_column": SCALED_TARGET_COLUMN,
            "method": "log1p followed by StandardScaler",
            "n_train_samples_seen": int(target_scaler.n_samples_seen_),
            "train_log1p_mean": float(target_scaler.mean_[0]),
            "train_log1p_variance": float(target_scaler.var_[0]),
            "train_log1p_scale": float(target_scaler.scale_[0]),
            "inverse_max_abs_error": inverse_max_abs_error,
        },
        "unchanged_feature_groups": {
            "identifiers": ["date", "store_nbr", "item_nbr"],
            "categorical_or_embedding_candidates": [
                "family", "class", "perishable", "city", "state", "type", "cluster"
            ],
            "binary_indicators": [
                "onpromotion", "promotion_missing", "was_observed", "is_holiday",
                "is_event", "is_additional", "is_work_day", "is_transfer"
            ],
            "calendar_codes": ["day_of_week", "month", "day_of_month", "week_of_year"],
            "already_bounded_cyclical_features": [
                "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
                "day_of_month_sin", "day_of_month_cos", "week_of_year_sin", "week_of_year_cos"
            ],
            "normalized_temporal_features": ["time_idx"],
        },
        "source_hashes_before_and_after": source_hashes_after,
        "outputs": {name: str(path) for name, path in output_files.items()},
        "scaler_artifact": str(scaler_path),
        "splits": [split_summary(name, frame) for name, frame in scaled_splits.items()],
        "sample": scaled_splits["train"][
            ["date", "store_nbr", "item_nbr", TARGET_COLUMN, SCALED_TARGET_COLUMN]
        ].head(5).to_dict(orient="records"),
    }
    report_path = OUTPUT_DIR / "scaling_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
