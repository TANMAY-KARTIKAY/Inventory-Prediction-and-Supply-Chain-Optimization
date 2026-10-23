from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TRAIN_END_DATE = pd.Timestamp("2017-07-16")
VALIDATION_END_DATE = pd.Timestamp("2017-07-31")
TEST_START_DATE = pd.Timestamp("2017-08-01")
TEST_END_DATE = pd.Timestamp("2017-08-15")


def describe_split(name: str, frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "split": name,
        "rows": int(len(frame)),
        "pairs": int(frame[["store_nbr", "item_nbr"]].drop_duplicates().shape[0]),
        "date_min": None if frame.empty else str(frame["date"].min().date()),
        "date_max": None if frame.empty else str(frame["date"].max().date()),
        "duplicate_store_item_dates": int(
            frame.duplicated(["store_nbr", "item_nbr", "date"]).sum()
        ),
    }
    if name == "test":
        observed_pairs = frame.loc[frame["was_observed"].eq(1), ["store_nbr", "item_nbr"]]
        summary["pairs_with_observed_sales_rows"] = int(observed_pairs.drop_duplicates().shape[0])
    return summary


def main() -> None:
    output_dir = Path(r"C:\CODE\favorita_inventory_pipeline\outputs")
    feature_path = output_dir / "feature_engineered_daily_through_2017-08-15.csv"

    featured = pd.read_csv(feature_path, parse_dates=["date"])
    source_min = featured["date"].min()
    source_max = featured["date"].max()

    train = featured.loc[featured["date"] <= TRAIN_END_DATE].copy()
    validation = featured.loc[
        (featured["date"] > TRAIN_END_DATE)
        & (featured["date"] <= VALIDATION_END_DATE)
    ].copy()
    test = featured.loc[
        (featured["date"] >= TEST_START_DATE)
        & (featured["date"] <= TEST_END_DATE)
    ].copy()

    splits = {"train": train, "validation": validation, "test": test}
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}_split.csv", index=False)

    report = {
        "source_date_min": str(source_min.date()),
        "source_date_max": str(source_max.date()),
        "cutoffs": {
            "train_end_date": str(TRAIN_END_DATE.date()),
            "validation_start_date": str((TRAIN_END_DATE + pd.Timedelta(days=1)).date()),
            "validation_end_date": str(VALIDATION_END_DATE.date()),
            "test_start_date": str(TEST_START_DATE.date()),
            "test_end_date": str(TEST_END_DATE.date()),
        },
        "splits": [describe_split(name, frame) for name, frame in splits.items()],
    }
    print(json.dumps(report, indent=2))

    if source_max < TEST_START_DATE:
        print(
            "WARNING: The source feature file ends before the approved test window. "
            "test_split.csv is intentionally empty; no dates were reassigned or fabricated."
        )


if __name__ == "__main__":
    main()
