from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import (
    PipelineConfig,
    create_feature_engineered_dataset,
    load_reference_tables,
    load_selected_train_rows,
    preprocess_selected_subset,
    validate_stage,
)


EXTENSION_END_DATE = "2017-08-15"


def main() -> None:
    data_dir = Path(
        r"C:\Users\tanma\OneDrive\Desktop\final dataset co\Corporación Favorita Grocery Sales Forecasting 2"
    )
    output_dir = Path(r"C:\CODE\favorita_inventory_pipeline\outputs")
    config = PipelineConfig(data_dir=data_dir, output_dir=output_dir)

    # Read the approved selection; this runner never computes candidate statistics.
    selected_pairs = pd.read_csv(output_dir / "selected_pairs.csv", parse_dates=["first_date", "last_date"])
    refs = load_reference_tables(config)

    print(f"Loading the approved {len(selected_pairs)} pairs through {EXTENSION_END_DATE}.")
    raw_selected = load_selected_train_rows(
        config,
        selected_pairs,
        end_date=EXTENSION_END_DATE,
    )
    print("Raw selected subset shape:", raw_selected.shape)
    print("Negative sales before clipping:", int((raw_selected["unit_sales"] < 0).sum()))

    processed = preprocess_selected_subset(
        raw_selected,
        selected_pairs,
        refs["items"],
        refs["stores"],
    )
    print("Preprocessing validation:", validate_stage(processed, "preprocessing"))
    processed.to_csv(output_dir / "processed_daily_through_2017-08-15.csv", index=False)

    featured = create_feature_engineered_dataset(processed, refs["stores"], refs["holidays"])
    print("Feature engineering validation:", validate_stage(featured, "feature_engineering"))
    featured.to_csv(
        output_dir / "feature_engineered_daily_through_2017-08-15.csv",
        index=False,
    )
    print("Saved extended feature-engineered data through", featured["date"].max().date())


if __name__ == "__main__":
    main()
