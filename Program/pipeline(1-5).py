from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    target_pairs: int = 400
    min_observations: int = 180
    selection_end_date: str = "2017-07-31"
    chunksize: int = 2_000_000
    store_cap_multiplier: float = 1.8
    family_cap_multiplier: float = 2.0

    @property
    def train_path(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def items_path(self) -> Path:
        return self.data_dir / "items.csv"

    @property
    def stores_path(self) -> Path:
        return self.data_dir / "stores.csv"

    @property
    def holidays_path(self) -> Path:
        return self.data_dir / "holidays_events.csv"


def ensure_output_dirs(config: PipelineConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)


def load_reference_tables(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    items = pd.read_csv(config.items_path)
    stores = pd.read_csv(config.stores_path)
    holidays = pd.read_csv(config.holidays_path, parse_dates=["date"])

    items["family"] = items["family"].astype("category")
    stores["city"] = stores["city"].astype("category")
    stores["state"] = stores["state"].astype("category")
    stores["type"] = stores["type"].astype("category")

    return {"items": items, "stores": stores, "holidays": holidays}


def compute_candidate_pair_statistics(
    config: PipelineConfig,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(config.selection_end_date)
    usecols = ["date", "store_nbr", "item_nbr", "unit_sales"]
    dtypes = {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}
    pair_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        config.train_path,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["date"],
        chunksize=config.chunksize,
    ):
        chunk = chunk.loc[chunk["date"] <= cutoff].copy()
        if chunk.empty:
            continue

        chunk["clipped_sales"] = chunk["unit_sales"].clip(lower=0)
        grouped = (
            chunk.groupby(["store_nbr", "item_nbr"], observed=True)
            .agg(
                observation_count=("date", "size"),
                total_sales=("clipped_sales", "sum"),
                first_date=("date", "min"),
                last_date=("date", "max"),
            )
            .reset_index()
        )
        pair_parts.append(grouped)

    if not pair_parts:
        raise ValueError("No train.csv rows were found before the configured selection_end_date.")

    pair_stats = pd.concat(pair_parts, ignore_index=True)
    pair_stats = (
        pair_stats.groupby(["store_nbr", "item_nbr"], observed=True)
        .agg(
            observation_count=("observation_count", "sum"),
            total_sales=("total_sales", "sum"),
            first_date=("first_date", "min"),
            last_date=("last_date", "max"),
        )
        .reset_index()
    )
    pair_stats["calendar_days"] = (
        pair_stats["last_date"] - pair_stats["first_date"]
    ).dt.days + 1
    pair_stats["coverage"] = (
        pair_stats["observation_count"] / pair_stats["calendar_days"]
    ).astype("float32")
    pair_stats["average_sales"] = (
        pair_stats["total_sales"] / pair_stats["observation_count"]
    ).astype("float32")
    return pair_stats.sort_values(["store_nbr", "item_nbr"]).reset_index(drop=True)


def _scaled(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=False) <= 1:
        return pd.Series(np.zeros(len(series), dtype="float32"), index=series.index)
    values = series.astype("float64")
    scaled = (values - values.min()) / (values.max() - values.min())
    return scaled.astype("float32")


def _quantile_labels(series: pd.Series, labels: list[str]) -> pd.Series:
    ranked = series.rank(method="first")
    bins = min(len(labels), max(1, series.nunique()))
    active = labels[:bins]
    return pd.qcut(ranked, q=bins, labels=active, duplicates="drop").astype("category")


def select_store_item_pairs(
    pair_stats: pd.DataFrame,
    items: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    candidates = pair_stats.loc[pair_stats["observation_count"] >= config.min_observations].copy()
    candidates = candidates.merge(
        items[["item_nbr", "family", "class", "perishable"]],
        on="item_nbr",
        how="left",
        validate="many_to_one",
    )
    if candidates[["family", "class", "perishable"]].isna().any().any():
        raise ValueError("Item metadata merge introduced missing values for candidate pairs.")

    candidates["log_total_sales"] = np.log1p(candidates["total_sales"])
    candidates["history_score"] = _scaled(candidates["observation_count"])
    candidates["coverage_score"] = _scaled(candidates["coverage"])
    candidates["activity_score"] = _scaled(candidates["log_total_sales"])
    candidates["base_score"] = (
        0.40 * candidates["coverage_score"]
        + 0.35 * candidates["history_score"]
        + 0.25 * candidates["activity_score"]
    ).astype("float32")

    store_counts = candidates["store_nbr"].value_counts()
    family_counts = candidates["family"].value_counts()
    perish_counts = candidates["perishable"].value_counts()
    candidates["scarcity_bonus"] = (
        0.18 / np.sqrt(candidates["store_nbr"].map(store_counts))
        + 0.14 / np.sqrt(candidates["family"].map(family_counts).astype("float32"))
        + 0.08 / np.sqrt(candidates["perishable"].map(perish_counts).astype("float32"))
    ).astype("float32")
    candidates["priority_score"] = (
        candidates["base_score"] + candidates["scarcity_bonus"]
    ).astype("float32")

    candidates["demand_bin"] = _quantile_labels(
        candidates["average_sales"], ["very_low", "low", "mid", "high", "very_high"]
    )
    candidates["coverage_bin"] = _quantile_labels(
        candidates["coverage"], ["very_low", "low", "mid", "high", "very_high"]
    )
    candidates["history_bin"] = _quantile_labels(
        candidates["observation_count"], ["very_short", "short", "mid", "long", "very_long"]
    )

    sorted_candidates = candidates.sort_values(
        ["priority_score", "coverage", "observation_count", "total_sales"],
        ascending=False,
    ).reset_index(drop=True)

    unique_stores = sorted_candidates["store_nbr"].nunique()
    unique_families = sorted_candidates["family"].nunique()
    store_cap = max(3, int(np.ceil((config.target_pairs / unique_stores) * config.store_cap_multiplier)))
    family_cap = max(4, int(np.ceil((config.target_pairs / unique_families) * config.family_cap_multiplier)))

    selected_rows: list[pd.Series] = []
    selected_keys: set[tuple[int, int]] = set()
    store_selected: dict[int, int] = {}
    family_selected: dict[Any, int] = {}

    def _can_select(row: pd.Series) -> bool:
        key = (int(row["store_nbr"]), int(row["item_nbr"]))
        if key in selected_keys:
            return False
        if store_selected.get(int(row["store_nbr"]), 0) >= store_cap:
            return False
        if family_selected.get(row["family"], 0) >= family_cap:
            return False
        return True

    def _take(row: pd.Series) -> None:
        key = (int(row["store_nbr"]), int(row["item_nbr"]))
        selected_keys.add(key)
        selected_rows.append(row)
        store_selected[int(row["store_nbr"])] = store_selected.get(int(row["store_nbr"]), 0) + 1
        family_selected[row["family"]] = family_selected.get(row["family"], 0) + 1

    # Pass 1: guarantee broad store representation.
    for _, row in sorted_candidates.groupby("store_nbr", observed=True).head(1).iterrows():
        if len(selected_rows) >= config.target_pairs:
            break
        if _can_select(row):
            _take(row)

    # Pass 2: broaden family representation for the remaining families.
    family_heads = sorted_candidates.groupby("family", observed=True).head(3)
    for _, row in family_heads.iterrows():
        if len(selected_rows) >= config.target_pairs:
            break
        if _can_select(row):
            _take(row)

    # Pass 3: fill the rest from the globally ranked pool.
    for _, row in sorted_candidates.iterrows():
        if len(selected_rows) >= config.target_pairs:
            break
        if _can_select(row):
            _take(row)

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    if len(selected) < config.target_pairs:
        raise ValueError(
            f"Only selected {len(selected)} pairs; target_pairs={config.target_pairs} may be too large for the current caps."
        )

    return selected.sort_values(["store_nbr", "item_nbr"]).reset_index(drop=True)


def load_selected_train_rows(
    config: PipelineConfig,
    selected_pairs: pd.DataFrame,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    extraction_end_date = pd.Timestamp(
        config.selection_end_date if end_date is None else end_date
    )
    selected_index = pd.MultiIndex.from_frame(selected_pairs[["store_nbr", "item_nbr"]])
    usecols = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
    dtypes = {
        "store_nbr": "int16",
        "item_nbr": "int32",
        "unit_sales": "float32",
        "onpromotion": "boolean",
    }
    collected: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        config.train_path,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["date"],
        chunksize=config.chunksize,
    ):
        chunk = chunk.loc[chunk["date"] <= extraction_end_date].copy()
        if chunk.empty:
            continue
        mask = pd.MultiIndex.from_arrays(
            [chunk["store_nbr"].to_numpy(), chunk["item_nbr"].to_numpy()]
        ).isin(selected_index)
        filtered = chunk.loc[mask]
        if not filtered.empty:
            collected.append(filtered)

    if not collected:
        raise ValueError("No train.csv rows matched the selected pairs.")

    subset = pd.concat(collected, ignore_index=True)
    duplicate_count = subset.duplicated(["store_nbr", "item_nbr", "date"]).sum()
    if duplicate_count:
        raise ValueError(
            f"Selected subset contains {duplicate_count} duplicate store-item-date rows before preprocessing."
        )
    return subset.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)


def preprocess_selected_subset(
    subset: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    items: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    processed = subset.copy()
    processed["promotion_missing"] = processed["onpromotion"].isna().astype("int8")
    processed["onpromotion"] = processed["onpromotion"].fillna(False).astype("int8")
    processed["unit_sales"] = processed["unit_sales"].clip(lower=0).astype("float32")
    processed["was_observed"] = np.int8(1)

    processed = processed.merge(
        items[["item_nbr", "family", "class", "perishable"]],
        on="item_nbr",
        how="left",
        validate="many_to_one",
    )
    processed = processed.merge(
        stores[["store_nbr", "city", "state", "type", "cluster"]],
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )
    if len(processed) != len(subset):
        raise ValueError("Metadata merges changed the selected row count.")
    if processed[
        ["family", "class", "perishable", "city", "state", "type", "cluster"]
    ].isna().any().any():
        raise ValueError("Metadata merges introduced missing values in the selected subset.")

    reindexed_parts: list[pd.DataFrame] = []
    for (store_nbr, item_nbr), group in processed.groupby(["store_nbr", "item_nbr"], sort=False):
        group = group.sort_values("date").copy()
        full_index = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        group = group.set_index("date").reindex(full_index)
        group.index.name = "date"

        # Internal missing dates are treated as zero-demand days for modeling.
        group["store_nbr"] = int(store_nbr)
        group["item_nbr"] = int(item_nbr)
        group["unit_sales"] = group["unit_sales"].fillna(0).astype("float32")
        group["onpromotion"] = group["onpromotion"].fillna(0).astype("int8")
        group["promotion_missing"] = group["promotion_missing"].fillna(1).astype("int8")
        group["was_observed"] = group["was_observed"].fillna(0).astype("int8")

        for col in ["family", "class", "perishable", "city", "state", "type", "cluster"]:
            group[col] = group[col].ffill().bfill()

        reindexed_parts.append(group.reset_index())

    daily = pd.concat(reindexed_parts, ignore_index=True)
    if daily.duplicated(["store_nbr", "item_nbr", "date"]).sum():
        raise ValueError("Duplicate store-item-date rows appeared after reindexing.")
    if (daily["unit_sales"] < 0).any():
        raise ValueError("Negative unit_sales remained after preprocessing.")

    expected_pairs = len(selected_pairs)
    actual_pairs = daily[["store_nbr", "item_nbr"]].drop_duplicates().shape[0]
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} selected pairs after preprocessing but found {actual_pairs}."
        )
    return daily.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["day_of_week"] = featured["date"].dt.dayofweek.astype("int8")
    featured["month"] = featured["date"].dt.month.astype("int8")
    featured["day_of_month"] = featured["date"].dt.day.astype("int8")
    featured["week_of_year"] = featured["date"].dt.isocalendar().week.astype("int16")

    featured["day_of_week_sin"] = np.sin(2 * np.pi * featured["day_of_week"] / 7).astype("float32")
    featured["day_of_week_cos"] = np.cos(2 * np.pi * featured["day_of_week"] / 7).astype("float32")
    featured["month_sin"] = np.sin(2 * np.pi * featured["month"] / 12).astype("float32")
    featured["month_cos"] = np.cos(2 * np.pi * featured["month"] / 12).astype("float32")

    # Cyclical encoding for day of month (period 31)
    featured["day_of_month_sin"] = np.sin(2 * np.pi * featured["day_of_month"] / 31).astype("float32")
    featured["day_of_month_cos"] = np.cos(2 * np.pi * featured["day_of_month"] / 31).astype("float32")

    # Cyclical encoding for week of year (period 53 to accommodate ISO week 53 without discontinuity)
    featured["week_of_year_sin"] = np.sin(2 * np.pi * featured["week_of_year"] / 53).astype("float32")
    featured["week_of_year_cos"] = np.cos(2 * np.pi * featured["week_of_year"] / 53).astype("float32")

    # Normalized time_idx: elapsed days since dataset start, normalized strictly by training span (1657 days)
    dataset_start_date = pd.Timestamp("2013-01-01")
    train_end_date = pd.Timestamp("2017-07-16")
    train_span_days = float((train_end_date - dataset_start_date).days)  # 1657.0
    featured["time_idx"] = (
        (featured["date"] - dataset_start_date).dt.days / train_span_days
    ).astype("float32")
    return featured


def build_holiday_features(
    daily: pd.DataFrame,
    stores: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    selected_stores = daily["store_nbr"].drop_duplicates().sort_values()
    store_frame = stores.loc[stores["store_nbr"].isin(selected_stores)].copy()
    store_dates = (
        daily[["store_nbr", "date"]]
        .drop_duplicates()
        .merge(store_frame[["store_nbr", "city", "state"]], on="store_nbr", how="left", validate="many_to_one")
    )

    active_holidays = holidays.loc[~holidays["transferred"]].copy()

    national = active_holidays.loc[active_holidays["locale"] == "National", ["date", "type"]].copy()
    national["key"] = 1

    regional = active_holidays.loc[active_holidays["locale"] == "Regional", ["date", "type", "locale_name"]].copy()
    local = active_holidays.loc[active_holidays["locale"] == "Local", ["date", "type", "locale_name"]].copy()

    store_dates["key"] = 1
    national_hits = store_dates.merge(national, on=["date", "key"], how="left")
    regional_hits = store_dates.merge(
        regional,
        left_on=["date", "state"],
        right_on=["date", "locale_name"],
        how="left",
    )
    local_hits = store_dates.merge(
        local,
        left_on=["date", "city"],
        right_on=["date", "locale_name"],
        how="left",
    )

    stacked = pd.concat(
        [
            national_hits[["store_nbr", "date", "type"]],
            regional_hits[["store_nbr", "date", "type"]],
            local_hits[["store_nbr", "date", "type"]],
        ],
        ignore_index=True,
    ).dropna(subset=["type"])

    if stacked.empty:
        features = store_dates[["store_nbr", "date"]].copy()
        features["is_holiday"] = 0
        features["is_event"] = 0
        features["is_additional"] = 0
        features["is_work_day"] = 0
        features["is_transfer"] = 0
        return features.drop(columns=["key"])

    stacked["is_holiday"] = stacked["type"].isin(["Holiday", "Bridge"]).astype("int8")
    stacked["is_event"] = stacked["type"].eq("Event").astype("int8")
    stacked["is_additional"] = stacked["type"].eq("Additional").astype("int8")
    stacked["is_work_day"] = stacked["type"].eq("Work Day").astype("int8")
    stacked["is_transfer"] = stacked["type"].eq("Transfer").astype("int8")

    features = (
        stacked.groupby(["store_nbr", "date"], observed=True)[
            ["is_holiday", "is_event", "is_additional", "is_work_day", "is_transfer"]
        ]
        .max()
        .reset_index()
    )
    return store_dates[["store_nbr", "date"]].merge(
        features,
        on=["store_nbr", "date"],
        how="left",
    ).fillna(0)


def create_feature_engineered_dataset(
    daily: pd.DataFrame,
    stores: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    featured = create_calendar_features(daily)
    holiday_features = build_holiday_features(featured, stores, holidays)
    featured = featured.merge(
        holiday_features,
        on=["store_nbr", "date"],
        how="left",
        validate="many_to_one",
    )
    holiday_cols = ["is_holiday", "is_event", "is_additional", "is_work_day", "is_transfer"]
    featured[holiday_cols] = featured[holiday_cols].fillna(0).astype("int8")

    for col in ["family", "city", "state", "type"]:
        featured[col] = featured[col].astype("category")

    if featured.duplicated(["store_nbr", "item_nbr", "date"]).sum():
        raise ValueError("Duplicate store-item-date rows appeared after feature engineering.")
    return featured.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)


def summarize_selected_pairs(selected_pairs: pd.DataFrame) -> dict[str, Any]:
    return {
        "selected_pairs": int(len(selected_pairs)),
        "selected_stores": int(selected_pairs["store_nbr"].nunique()),
        "selected_families": int(selected_pairs["family"].nunique()),
        "perishable_pairs": int(selected_pairs["perishable"].eq(1).sum()),
        "non_perishable_pairs": int(selected_pairs["perishable"].eq(0).sum()),
        "min_history": int(selected_pairs["observation_count"].min()),
        "median_history": float(selected_pairs["observation_count"].median()),
        "max_history": int(selected_pairs["observation_count"].max()),
        "min_coverage": float(selected_pairs["coverage"].min()),
        "median_coverage": float(selected_pairs["coverage"].median()),
        "max_coverage": float(selected_pairs["coverage"].max()),
        "avg_sales_median": float(selected_pairs["average_sales"].median()),
        "date_min": str(selected_pairs["first_date"].min().date()),
        "date_max": str(selected_pairs["last_date"].max().date()),
    }


def validate_stage(df: pd.DataFrame, stage_name: str) -> dict[str, Any]:
    return {
        "stage": stage_name,
        "rows": int(len(df)),
        "pairs": int(df[["store_nbr", "item_nbr"]].drop_duplicates().shape[0]),
        "duplicates": int(df.duplicated(["store_nbr", "item_nbr", "date"]).sum()),
        "negative_sales": int((df["unit_sales"] < 0).sum()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "missing_values": df.isna().sum().loc[lambda s: s > 0].to_dict(),
    }
