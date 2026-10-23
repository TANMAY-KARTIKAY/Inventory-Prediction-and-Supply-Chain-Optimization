from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

INPUT_WINDOW = 28
FORECAST_HORIZON = 7
TOTAL_WINDOW = INPUT_WINDOW + FORECAST_HORIZON  # 35 days

TRAIN_END_DATE = pd.Timestamp("2017-07-16")
VAL_START_DATE = pd.Timestamp("2017-07-17")
VAL_END_DATE = pd.Timestamp("2017-07-31")
TEST_START_DATE = pd.Timestamp("2017-08-01")
TEST_END_DATE = pd.Timestamp("2017-08-15")

OUTPUT_DIR = Path(r"C:\CODE\favorita_inventory_pipeline\outputs")
SOURCE_FILES = {
    "train": OUTPUT_DIR / "train_split_scaled.csv",
    "validation": OUTPUT_DIR / "validation_split_scaled.csv",
    "test": OUTPUT_DIR / "test_split_scaled.csv",
}

PAST_FEATURES: list[str] = [
    "unit_sales_scaled",
    "onpromotion",
    "promotion_missing",
    "was_observed",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_month_sin",
    "day_of_month_cos",
    "week_of_year_sin",
    "week_of_year_cos",
    "time_idx",
    "is_holiday",
    "is_event",
    "is_additional",
    "is_work_day",
    "is_transfer",
]

FUTURE_FEATURES: list[str] = [
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_month_sin",
    "day_of_month_cos",
    "week_of_year_sin",
    "week_of_year_cos",
    "time_idx",
    "is_holiday",
    "is_event",
    "is_additional",
    "is_work_day",
    "is_transfer",
]

STATIC_METADATA_COLS: list[str] = [
    "store_nbr",
    "item_nbr",
    "family",
    "class",
    "perishable",
    "city",
    "state",
    "type",
    "cluster",
]

TARGET_COLUMN = "unit_sales_scaled"
RAW_TARGET_COLUMN = "unit_sales"


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_npz_mmap(npz_path: Path | str) -> dict[str, np.memmap]:
    """Memory-map uncompressed (.npz / ZIP_STORED) array files directly from disk."""
    npz_path = Path(npz_path)
    arrays: dict[str, np.memmap] = {}
    with zipfile.ZipFile(npz_path, "r") as z:
        for info in z.infolist():
            name = info.filename
            key = name[:-4] if name.endswith(".npy") else name
            if info.compress_type != 0:
                raise ValueError(
                    f"Array '{name}' in '{npz_path.name}' is compressed (type {info.compress_type}). "
                    "Memory-mapping requires uncompressed ZIP_STORED archives."
                )
            with open(npz_path, "rb") as f:
                f.seek(info.header_offset)
                hdr = f.read(30)
                fn_len = int.from_bytes(hdr[26:28], "little")
                ex_len = int.from_bytes(hdr[28:30], "little")
                f.seek(info.header_offset + 30 + fn_len + ex_len)
                magic = np.lib.format.read_magic(f)
                if magic == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                offset = f.tell()
                arrays[key] = np.memmap(
                    npz_path,
                    dtype=dtype,
                    mode="r",
                    offset=offset,
                    shape=shape,
                    order="F" if fortran else "C",
                )
    return arrays


class FavoritaMultivariateSequenceDataset(Dataset):
    """PyTorch Dataset for 28-day past historical (18 feats) + 7-day future known (14 feats) forecasting.
    
    Supports memory-mapped loading from .npz archives for minimal RAM usage.
    """

    def __init__(
        self,
        past_x: np.ndarray,
        future_x: np.ndarray,
        y: np.ndarray,
        y_raw: np.ndarray | None = None,
        static_metadata: dict[str, np.ndarray] | None = None,
        dates: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.past_x = past_x
        self.future_x = future_x
        self.y = y
        self.y_raw = y_raw
        self.static_metadata = static_metadata or {}
        self.dates = dates or {}

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "past_x": torch.from_numpy(np.array(self.past_x[idx], dtype=np.float32)),
            "future_x": torch.from_numpy(np.array(self.future_x[idx], dtype=np.float32)),
            "y": torch.from_numpy(np.array(self.y[idx], dtype=np.float32)),
        }
        if self.y_raw is not None:
            item["y_raw"] = torch.from_numpy(np.array(self.y_raw[idx], dtype=np.float32))
        for col, arr in self.static_metadata.items():
            val = arr[idx]
            if isinstance(val, (int, np.integer)):
                item[col] = torch.tensor(int(val), dtype=torch.long)
            elif isinstance(val, (float, np.floating)):
                item[col] = torch.tensor(float(val), dtype=torch.float32)
            else:
                item[col] = str(val)
        return item

    @classmethod
    def from_npz(
        cls, npz_path: Path | str, mmap_mode: str | None = "r"
    ) -> FavoritaMultivariateSequenceDataset:
        npz_path = Path(npz_path)
        if mmap_mode:
            data = load_npz_mmap(npz_path)
        else:
            data = dict(np.load(npz_path))
        static = {}
        for col in STATIC_METADATA_COLS:
            if col in data:
                static[col] = data[col]
        dates = {}
        for date_key in ["input_start_date", "input_end_date", "target_start_date", "target_end_date"]:
            if date_key in data:
                dates[date_key] = data[date_key]
        return cls(
            past_x=data["past_x"],
            future_x=data["future_x"],
            y=data["y"],
            y_raw=data["y_raw"] if "y_raw" in data else None,
            static_metadata=static,
            dates=dates,
        )


def build_multivariate_sequences(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    temp_dir: Path,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    print("Combining splits to construct continuous timelines per pair...")
    load_cols = list(
        dict.fromkeys(
            ["date"]
            + STATIC_METADATA_COLS
            + PAST_FEATURES
            + [RAW_TARGET_COLUMN]
        )
    )
    combined = pd.concat([train_df[load_cols], val_df[load_cols], test_df[load_cols]], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

    train_end = np.datetime64(TRAIN_END_DATE)
    val_start = np.datetime64(VAL_START_DATE)
    val_end = np.datetime64(VAL_END_DATE)
    test_start = np.datetime64(TEST_START_DATE)
    test_end = np.datetime64(TEST_END_DATE)

    # Pass 1: Count sequence counts per split and per pair
    print("Pass 1: Identifying sequence boundaries and window counts...")
    split_pair_counts: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    grouped = list(combined.groupby(["store_nbr", "item_nbr"], sort=False))

    for (store_nbr, item_nbr), group in grouped:
        dates = group["date"].to_numpy(dtype="datetime64[D]")
        n = len(dates)
        if n < TOTAL_WINDOW:
            continue

        M = n - TOTAL_WINDOW + 1
        t_starts = dates[INPUT_WINDOW : INPUT_WINDOW + M]
        t_ends = dates[TOTAL_WINDOW - 1 : TOTAL_WINDOW - 1 + M]

        tr_mask = t_ends <= train_end
        val_mask = (t_starts >= val_start) & (t_ends <= val_end)
        te_mask = (t_starts >= test_start) & (t_ends <= test_end)

        first_row = group.iloc[0]
        static_vals = {col: first_row[col] for col in STATIC_METADATA_COLS}

        for split_name, mask in [("train", tr_mask), ("validation", val_mask), ("test", te_mask)]:
            k = int(np.sum(mask))
            if k > 0:
                split_pair_counts[split_name].append({
                    "store_nbr": int(store_nbr),
                    "item_nbr": int(item_nbr),
                    "count": k,
                    "mask": mask,
                    "static": static_vals,
                })

    split_totals = {s: sum(p["count"] for p in split_pair_counts[s]) for s in ["train", "validation", "test"]}
    print(f"Calculated sequence counts: Train={split_totals['train']:,}, Val={split_totals['validation']:,}, Test={split_totals['test']:,}")

    # Pass 2: Allocate memory-mapped scratch arrays and populate
    print("Pass 2: Allocating memory-mapped temporary buffers and populating multivariate sequences...")
    memmaps: dict[str, dict[str, np.ndarray]] = {}
    for s, total_seqs in split_totals.items():
        memmaps[s] = {
            "past_x": np.lib.format.open_memmap(
                temp_dir / f"{s}_past_x.npy",
                mode="w+",
                dtype="float32",
                shape=(total_seqs, INPUT_WINDOW, len(PAST_FEATURES)),
            ),
            "future_x": np.lib.format.open_memmap(
                temp_dir / f"{s}_future_x.npy",
                mode="w+",
                dtype="float32",
                shape=(total_seqs, FORECAST_HORIZON, len(FUTURE_FEATURES)),
            ),
            "y": np.lib.format.open_memmap(
                temp_dir / f"{s}_y.npy",
                mode="w+",
                dtype="float32",
                shape=(total_seqs, FORECAST_HORIZON),
            ),
            "y_raw": np.lib.format.open_memmap(
                temp_dir / f"{s}_y_raw.npy",
                mode="w+",
                dtype="float32",
                shape=(total_seqs, FORECAST_HORIZON),
            ),
        }

    # Tracking index insertion
    split_ptrs = {s: 0 for s in ["train", "validation", "test"]}
    metadata_accum: dict[str, dict[str, list[Any]]] = {
        s: {
            "store_nbr": [],
            "item_nbr": [],
            "family": [],
            "class": [],
            "perishable": [],
            "city": [],
            "state": [],
            "type": [],
            "cluster": [],
            "input_start_date": [],
            "input_end_date": [],
            "target_start_date": [],
            "target_end_date": [],
        }
        for s in ["train", "validation", "test"]
    }

    # Pair information lookup
    pair_info_map = {
        s: {(p["store_nbr"], p["item_nbr"]): p for p in split_pair_counts[s]}
        for s in ["train", "validation", "test"]
    }

    future_indices = [PAST_FEATURES.index(f) for f in FUTURE_FEATURES]
    target_idx = PAST_FEATURES.index(TARGET_COLUMN)

    index_records = []
    current_seq_id = 0

    for (store_nbr, item_nbr), group in grouped:
        dates = group["date"].to_numpy(dtype="datetime64[D]")
        n = len(dates)
        if n < TOTAL_WINDOW:
            continue

        past_feat_matrix = group[PAST_FEATURES].to_numpy(dtype=np.float32)
        raw_target_series = group[RAW_TARGET_COLUMN].to_numpy(dtype=np.float32)

        win_past = np.lib.stride_tricks.sliding_window_view(past_feat_matrix, window_shape=TOTAL_WINDOW, axis=0)
        win_past = np.swapaxes(win_past, 1, 2)  # shape (M, 35, 18)
        M = len(win_past)

        win_raw = np.lib.stride_tricks.sliding_window_view(raw_target_series, window_shape=TOTAL_WINDOW, axis=0)
        win_y_raw = win_raw[:, INPUT_WINDOW:]  # shape (M, 7)

        in_starts = dates[:M]
        in_ends = dates[INPUT_WINDOW - 1 : INPUT_WINDOW - 1 + M]
        t_starts = dates[INPUT_WINDOW : INPUT_WINDOW + M]
        t_ends = dates[TOTAL_WINDOW - 1 : TOTAL_WINDOW - 1 + M]

        key = (int(store_nbr), int(item_nbr))

        for s in ["train", "validation", "test"]:
            if key not in pair_info_map[s]:
                continue
            p_info = pair_info_map[s][key]
            mask = p_info["mask"]
            k = p_info["count"]

            ptr = split_ptrs[s]
            # past_x: [k, 28, 18]
            memmaps[s]["past_x"][ptr : ptr + k] = win_past[mask, :INPUT_WINDOW, :]
            # future_x: [k, 7, 14]
            memmaps[s]["future_x"][ptr : ptr + k] = win_past[mask, INPUT_WINDOW:, :][:, :, future_indices]
            # y: [k, 7]
            memmaps[s]["y"][ptr : ptr + k] = win_past[mask, INPUT_WINDOW:, target_idx]
            # y_raw: [k, 7]
            memmaps[s]["y_raw"][ptr : ptr + k] = win_y_raw[mask]

            split_ptrs[s] += k

            static = p_info["static"]
            for col in STATIC_METADATA_COLS:
                metadata_accum[s][col].append(np.full(k, static[col]))

            in_s_str = in_starts[mask].astype(str)
            in_e_str = in_ends[mask].astype(str)
            t_s_str = t_starts[mask].astype(str)
            t_e_str = t_ends[mask].astype(str)

            metadata_accum[s]["input_start_date"].append(in_s_str)
            metadata_accum[s]["input_end_date"].append(in_e_str)
            metadata_accum[s]["target_start_date"].append(t_s_str)
            metadata_accum[s]["target_end_date"].append(t_e_str)

            for i in range(k):
                index_records.append({
                    "sequence_id": current_seq_id,
                    "split": s,
                    "store_nbr": int(store_nbr),
                    "item_nbr": int(item_nbr),
                    "input_start_date": str(in_s_str[i]),
                    "input_end_date": str(in_e_str[i]),
                    "target_start_date": str(t_s_str[i]),
                    "target_end_date": str(t_e_str[i]),
                })
                current_seq_id += 1

    # Flush memmaps
    for s in ["train", "validation", "test"]:
        for arr in memmaps[s].values():
            arr.flush()

    index_df = pd.DataFrame(index_records)

    assembled_outputs: dict[str, dict[str, Any]] = {}
    report_splits: dict[str, Any] = {}

    for s in ["train", "validation", "test"]:
        total_seqs = split_totals[s]
        p_counts = [p["count"] for p in split_pair_counts[s]]
        p_series = pd.Series(p_counts)

        assembled_outputs[s] = {
            "past_x": memmaps[s]["past_x"],
            "future_x": memmaps[s]["future_x"],
            "y": memmaps[s]["y"],
            "y_raw": memmaps[s]["y_raw"],
            "past_feature_names": np.array(PAST_FEATURES),
            "future_feature_names": np.array(FUTURE_FEATURES),
        }
        for col in STATIC_METADATA_COLS:
            assembled_outputs[s][col] = np.concatenate(metadata_accum[s][col])
        for date_key in ["input_start_date", "input_end_date", "target_start_date", "target_end_date"]:
            assembled_outputs[s][date_key] = np.concatenate(metadata_accum[s][date_key])

        report_splits[s] = {
            "sequence_count": int(total_seqs),
            "participating_pair_count": len(p_counts),
            "past_x_shape": list(memmaps[s]["past_x"].shape),
            "future_x_shape": list(memmaps[s]["future_x"].shape),
            "y_shape": list(memmaps[s]["y"].shape),
            "y_raw_shape": list(memmaps[s]["y_raw"].shape),
            "windows_per_pair": {
                "min": int(p_series.min()) if not p_series.empty else 0,
                "median": float(p_series.median()) if not p_series.empty else 0.0,
                "mean": float(p_series.mean()) if not p_series.empty else 0.0,
                "max": int(p_series.max()) if not p_series.empty else 0,
                "distribution": {str(k): int(v) for k, v in p_series.value_counts().items()},
            },
            "past_features": PAST_FEATURES,
            "future_features": FUTURE_FEATURES,
            "static_metadata_cols": STATIC_METADATA_COLS,
        }

    return assembled_outputs, index_df, report_splits


def main() -> None:
    print("Verifying source file SHA-256 integrity before multivariate sequence generation...")
    source_hashes_before = {name: file_sha256(path) for name, path in SOURCE_FILES.items()}

    print("Loading scaled CSV splits...")
    splits = {name: pd.read_csv(path) for name, path in SOURCE_FILES.items()}

    temp_dir_path = OUTPUT_DIR / "temp_mmap_build"
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        assembled_outputs, index_df, report_splits = build_multivariate_sequences(
            splits["train"], splits["validation"], splits["test"], temp_dir_path
        )

        print("\nExporting final multivariate sequence archives (.npz)...")
        for s in ["train", "validation", "test"]:
            out_npz = OUTPUT_DIR / f"sequences_{s}.npz"
            print(f"Writing {out_npz.name}...")
            np.savez(out_npz, **assembled_outputs[s])
            size_mb = round(out_npz.stat().st_size / (1024 * 1024), 2)
            print(f"Successfully saved {out_npz.name} ({size_mb} MB)")
            assembled_outputs[s].clear()

        del assembled_outputs
        gc.collect()

    finally:
        gc.collect()
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path, ignore_errors=True)

    print("\nSaving sequence manifest index...")
    index_path = OUTPUT_DIR / "sequence_index.csv.gz"
    index_df.to_csv(index_path, index=False, compression="gzip")
    print(f"Saved {index_path.name} ({round(index_path.stat().st_size / (1024*1024), 2)} MB)")

    summary_path = OUTPUT_DIR / "sequence_index_summary.csv"
    summary_df = (
        index_df.groupby(["split", "store_nbr", "item_nbr"])
        .agg(
            sequence_count=("sequence_id", "count"),
            input_start_first=("input_start_date", "min"),
            input_end_last=("input_end_date", "max"),
            target_start_first=("target_start_date", "min"),
            target_end_last=("target_end_date", "max"),
        )
        .reset_index()
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path.name}")

    print("Verifying source file SHA-256 integrity after generation...")
    source_hashes_after = {name: file_sha256(path) for name, path in SOURCE_FILES.items()}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A source data preparation file was modified during sequence generation!")

    report = {
        "sequence_design": {
            "input_window_days": INPUT_WINDOW,
            "forecast_horizon_days": FORECAST_HORIZON,
            "total_window_days": TOTAL_WINDOW,
            "past_feature_count": len(PAST_FEATURES),
            "future_feature_count": len(FUTURE_FEATURES),
            "past_features": PAST_FEATURES,
            "future_features": FUTURE_FEATURES,
            "static_metadata_cols": STATIC_METADATA_COLS,
            "stride": 1,
        },
        "feature_specifications": {
            "time_idx_definition": "time_idx = (date - dataset_start_date).days / train_span_days",
            "time_idx_normalization": {
                "dataset_start_date": "2013-01-01",
                "train_end_date": "2017-07-16",
                "train_span_days": 1657.0,
                "train_min_value": 0.0,
                "train_max_value": 1.0,
                "formula": "(date - Timestamp('2013-01-01')).days / 1657.0",
                "method": "Fixed min-max scaling derived strictly from training span (2013-01-01 to 2017-07-16); linear continuation for validation/test",
            },
            "cyclical_encodings": {
                "day_of_week": "sin/cos with period 7 (0 to 6)",
                "month": "sin/cos with period 12 (1 to 12)",
                "day_of_month": "sin/cos with period 31: sin(2*pi*day/31), cos(2*pi*day/31)",
                "week_of_year": "sin/cos with period 53: sin(2*pi*week/53), cos(2*pi*week/53) to smoothly accommodate ISO week 53 without discontinuity",
            },
        },
        "cutoffs": {
            "train_end_date": str(TRAIN_END_DATE.date()),
            "validation_start_date": str(VAL_START_DATE.date()),
            "validation_end_date": str(VAL_END_DATE.date()),
            "test_start_date": str(TEST_START_DATE.date()),
            "test_end_date": str(TEST_END_DATE.date()),
        },
        "source_hashes_preserved": source_hashes_after,
        "splits": report_splits,
        "generated_files": {
            "train_sequences": str(OUTPUT_DIR / "sequences_train.npz"),
            "validation_sequences": str(OUTPUT_DIR / "sequences_validation.npz"),
            "test_sequences": str(OUTPUT_DIR / "sequences_test.npz"),
            "sequence_index": str(index_path),
            "sequence_summary": str(summary_path),
        },
    }

    report_path = OUTPUT_DIR / "sequence_generation_multivariate_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {report_path.name}")

    print("\n" + "=" * 65)
    print("MULTIVARIATE SEQUENCE GENERATION COMPLETE")
    print("=" * 65)
    for s in ["train", "validation", "test"]:
        s_info = report_splits[s]
        print(f"--- {s.upper()} ---")
        print(f"  sequences: {s_info['sequence_count']:,}")
        print(f"  participating pairs: {s_info['participating_pair_count']}")
        print(f"  past_x shape: {s_info['past_x_shape']}")
        print(f"  future_x shape: {s_info['future_x_shape']}")
        print(f"  y shape: {s_info['y_shape']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
