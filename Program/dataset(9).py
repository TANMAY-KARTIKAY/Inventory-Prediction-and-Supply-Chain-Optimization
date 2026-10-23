from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_OUTPUT_DIR = Path(r"C:\CODE\favorita_inventory_pipeline\outputs")

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

DATE_METADATA_COLS: list[str] = [
    "input_start_date",
    "input_end_date",
    "target_start_date",
    "target_end_date",
]


def load_npz_mmap(npz_path: Path | str) -> dict[str, np.memmap]:
    """Memory-map uncompressed (.npz / ZIP_STORED) array files directly from disk.
    
    This avoids loading multi-gigabyte sequence archives into RAM.
    """
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


class FavoritaMultivariateDataset(Dataset):
    """Memory-efficient PyTorch Dataset for 28-day past historical + 7-day future known forecasting.
    
    Streams past_x [28, 15], future_x [7, 11], and y [7] directly from memory-mapped files.
    """

    def __init__(
        self,
        arrays: dict[str, Any],
        split_name: str = "train",
    ) -> None:
        self.split_name = split_name
        self.arrays = arrays
        self.past_x = arrays["past_x"]
        self.future_x = arrays["future_x"]
        self.y = arrays["y"]
        self.y_raw = arrays.get("y_raw", None)
        self.length = len(self.y)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "past_x": torch.from_numpy(np.array(self.past_x[idx], dtype=np.float32)),
            "future_x": torch.from_numpy(np.array(self.future_x[idx], dtype=np.float32)),
            "y": torch.from_numpy(np.array(self.y[idx], dtype=np.float32)),
            "sequence_idx": idx,
        }
        if self.y_raw is not None:
            item["y_raw"] = torch.from_numpy(np.array(self.y_raw[idx], dtype=np.float32))

        # Static pair metadata
        for col in STATIC_METADATA_COLS:
            if col in self.arrays:
                val = self.arrays[col][idx]
                if isinstance(val, (int, np.integer)):
                    item[col] = torch.tensor(int(val), dtype=torch.long)
                elif isinstance(val, (float, np.floating)):
                    item[col] = torch.tensor(float(val), dtype=torch.float32)
                else:
                    item[col] = str(val)

        # Date metadata
        for d_col in DATE_METADATA_COLS:
            if d_col in self.arrays:
                item[d_col] = str(self.arrays[d_col][idx])

        return item

    @classmethod
    def from_npz(
        cls,
        npz_path: Path | str,
        split_name: str = "train",
        use_mmap: bool = True,
    ) -> FavoritaMultivariateDataset:
        npz_path = Path(npz_path)
        if use_mmap:
            arrays = load_npz_mmap(npz_path)
        else:
            arrays = dict(np.load(npz_path))
        return cls(arrays, split_name=split_name)


def create_dataloaders(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[
    dict[str, DataLoader],
    dict[str, FavoritaMultivariateDataset],
]:
    """Build train, validation, and test PyTorch DataLoaders using memory-mapped sequence archives."""
    output_dir = Path(output_dir)

    split_files = {
        "train": output_dir / "sequences_train.npz",
        "validation": output_dir / "sequences_validation.npz",
        "test": output_dir / "sequences_test.npz",
    }

    datasets: dict[str, FavoritaMultivariateDataset] = {}
    dataloaders: dict[str, DataLoader] = {}

    for split_name, npz_path in split_files.items():
        if not npz_path.exists():
            raise FileNotFoundError(f"Sequence file not found: {npz_path}")

        dataset = FavoritaMultivariateDataset.from_npz(
            npz_path, split_name=split_name, use_mmap=True
        )
        datasets[split_name] = dataset

        # Shuffle exclusively for training; deterministic order for validation and test
        is_train = split_name == "train"
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        dataloaders[split_name] = loader

    return dataloaders, datasets


def verify_dataloaders(batch_size: int = 64) -> dict[str, Any]:
    """Rigorously verify dataset shapes, batch dimensions, dtypes, shuffling, and memory mapping."""
    import psutil

    process = psutil.Process()
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    dataloaders, datasets = create_dataloaders(batch_size=batch_size)

    rss_after_load_mb = process.memory_info().rss / (1024 * 1024)

    expected_lengths = {
        "train": 624451,
        "validation": 3299,
        "test": 3237,
    }

    report: dict[str, Any] = {
        "batch_size": batch_size,
        "memory_info": {
            "initial_rss_mb": round(rss_before_mb, 2),
            "after_datasets_loaded_rss_mb": round(rss_after_load_mb, 2),
            "rss_delta_mb": round(rss_after_load_mb - rss_before_mb, 2),
            "memory_mapping_verified": True,
        },
        "splits": {},
    }

    for split_name in ["train", "validation", "test"]:
        ds = datasets[split_name]
        loader = dataloaders[split_name]

        # 1. Verify Dataset length
        actual_len = len(ds)
        expected_len = expected_lengths[split_name]
        assert actual_len == expected_len, f"{split_name} length mismatch: {actual_len} != {expected_len}"

        # 2. Verify underlying arrays are actual np.memmap
        assert isinstance(ds.past_x, np.memmap), f"{split_name} past_x is not np.memmap!"
        assert isinstance(ds.future_x, np.memmap), f"{split_name} future_x is not np.memmap!"
        assert isinstance(ds.y, np.memmap), f"{split_name} y is not np.memmap!"

        # 3. Fetch one batch from DataLoader
        batch = next(iter(loader))

        past_x_batch = batch["past_x"]
        future_x_batch = batch["future_x"]
        y_batch = batch["y"]
        seq_idx_batch = batch["sequence_idx"]

        # Verify shapes (18 past features, 14 future-known features)
        assert past_x_batch.shape == (batch_size, 28, 18), f"Unexpected past_x shape: {past_x_batch.shape}"
        assert future_x_batch.shape == (batch_size, 7, 14), f"Unexpected future_x shape: {future_x_batch.shape}"
        assert y_batch.shape == (batch_size, 7), f"Unexpected y shape: {y_batch.shape}"

        # Verify dtypes
        assert past_x_batch.dtype == torch.float32, f"past_x dtype {past_x_batch.dtype} != float32"
        assert future_x_batch.dtype == torch.float32, f"future_x dtype {future_x_batch.dtype} != float32"
        assert y_batch.dtype == torch.float32, f"y dtype {y_batch.dtype} != float32"

        # Verify shuffling behavior
        first_64_indices = seq_idx_batch.tolist()
        expected_sequential = list(range(batch_size))
        is_shuffled = first_64_indices != expected_sequential

        if split_name == "train":
            assert is_shuffled, "Train DataLoader batch is NOT shuffled!"
        else:
            assert not is_shuffled, f"{split_name} DataLoader batch was unexpectedly shuffled!"

        report["splits"][split_name] = {
            "dataset_length": actual_len,
            "total_batches": len(loader),
            "batch_past_x_shape": list(past_x_batch.shape),
            "batch_past_x_dtype": str(past_x_batch.dtype),
            "batch_future_x_shape": list(future_x_batch.shape),
            "batch_future_x_dtype": str(future_x_batch.dtype),
            "batch_y_shape": list(y_batch.shape),
            "batch_y_dtype": str(y_batch.dtype),
            "is_shuffled": is_shuffled,
            "first_5_batch_indices": first_64_indices[:5],
            "sample_store_nbrs": batch["store_nbr"][:5].tolist(),
            "sample_item_nbrs": batch["item_nbr"][:5].tolist(),
        }

    rss_after_batches_mb = process.memory_info().rss / (1024 * 1024)
    report["memory_info"]["after_fetching_batches_rss_mb"] = round(rss_after_batches_mb, 2)

    return report


def main() -> None:
    print("=" * 65)
    print("STEP 10: PYTORCH DATASET AND DATALOADER VERIFICATION")
    print("=" * 65)

    results = verify_dataloaders(batch_size=64)

    print(f"\nMemory-Mapping Status: {results['memory_info']['memory_mapping_verified']}")
    print(f"Initial Process RSS:   {results['memory_info']['initial_rss_mb']} MB")
    print(f"After Datasets Loaded: {results['memory_info']['after_datasets_loaded_rss_mb']} MB (Delta: {results['memory_info']['rss_delta_mb']} MB)")
    print(f"After Fetching Batches: {results['memory_info']['after_fetching_batches_rss_mb']} MB")

    for split_name in ["train", "validation", "test"]:
        s_info = results["splits"][split_name]
        print(f"\n--- {split_name.upper()} ---")
        print(f"  Dataset length:         {s_info['dataset_length']:,} sequences")
        print(f"  Total batches (B=64):   {s_info['total_batches']:,}")
        print(f"  Batch past_x:           {s_info['batch_past_x_shape']} ({s_info['batch_past_x_dtype']})")
        print(f"  Batch future_x:         {s_info['batch_future_x_shape']} ({s_info['batch_future_x_dtype']})")
        print(f"  Batch y:                {s_info['batch_y_shape']} ({s_info['batch_y_dtype']})")
        print(f"  Shuffled:               {s_info['is_shuffled']}")
        print(f"  First 5 batch indices:  {s_info['first_5_batch_indices']}")

    print("\n" + "=" * 65)
    print("ALL STEP 10 VERIFICATIONS PASSED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
