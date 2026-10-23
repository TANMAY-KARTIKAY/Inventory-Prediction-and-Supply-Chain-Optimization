"""
audit_excluded_pairs.py
Amendment F: Detailed audit of the 37 excluded store-item pairs from the original 400.
Outputs detailed breakdown to revision_experiments/outputs/tables/excluded_pairs_audit.csv
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
pairs_path = os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "selected_pairs.csv")
df_pairs = pd.read_csv(pairs_path)

test_npz_path = os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "sequences_test.npz")
test_npz = np.load(test_npz_path, mmap_mode="r")

val_npz_path = os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "sequences_validation.npz")
val_npz = np.load(val_npz_path, mmap_mode="r")

train_npz_path = os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "sequences_train.npz")
train_npz = np.load(train_npz_path, mmap_mode="r")

# Pairs in test sequences
test_pairs = set(zip(test_npz["store_nbr"], test_npz["item_nbr"]))
val_pairs  = set(zip(val_npz["store_nbr"], val_npz["item_nbr"]))
train_pairs = set(zip(train_npz["store_nbr"], train_npz["item_nbr"]))
all_400_pairs = set(zip(df_pairs["store_nbr"], df_pairs["item_nbr"]))

excluded_pairs = all_400_pairs - test_pairs
print(f"Total pairs in selected_pairs.csv: {len(all_400_pairs)}")
print(f"Pairs present in sequences_test.npz: {len(test_pairs)}")
print(f"Pairs present in sequences_validation.npz: {len(val_pairs)}")
print(f"Pairs present in sequences_train.npz: {len(train_pairs)}")
print(f"Excluded from test simulation: {len(excluded_pairs)}")

# Audit each excluded pair
records = []
for s_nbr, i_nbr in sorted(excluded_pairs):
    pair_row = df_pairs[(df_pairs["store_nbr"] == s_nbr) & (df_pairs["item_nbr"] == i_nbr)].iloc[0]
    in_val = (s_nbr, i_nbr) in val_pairs
    in_train = (s_nbr, i_nbr) in train_pairs
    
    # Reason
    if not in_val:
        reason = "Discontinued before Validation Window (sales ended before 2017-07-17)"
    else:
        reason = "Discontinued before Test Window (sales ended between 2017-07-17 and 2017-08-01)"
    
    records.append({
        "store_nbr": s_nbr,
        "item_nbr": i_nbr,
        "family": pair_row.get("family", "UNKNOWN"),
        "class": pair_row.get("class", "UNKNOWN"),
        "perishable": pair_row.get("perishable", "UNKNOWN"),
        "average_sales": round(float(pair_row.get("average_sales", 0.0)), 2),
        "present_in_train": in_train,
        "present_in_validation": in_val,
        "present_in_test": False,
        "exclusion_reason": reason
    })

df_exc = pd.DataFrame(records)
out_csv = os.path.join(BASE_DIR, "revision_experiments", "outputs", "tables", "excluded_pairs_audit.csv")
df_exc.to_csv(out_csv, index=False)
print(f"Saved excluded pairs audit to: {out_csv}")
print(df_exc.groupby("exclusion_reason").size())
