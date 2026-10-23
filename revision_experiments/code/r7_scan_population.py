"""
r7_scan_population.py
Scans full train.csv in chunks to compute average daily sales for eligible population
(>= 180 non-zero observations before cutoff 2017-07-31).
Reports population share per bin using the Section B quintile thresholds,
side by side with the 400-pair cohort and the 363 active pairs.
"""

import os
import time
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
TRAIN_PATH = os.path.join(BASE_DIR, "Corporación Favorita Grocery Sales Forecasting 2", "train.csv")
ITEMS_PATH = os.path.join(BASE_DIR, "Corporación Favorita Grocery Sales Forecasting 2", "items.csv")
TABLES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

CUTOFF = "2017-07-31"
MIN_OBS = 180
CHUNK_SIZE = 5_000_000

print(f"Scanning train.csv from {TRAIN_PATH} with cutoff <= {CUTOFF}...")
t0 = time.time()

usecols = ["date", "store_nbr", "item_nbr", "unit_sales"]
dtypes = {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}

chunk_aggregates = []
chunk_idx = 0

for chunk in pd.read_csv(TRAIN_PATH, usecols=usecols, dtype=dtypes, chunksize=CHUNK_SIZE):
    chunk_idx += 1
    # Filter before cutoff
    chunk = chunk[chunk["date"] <= CUTOFF]
    if chunk.empty:
        continue
    
    # Clip negative returns to 0
    chunk["unit_sales"] = chunk["unit_sales"].clip(lower=0.0)
    
    # Only non-zero sales count towards positive observation count if non-zero is required,
    # but in pipeline(1-5).py, any row in train.csv is an observation (unit_sales > 0 transactions)
    # Filter where unit_sales > 0:
    chunk_pos = chunk[chunk["unit_sales"] > 0]
    
    grp = chunk_pos.groupby(["store_nbr", "item_nbr"]).agg(
        obs_count=("unit_sales", "size"),
        tot_sales=("unit_sales", "sum")
    ).reset_index()
    
    chunk_aggregates.append(grp)
    print(f"Processed chunk {chunk_idx} ({len(chunk)} rows) in {time.time() - t0:.1f}s")

print("Merging chunk aggregates...")
combined = pd.concat(chunk_aggregates, ignore_index=True)
pop_df = combined.groupby(["store_nbr", "item_nbr"]).agg(
    observation_count=("obs_count", "sum"),
    total_sales=("tot_sales", "sum")
).reset_index()

pop_df["average_sales"] = pop_df["total_sales"] / pop_df["observation_count"]

# Merge item metadata to check perishability
items_df = pd.read_csv(ITEMS_PATH, usecols=["item_nbr", "perishable"])
pop_df = pop_df.merge(items_df, on="item_nbr", how="left")

# Save full candidate summary
cand_df = pop_df[pop_df["observation_count"] >= MIN_OBS].copy()
print(f"Total store-item pairs in train.csv: {len(pop_df)}")
print(f"Eligible population (obs >= {MIN_OBS}): {len(cand_df)} pairs")

cand_df.to_parquet(os.path.join(TABLES_DIR, "eligible_population_pairs.parquet"), index=False)
print(f"Completed in {time.time() - t0:.1f}s")
