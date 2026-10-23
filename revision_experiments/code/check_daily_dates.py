import os
import numpy as np
import pandas as pd

# Load test_split.csv
test_csv = pd.read_csv(r"Temporal Split(6)\test_split.csv")
print("test_split.csv date range:", test_csv["date"].min(), "to", test_csv["date"].max())
print("Unique stores in test_split:", test_csv["store_nbr"].nunique(), "items:", test_csv["item_nbr"].nunique())

# Load validation_split.csv
val_csv = pd.read_csv(r"Temporal Split(6)\validation_split.csv")
print("validation_split.csv date range:", val_csv["date"].min(), "to", val_csv["date"].max())
print("Unique stores in validation_split:", val_csv["store_nbr"].nunique(), "items:", val_csv["item_nbr"].nunique())

# Check sequence of dates
dates_test = sorted(test_csv["date"].unique())
print("Test dates (15 days):", dates_test)
dates_val = sorted(val_csv["date"].unique())
print("Validation dates (15 days):", dates_val)
