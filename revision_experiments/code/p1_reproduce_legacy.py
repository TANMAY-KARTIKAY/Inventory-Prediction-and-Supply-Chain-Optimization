"""
p1_reproduce_legacy.py
Phase 1: Exact Reproduction of Legacy Simulation Benchmark (Test Window: Aug 1-15, 2017)
Label: "legacy (contains look-ahead if confirmed)"

This script reproduces the exact results from MHQF optm.ipynb (Cell 1, Cell 2, Cell 5)
and compares against Multi-Horizon Quantile Forecasting(process)/supply_chain_optimization_results.csv.

Policies evaluated:
1. Mean-demand Gaussian (alpha=1.0, beta=4.0) [Legacy name: Classical Gaussian]
2. Huber TFT + Gaussian (alpha=1.0, beta=4.0)
3. Quantile Default (alpha=1.0, beta=4.0)
4. Quantile TFT + Conformal PSO (alpha=0.1000, beta=3.4879)
5. Quantile TFT + Conformal GWO (alpha=0.1010, beta=3.4890)
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Hard rule 6: Fix seeds, record library versions, log runtime
np.random.seed(42)
t_start = time.time()

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)\Multi-Horizon Quantile Forecasting(process)"
OUTPUT_TABLE = r"C:\Users\tanma\OneDrive\Desktop\project(1)\revision_experiments\outputs\tables\reproduction_legacy_benchmark.csv"
LEGACY_REFERENCE_FILE = os.path.join(BASE_DIR, "supply_chain_optimization_results.csv")

print("=" * 80)
print("PHASE 1: EXACT REPRODUCTION OF LEGACY BENCHMARK SIMULATION")
print(f"Python Version: {sys.version}")
print(f"NumPy Version:  {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"Random Seed:    42")
print("=" * 80)

# 1. Economic Cost Parameters (Identical to MHQF optm.ipynb Cell 2)
UNIT_HOLDING_DRY     = 0.10   # $ / unit / day (Shelf-stable dry goods)
UNIT_HOLDING_PERISH  = 0.15   # $ / unit / day (Chilled / perishable grocery)
STOCKOUT_PENALTY     = 2.50   # $ / unit of unmet customer demand (lost sales)
FIXED_ORDER_COST     = 8.00   # $ / purchase order placed (truck dispatch)
LEAD_TIME_DAYS       = 7      # Supplier delivery lead time (7 days)

# 2. Ingest Forecasts and Metadata
df_q = pd.read_csv(os.path.join(BASE_DIR, "tft_quantile_forecasts.csv"))
test_npz = np.load(os.path.join(BASE_DIR, "sequences_test.npz"), mmap_mode="r")
df_pairs = pd.read_csv(os.path.join(BASE_DIR, "selected_pairs.csv"))

seq_store = test_npz["store_nbr"]
seq_item  = test_npz["item_nbr"]
seq_start = test_npz["target_start_date"]

df_q["store_nbr"] = df_q["sequence_idx"].map(lambda idx: seq_store[idx])
df_q["item_nbr"]  = df_q["sequence_idx"].map(lambda idx: seq_item[idx])
df_q["target_start_date"] = df_q["sequence_idx"].map(lambda idx: seq_start[idx])

# Identify High-Volume Tier (average_sales >= 100)
high_vol_pairs = set(tuple(x) for x in df_pairs[df_pairs["average_sales"] >= 100][["store_nbr", "item_nbr"]].values)
df_q["is_high_vol"] = df_q.apply(lambda r: (r["store_nbr"], r["item_nbr"]) in high_vol_pairs, axis=1)

# Apply hardcoded conformal offsets from legacy notebook
OFFSET_HIGH = 4.51
OFFSET_STD  = 0.18
df_q["p90_calibrated"] = df_q["p90_forecast"] + np.where(df_q["is_high_vol"], OFFSET_HIGH, OFFSET_STD)

# Build 15-day continuous series per store-item pair (MHQF optm.ipynb Cell 2)
pair_daily_actual  = {}
pair_daily_p50     = {}
pair_daily_p90     = {}
pair_is_perishable = {}

for (s_nbr, i_nbr), group in df_q.groupby(["store_nbr", "item_nbr"]):
    day1_group = group[group["forecast_day"] == 1].sort_values("target_start_date")
    daily_act = list(day1_group["actual_sales"].values)
    daily_p50 = list(day1_group["p50_forecast"].values)
    daily_p90 = list(day1_group["p90_calibrated"].values)
    
    if len(daily_act) < 15:
        last_start = group["target_start_date"].max()
        last_window = group[group["target_start_date"] == last_start].set_index("forecast_day")
        needed = 15 - len(daily_act)
        for step in range(2, 2 + needed):
            day_key = min(step, 7)
            if day_key in last_window.index:
                row = last_window.loc[day_key]
                daily_act.append(row["actual_sales"])
                daily_p50.append(row["p50_forecast"])
                daily_p90.append(row["p90_calibrated"])
            else:
                daily_act.append(daily_act[-1])
                daily_p50.append(daily_p50[-1])
                daily_p90.append(daily_p90[-1])

    pair_daily_actual[(s_nbr, i_nbr)] = np.array(daily_act[:15], dtype=np.float64)
    pair_daily_p50[(s_nbr, i_nbr)]    = np.array(daily_p50[:15], dtype=np.float64)
    pair_daily_p90[(s_nbr, i_nbr)]    = np.array(daily_p90[:15], dtype=np.float64)

    is_p = df_pairs[(df_pairs["store_nbr"] == s_nbr) & (df_pairs["item_nbr"] == i_nbr)]["perishable"].values
    pair_is_perishable[(s_nbr, i_nbr)] = bool(is_p[0]) if len(is_p) > 0 else False

print(f"Constructed 15-day continuous series for {len(pair_daily_actual)} active store-item pairs.")

# Ingest Huber baseline forecasts
df_huber = pd.read_csv(os.path.join(BASE_DIR, "tft_test_forecasts.csv"))
pair_huber_daily = {}
for (s_nbr, i_nbr), group in df_huber.groupby(["store_nbr", "item_nbr"]):
    group = group.sort_values("target_start_date")
    fc = [r["pred_day_1"] for _, r in group.iterrows()]
    if len(fc) < 15:
        last = group.iloc[-1]
        for step in range(2, 2 + (15 - len(fc))):
            fc.append(last[f"pred_day_{min(step, 7)}"])
    pair_huber_daily[(s_nbr, i_nbr)] = np.array(fc[:15], dtype=np.float64)

# 3. Simulator Function (Cell 5 implementation)
def run_benchmark_policy(policy_name, alpha_val, beta_val):
    tot_hold = 0.0
    tot_stockout = 0.0
    tot_order = 0.0
    tot_demand = 0.0
    tot_sales = 0.0
    daily_spend = np.zeros(15)

    for (s_nbr, i_nbr), demand in pair_daily_actual.items():
        is_perish = pair_is_perishable[(s_nbr, i_nbr)]
        h_cost = UNIT_HOLDING_PERISH if is_perish else UNIT_HOLDING_DRY
        
        # LOOKAHEAD LEAKAGE POINT: mean_d computed from 15-day actual test demand
        mean_d = max(1.0, np.mean(demand))

        if policy_name in ["Classical Gaussian", "Mean-demand Gaussian"]:
            lead_exp = mean_d * LEAD_TIME_DAYS
            # LOOKAHEAD LEAKAGE: np.std(demand) computed over 15-day test sales
            ss = alpha_val * 1.282 * (np.std(demand) * np.sqrt(LEAD_TIME_DAYS) + 1.0)
        elif policy_name == "Huber TFT + Gaussian":
            h_fc = pair_huber_daily.get((s_nbr, i_nbr), np.full(15, mean_d))
            lead_exp = np.sum(h_fc[:LEAD_TIME_DAYS])
            # LOOKAHEAD LEAKAGE: np.std(demand - h_fc) computed over 15-day test sales
            ss = alpha_val * 1.282 * (np.std(demand - h_fc) * np.sqrt(LEAD_TIME_DAYS) + 1.0)
        elif policy_name == "Quantile Default":
            p50 = pair_daily_p50[(s_nbr, i_nbr)]
            p90 = pair_daily_p90[(s_nbr, i_nbr)]
            lead_exp = np.sum(p50[:LEAD_TIME_DAYS])
            ss = 1.0 * np.sum(np.maximum(0.0, p90[:LEAD_TIME_DAYS] - p50[:LEAD_TIME_DAYS]))
        elif policy_name in ["Quantile TFT + Conformal PSO", "Quantile TFT + Conformal GWO"]:
            p50 = pair_daily_p50[(s_nbr, i_nbr)]
            p90 = pair_daily_p90[(s_nbr, i_nbr)]
            lead_exp = np.sum(p50[:LEAD_TIME_DAYS])
            ss = alpha_val * np.sum(np.maximum(0.0, p90[:LEAD_TIME_DAYS] - p50[:LEAD_TIME_DAYS]))
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        s = lead_exp + ss
        S = s + (beta_val * mean_d)
        on_hand = S * 0.70
        pipeline = [0.0] * LEAD_TIME_DAYS

        for t in range(15):
            d_t = demand[t]
            tot_demand += d_t
            arrivals = pipeline.pop(0)
            on_hand += arrivals

            sales = min(on_hand, d_t)
            stockout = max(0.0, d_t - on_hand)
            on_hand = max(0.0, on_hand - d_t)

            tot_sales += sales
            c_so = stockout * STOCKOUT_PENALTY
            c_hd = on_hand * h_cost
            tot_stockout += c_so
            tot_hold += c_hd

            c_ord = 0.0
            inv_pos = on_hand + sum(pipeline)
            if inv_pos <= s:
                qty = max(0.0, S - inv_pos)
                pipeline.append(qty)
                c_ord = FIXED_ORDER_COST
                tot_order += c_ord
            else:
                pipeline.append(0.0)

            daily_spend[t] += (c_hd + c_so + c_ord)

    sl = (tot_sales / tot_demand) * 100.0 if tot_demand > 0 else 100.0
    tot_cost = tot_hold + tot_stockout + tot_order
    return {
        "Policy": policy_name,
        "alpha": alpha_val,
        "beta": beta_val,
        "Total Cost ($)": tot_cost,
        "Holding Cost ($)": tot_hold,
        "Stockout Cost ($)": tot_stockout,
        "Ordering Cost ($)": tot_order,
        "Fill Rate (%)": sl,
        "Evaluation Mode": "legacy (contains look-ahead if confirmed)",
        "Daily Spend": daily_spend
    }

# 4. Run All Policies
policies = [
    ("Mean-demand Gaussian", 1.0, 4.0),
    ("Huber TFT + Gaussian", 1.0, 4.0),
    ("Quantile Default", 1.0, 4.0),
    ("Quantile TFT + Conformal PSO", 0.1000, 3.4879),
    ("Quantile TFT + Conformal GWO", 0.1010, 3.4890),
]

results = []
for p_name, a_val, b_val in policies:
    res = run_benchmark_policy(p_name, a_val, b_val)
    results.append(res)

df_reproduced = pd.DataFrame(results)

# Compare with Huber TFT + Gaussian
base_cost = df_reproduced.loc[df_reproduced["Policy"] == "Huber TFT + Gaussian", "Total Cost ($)"].values[0]
df_reproduced["Savings vs Huber ($)"] = base_cost - df_reproduced["Total Cost ($)"]
df_reproduced["Savings (%)"] = (df_reproduced["Savings vs Huber ($)"] / base_cost) * 100.0

# 5. Load and Verify against Legacy Reference
print("\n" + "=" * 80)
print("VERIFICATION AGAINST ORIGINAL REFERENCE LEDGER:")
print(f"Loading reference: {LEGACY_REFERENCE_FILE}")
df_ref = pd.read_csv(LEGACY_REFERENCE_FILE)
print(df_ref[["Policy", "Total Cost ($)", "Fill Rate (%)"]])

print("\n" + "=" * 80)
print("REPRODUCED RESULTS:")
print(df_reproduced[["Policy", "alpha", "beta", "Total Cost ($)", "Holding Cost ($)", "Stockout Cost ($)", "Ordering Cost ($)", "Fill Rate (%)", "Savings (%)"]])

# Assertions
for idx, row in df_reproduced.iterrows():
    p_name = row["Policy"]
    ref_name = "Classical Gaussian" if p_name == "Mean-demand Gaussian" else p_name
    match = df_ref[df_ref["Policy"] == ref_name]
    if len(match) > 0:
        ref_row = match.iloc[0]
        cost_diff = abs(row["Total Cost ($)"] - ref_row["Total Cost ($)"])
        cost_pct_diff = (cost_diff / ref_row["Total Cost ($)"]) * 100.0
        sl_diff = abs(row["Fill Rate (%)"] - ref_row["Fill Rate (%)"])
        
        print(f"Policy: {p_name:<30} | Diff Cost: ${cost_diff:,.4f} ({cost_pct_diff:.4f}%) | Diff SL: {sl_diff:.4f}%")
        assert cost_pct_diff <= 0.1, f"Cost difference {cost_pct_diff}% exceeds 0.1% threshold for {p_name}!"
        assert sl_diff <= 0.05, f"Fill rate difference {sl_diff}% exceeds 0.05% threshold for {p_name}!"
    else:
        print(f"Policy: {p_name:<30} | Evaluated (not in 4-policy reference CSV)")

# 6. Save reproduction table
os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)
export_cols = ["Policy", "alpha", "beta", "Total Cost ($)", "Holding Cost ($)", "Stockout Cost ($)", "Ordering Cost ($)", "Fill Rate (%)", "Savings vs Huber ($)", "Savings (%)", "Evaluation Mode"]
df_reproduced[export_cols].to_csv(OUTPUT_TABLE, index=False)
print(f"\n[OK] Successfully saved reproduction table to: {OUTPUT_TABLE}")

elapsed = time.time() - t_start
print(f"Reproduction completed successfully in {elapsed:.2f} seconds.")
print("=" * 80)
