"""
save_sim_inputs.py
Precomputes and serializes all preprocessed matrices into simulation_inputs.npz
so that E1, E2, E3, E4, etc. load instant NumPy arrays without reloading 600k CSV rows.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
TABLES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

print("Precomputing simulation inputs...")

UNIT_HOLDING_DRY     = 0.10
UNIT_HOLDING_PERISH  = 0.15
LEAD_TIME_DAYS       = 7
OFFSET_STD_VAL       = 0.1785
OFFSET_HVOL_VAL      = 4.5766

# Active 363 pairs
test_npz = np.load(os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "sequences_test.npz"), mmap_mode="r")
active_test_pairs = sorted(list(set(zip(test_npz["store_nbr"], test_npz["item_nbr"]))))
N = len(active_test_pairs)

pairs_df = pd.read_csv(os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "selected_pairs.csv"))
perish_map = { (int(r["store_nbr"]), int(r["item_nbr"])): bool(r["perishable"]) for _, r in pairs_df.iterrows() }
hvol_map   = { (int(r["store_nbr"]), int(r["item_nbr"])): bool(r["average_sales"] >= 100.0) for _, r in pairs_df.iterrows() }

h_cost_vec = np.array([UNIT_HOLDING_PERISH if perish_map[p] else UNIT_HOLDING_DRY for p in active_test_pairs], dtype=np.float64)
is_hvol_vec = np.array([hvol_map[p] for p in active_test_pairs], dtype=bool)

# Trailing demand for validation (2017-06-19 to 2017-07-16)
df_train = pd.read_csv(os.path.join(BASE_DIR, "Temporal Split(6)", "train_split.csv"), usecols=["date", "store_nbr", "item_nbr", "unit_sales"])
df_v_trail = df_train[df_train["date"] >= "2017-06-19"]

dbar_val = {}
sigma_val = {}
for (s, i), grp in df_v_trail.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s), int(i))
    if pair in perish_map:
        sales = grp["unit_sales"].values
        dbar_val[pair] = max(1.0, float(np.mean(sales)))
        sigma_val[pair] = float(np.std(sales) * np.sqrt(LEAD_TIME_DAYS) + 1.0)

val_dbar_vec  = np.array([dbar_val.get(p, 1.0) for p in active_test_pairs], dtype=np.float64)
val_sigma_vec = np.array([sigma_val.get(p, 5.0) for p in active_test_pairs], dtype=np.float64)

# Trailing demand for test (2017-07-04 to 2017-07-31)
df_val = pd.read_csv(os.path.join(BASE_DIR, "Temporal Split(6)", "validation_split.csv"), usecols=["date", "store_nbr", "item_nbr", "unit_sales"])
df_t_trail = pd.concat([df_train[df_train["date"] >= "2017-07-04"], df_val], ignore_index=True)

dbar_test = {}
sigma_test = {}
for (s, i), grp in df_t_trail.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s), int(i))
    if pair in perish_map:
        sales = grp["unit_sales"].values
        dbar_test[pair] = max(1.0, float(np.mean(sales)))
        sigma_test[pair] = float(np.std(sales) * np.sqrt(LEAD_TIME_DAYS) + 1.0)

test_dbar_vec  = np.array([dbar_test.get(p, 1.0) for p in active_test_pairs], dtype=np.float64)
test_sigma_vec = np.array([sigma_test.get(p, 5.0) for p in active_test_pairs], dtype=np.float64)

# Load validation forecasts
df_val_q = pd.read_csv(os.path.join(TABLES_DIR, "validation_forecasts_quantile.csv"))
df_val_h = pd.read_csv(os.path.join(TABLES_DIR, "validation_forecasts_huber.csv"))

val_act_dict, val_p50_dict, val_p90_dict, val_p90_raw_dict = {}, {}, {}, {}
for (s, i), grp in df_val_q.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s), int(i))
    if pair in perish_map:
        d1 = grp[grp["forecast_day"] == 1].sort_values("target_start_date")
        act = list(d1["actual_sales"].values)
        p50 = list(d1["p50_forecast"].values)
        p90_raw = list(d1["p90_forecast"].values)
        off = OFFSET_HVOL_VAL if hvol_map[pair] else OFFSET_STD_VAL
        p90_cal = [x + off for x in p90_raw]
        if len(act) < 15:
            act += [act[-1]] * (15 - len(act))
            p50 += [p50[-1]] * (15 - len(p50))
            p90_raw += [p90_raw[-1]] * (15 - len(p90_raw))
            p90_cal += [p90_cal[-1]] * (15 - len(p90_cal))
        val_act_dict[pair] = act[:15]
        val_p50_dict[pair] = p50[:15]
        val_p90_dict[pair] = p90_cal[:15]
        val_p90_raw_dict[pair] = p90_raw[:15]

val_demand_mat  = np.array([val_act_dict.get(p, [dbar_val.get(p, 1.0)]*15) for p in active_test_pairs], dtype=np.float64)
val_p50_mat     = np.array([val_p50_dict.get(p, [dbar_val.get(p, 1.0)]*15) for p in active_test_pairs], dtype=np.float64)
val_p90_mat     = np.array([val_p90_dict.get(p, [dbar_val.get(p, 1.0)*1.5]*15) for p in active_test_pairs], dtype=np.float64)
val_p90_raw_mat = np.array([val_p90_raw_dict.get(p, [dbar_val.get(p, 1.0)*1.5]*15) for p in active_test_pairs], dtype=np.float64)

val_huber_dict = {}
for (s, i), grp in df_val_h.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s), int(i))
    if pair in perish_map:
        grp = grp.sort_values("target_start_date")
        fc = [r["pred_day_1"] for _, r in grp.iterrows()]
        if len(fc) < 15:
            last_r = grp.iloc[-1]
            for st in range(2, 2 + (15 - len(fc))):
                fc.append(last_r[f"pred_day_{min(st, 7)}"])
        val_huber_dict[pair] = fc[:15]

val_huber_mat = np.array([val_huber_dict.get(p, [dbar_val.get(p, 1.0)]*15) for p in active_test_pairs], dtype=np.float64)

# Compute validation residual std for Huber
sig_res_list = []
for idx in range(N):
    res = val_demand_mat[idx] - val_huber_mat[idx]
    sig_res_list.append(float(np.std(res) * np.sqrt(LEAD_TIME_DAYS) + 1.0))
val_sig_res_vec = np.array(sig_res_list, dtype=np.float64)

# Load test forecasts
df_test_q = pd.read_csv(os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "tft_quantile_forecasts.csv"))
df_test_q["store_nbr"] = df_test_q["sequence_idx"].map(lambda idx: test_npz["store_nbr"][idx])
df_test_q["item_nbr"]  = df_test_q["sequence_idx"].map(lambda idx: test_npz["item_nbr"][idx])

test_act_dict, test_p50_dict, test_p90_dict, test_p90_raw_dict = {}, {}, {}, {}
for (s, i), grp in df_test_q.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s), int(i))
    if pair in perish_map:
        d1 = grp[grp["forecast_day"] == 1]
        act = list(d1["actual_sales"].values)
        p50 = list(d1["p50_forecast"].values)
        p90_raw = list(d1["p90_forecast"].values)
        off = OFFSET_HVOL_VAL if hvol_map[pair] else OFFSET_STD_VAL
        p90_cal = [x + off for x in p90_raw]
        if len(act) < 15:
            act += [act[-1]] * (15 - len(act))
            p50 += [p50[-1]] * (15 - len(p50))
            p90_raw += [p90_raw[-1]] * (15 - len(p90_raw))
            p90_cal += [p90_cal[-1]] * (15 - len(p90_cal))
        test_act_dict[pair] = act[:15]
        test_p50_dict[pair] = p50[:15]
        test_p90_dict[pair] = p90_cal[:15]
        test_p90_raw_dict[pair] = p90_raw[:15]

test_demand_mat  = np.array([test_act_dict[p] for p in active_test_pairs], dtype=np.float64)
test_p50_mat     = np.array([test_p50_dict[p] for p in active_test_pairs], dtype=np.float64)
test_p90_mat     = np.array([test_p90_dict[p] for p in active_test_pairs], dtype=np.float64)
test_p90_raw_mat = np.array([test_p90_raw_dict[p] for p in active_test_pairs], dtype=np.float64)

df_test_h = pd.read_csv(os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "tft_test_forecasts.csv"))
test_huber_dict = {}
for (s, i), grp in df_test_h.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s), int(i))
    if pair in perish_map:
        fc = [r["pred_day_1"] for _, r in grp.iterrows()]
        if len(fc) < 15:
            last_r = grp.iloc[-1]
            for st in range(2, 2 + (15 - len(fc))):
                fc.append(last_r[f"pred_day_{min(st, 7)}"])
        test_huber_dict[pair] = fc[:15]

test_huber_mat = np.array([test_huber_dict[p] for p in active_test_pairs], dtype=np.float64)

# Save to .npz file
out_npz_path = os.path.join(TABLES_DIR, "simulation_inputs.npz")
np.savez_compressed(
    out_npz_path,
    val_demand_mat=val_demand_mat,
    val_p50_mat=val_p50_mat,
    val_p90_mat=val_p90_mat,
    val_p90_raw_mat=val_p90_raw_mat,
    val_huber_mat=val_huber_mat,
    val_dbar_vec=val_dbar_vec,
    val_sigma_vec=val_sigma_vec,
    val_sig_res_vec=val_sig_res_vec,
    test_demand_mat=test_demand_mat,
    test_p50_mat=test_p50_mat,
    test_p90_mat=test_p90_mat,
    test_p90_raw_mat=test_p90_raw_mat,
    test_huber_mat=test_huber_mat,
    test_dbar_vec=test_dbar_vec,
    test_sigma_vec=test_sigma_vec,
    h_cost_vec=h_cost_vec,
    is_hvol_vec=is_hvol_vec,
)

print(f"Successfully saved all simulation matrices to {out_npz_path}!")
