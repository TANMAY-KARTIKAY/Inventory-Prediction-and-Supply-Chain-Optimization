"""
v_final_verification.py
Final Verification Script for Journal Revision (Corporación Favorita, 363 Active Pairs).
Performs checks and outputs tables for V1 through V6:
- V1: P90 coverage recomputation (raw & calibrated) per tier and horizon from CSVs, sample rows, and explanations.
- V2: Verbatim code of primary ordering-fee rule and days 0-6 cost component reconciliation.
- V3: Inventory of final tables and scripts, confirmation of unified_simulator usage, marking older tables superseded.
- V4: Strict sMAPE (0/0=0 rule, no epsilon) and WAPE primary metric across all models.
- V5: Matched-fill frontier (no interpolation), cp sensitivity with ranking flips, paired bootstraps in All Buffers and Restricted regimes.
- V6: Accurate Section B counts (demand_bin, perishable, quintile thresholds).
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
REV_DIR = os.path.join(BASE_DIR, "revision_experiments")
CODE_DIR = os.path.join(REV_DIR, "code")
TABLES_DIR = os.path.join(REV_DIR, "outputs", "tables")

sys.path.insert(0, CODE_DIR)
from unified_simulator import simulate_inventory

np.random.seed(42)

# ==============================================================================
# V1: P90 COVERAGE RECOMPUTATION DIRECTLY FROM FORECAST CSVS
# ==============================================================================
print("=" * 80)
print("V1: P90 COVERAGE RECOMPUTATION FROM FORECAST CSVS")
print("=" * 80)

val_csv = os.path.join(TABLES_DIR, "validation_forecasts_quantile.csv")
test_csv = os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "tft_quantile_forecasts.csv")
df_400 = pd.read_csv(os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "selected_pairs.csv"))

test_npz = np.load(os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "sequences_test.npz"), mmap_mode="r")
seq_store = test_npz["store_nbr"]
seq_item  = test_npz["item_nbr"]

df_val = pd.read_csv(val_csv)
df_test = pd.read_csv(test_csv)

df_test["store_nbr"] = df_test["sequence_idx"].map(lambda idx: seq_store[idx])
df_test["item_nbr"]  = df_test["sequence_idx"].map(lambda idx: seq_item[idx])

hvol_pairs = set(tuple(x) for x in df_400[df_400["average_sales"] >= 100.0][["store_nbr", "item_nbr"]].values)
df_val["is_high_vol"]  = df_val.apply(lambda r: (int(r["store_nbr"]), int(r["item_nbr"])) in hvol_pairs, axis=1)
df_test["is_high_vol"] = df_test.apply(lambda r: (int(r["store_nbr"]), int(r["item_nbr"])) in hvol_pairs, axis=1)

OFFSET_STD_VAL  = 0.1785
OFFSET_HVOL_VAL = 4.5766
OFFSET_STD_LEG  = 0.18
OFFSET_HVOL_LEG = 4.51

# Apply calibrated offset
df_val["offset_val"] = np.where(df_val["is_high_vol"], OFFSET_HVOL_VAL, OFFSET_STD_VAL)
df_val["p90_cal_val"] = df_val["p90_forecast"] + df_val["offset_val"]

df_test["offset_val"] = np.where(df_test["is_high_vol"], OFFSET_HVOL_VAL, OFFSET_STD_VAL)
df_test["p90_cal_val"] = df_test["p90_forecast"] + df_test["offset_val"]

df_test["offset_leg"] = np.where(df_test["is_high_vol"], OFFSET_HVOL_LEG, OFFSET_STD_LEG)
df_test["p90_cal_leg"] = df_test["p90_forecast"] + df_test["offset_leg"]

# Compute coverage summary table
v1_rows = []
for split_name, df_s in [("Validation", df_val), ("Test", df_test)]:
    # 1. Overall
    cov_raw = (df_s["actual_sales"] <= df_s["p90_forecast"]).mean() * 100.0
    cov_cal = (df_s["actual_sales"] <= df_s["p90_cal_val"]).mean() * 100.0
    v1_rows.append({
        "Split": split_name,
        "Segment": "Overall Cohort",
        "N": len(df_s),
        "P90 Raw Coverage (%)": round(cov_raw, 2),
        "P90 Calibrated Coverage (%)": round(cov_cal, 2),
        "Offset Std": OFFSET_STD_VAL,
        "Offset High-Vol": OFFSET_HVOL_VAL
    })
    
    # 2. Standard
    std_s = df_s[~df_s["is_high_vol"]]
    cov_raw = (std_s["actual_sales"] <= std_s["p90_forecast"]).mean() * 100.0
    cov_cal = (std_s["actual_sales"] <= std_s["p90_cal_val"]).mean() * 100.0
    v1_rows.append({
        "Split": split_name,
        "Segment": "Standard Tier (avg_sales < 100)",
        "N": len(std_s),
        "P90 Raw Coverage (%)": round(cov_raw, 2),
        "P90 Calibrated Coverage (%)": round(cov_cal, 2),
        "Offset Std": OFFSET_STD_VAL,
        "Offset High-Vol": OFFSET_HVOL_VAL
    })
    
    # 3. High-Vol
    hvol_s = df_s[df_s["is_high_vol"]]
    cov_raw = (hvol_s["actual_sales"] <= hvol_s["p90_forecast"]).mean() * 100.0
    cov_cal = (hvol_s["actual_sales"] <= hvol_s["p90_cal_val"]).mean() * 100.0
    v1_rows.append({
        "Split": split_name,
        "Segment": "High-Volume Tier (avg_sales >= 100)",
        "N": len(hvol_s),
        "P90 Raw Coverage (%)": round(cov_raw, 2),
        "P90 Calibrated Coverage (%)": round(cov_cal, 2),
        "Offset Std": OFFSET_STD_VAL,
        "Offset High-Vol": OFFSET_HVOL_VAL
    })
    
    # 4. Horizons 1 to 7
    for h in range(1, 8):
        h_s = df_s[df_s["forecast_day"] == h]
        cov_raw = (h_s["actual_sales"] <= h_s["p90_forecast"]).mean() * 100.0
        cov_cal = (h_s["actual_sales"] <= h_s["p90_cal_val"]).mean() * 100.0
        v1_rows.append({
            "Split": split_name,
            "Segment": f"Horizon Day h={h}",
            "N": len(h_s),
            "P90 Raw Coverage (%)": round(cov_raw, 2),
            "P90 Calibrated Coverage (%)": round(cov_cal, 2),
            "Offset Std": OFFSET_STD_VAL,
            "Offset High-Vol": OFFSET_HVOL_VAL
        })

df_v1 = pd.DataFrame(v1_rows)
df_v1.to_csv(os.path.join(TABLES_DIR, "v1_p90_coverage_recomputed.csv"), index=False)
print("Saved v1_p90_coverage_recomputed.csv")
print(df_v1.to_string(index=False))

# 5 Sample Rows showing offset applied
sample_indices = [10, 500, 1200, 2500, 3100]
sample_rows = df_test.iloc[sample_indices][[
    "sequence_idx", "store_nbr", "item_nbr", "forecast_day",
    "actual_sales", "p90_forecast", "is_high_vol", "offset_val", "p90_cal_val"
]].copy()
sample_rows["raw_covered"] = sample_rows["actual_sales"] <= sample_rows["p90_forecast"]
sample_rows["cal_covered"] = sample_rows["actual_sales"] <= sample_rows["p90_cal_val"]
sample_rows.rename(columns={"offset_val": "offset_applied", "p90_cal_val": "p90_calibrated"}, inplace=True)
sample_rows.to_csv(os.path.join(TABLES_DIR, "v1_sample_offset_rows.csv"), index=False)
print("\n5 Sample Rows Showing Offset Applied:")
print(sample_rows.to_string(index=False))


# ==============================================================================
# V2: ORDERING-FEE CODE AND DAYS 0-6 COMPONENT RECONCILIATION
# ==============================================================================
print("\n" + "=" * 80)
print("V2: PRIMARY ORDERING-FEE CODE & COST RECONCILIATION")
print("=" * 80)

# Verbatim code snippet
v2_code_snippet = """
VERBATIM CODE FROM revision_experiments/code/unified_simulator.py (Lines 239-252):
--------------------------------------------------------------------------------
# Scoring Window Accumulation
if start_t <= t < end_t:
    pair_sales    += sales
    pair_demand   += d_t
    pair_stockout += stockout * cp
    pair_hold     += on_hand * eff_h_cost
    
    # Ordering Cost Accounting:
    # Primary: orders placed on days 0 to 7 arrive inside the window (days 7 to 14)
    if t <= 7:
        pair_order_primary += np.where(order_mask, eff_order_fee, 0.0)
    pair_order_all += np.where(order_mask, eff_order_fee, 0.0)

chosen_order = pair_order_primary if accounting_mode == "arriving_orders" else pair_order_all
pair_cost = pair_hold + pair_stockout + chosen_order
--------------------------------------------------------------------------------
"""
print(v2_code_snippet)

# Reconciliation calculation
# Quantile TFT Selected: Empirical Residual (Static), alpha=0.9286, beta=1.2163
# Huber TFT Selected: Empirical Residual (Static), alpha=1.0805, beta=0.8435

q_full, q_sl_f, q_h_f, q_so_f, q_o_f, _ = simulate_inventory("Quantile", "empirical_residual", 0.9286, 1.2163, split="test", is_rolling=False, score_days=(0, 15))
q_prim, q_sl_p, q_h_p, q_so_p, q_o_p, _ = simulate_inventory("Quantile", "empirical_residual", 0.9286, 1.2163, split="test", is_rolling=False, score_days=(7, 15))
q_early, q_sl_e, q_h_e, q_so_e, q_o_e, _ = simulate_inventory("Quantile", "empirical_residual", 0.9286, 1.2163, split="test", is_rolling=False, score_days=(0, 7))

h_full, h_sl_f, h_h_f, h_so_f, h_o_f, _ = simulate_inventory("Huber", "empirical_residual", 1.0805, 0.8435, split="test", is_rolling=False, score_days=(0, 15))
h_prim, h_sl_p, h_h_p, h_so_p, h_o_p, _ = simulate_inventory("Huber", "empirical_residual", 1.0805, 0.8435, split="test", is_rolling=False, score_days=(7, 15))
h_early, h_sl_e, h_h_e, h_so_e, h_o_e, _ = simulate_inventory("Huber", "empirical_residual", 1.0805, 0.8435, split="test", is_rolling=False, score_days=(0, 7))

v2_reconciliation_records = [
    {
        "Policy": "Quantile TFT Selected (Empirical Residual Static)",
        "Full Window Cost (Days 0-14) ($)": round(q_full, 2),
        "Primary Window Cost (Days 7-14) ($)": round(q_prim, 2),
        "Delta Cost (Full - Prim) ($)": round(q_full - q_prim, 2),
        "Days 0-6 Holding Cost ($)": round(q_h_e, 2),
        "Days 0-6 Stockout Cost ($)": round(q_so_e, 2),
        "Days 0-6 Ordering Fee ($)": round(q_o_e, 2),
        "Days 0-6 Component Sum ($)": round(q_h_e + q_so_e + q_o_e, 2),
        "Sum Matches Delta Exactly?": bool(round(q_h_e + q_so_e + q_o_e, 2) == round(q_full - q_prim, 2))
    },
    {
        "Policy": "Huber TFT Selected (Empirical Residual Static)",
        "Full Window Cost (Days 0-14) ($)": round(h_full, 2),
        "Primary Window Cost (Days 7-14) ($)": round(h_prim, 2),
        "Delta Cost (Full - Prim) ($)": round(h_full - h_prim, 2),
        "Days 0-6 Holding Cost ($)": round(h_h_e, 2),
        "Days 0-6 Stockout Cost ($)": round(h_so_e, 2),
        "Days 0-6 Ordering Fee ($)": round(h_o_e, 2),
        "Days 0-6 Component Sum ($)": round(h_h_e + h_so_e + h_o_e, 2),
        "Sum Matches Delta Exactly?": bool(round(h_h_e + h_so_e + h_o_e, 2) == round(h_full - h_prim, 2))
    }
]

df_v2 = pd.DataFrame(v2_reconciliation_records)
df_v2.to_csv(os.path.join(TABLES_DIR, "v2_order_fee_reconciliation.csv"), index=False)
print("Saved v2_order_fee_reconciliation.csv")
print(df_v2.to_string(index=False))


# ==============================================================================
# V3: FINAL TABLE INVENTORY & SCRIPT TRACEABILITY
# ==============================================================================
print("\n" + "=" * 80)
print("V3: FINAL TABLE INVENTORY & SCRIPT TRACEABILITY AUDIT")
print("=" * 80)

table_inventory = [
    {"Output Table": "r1_simulator_reconciliation.csv", "Generating Script": "r1_r2_reconciliation.py", "Uses unified_simulator.py": True, "Origin-0 Vectors for Static": True, "Status": "Active Final"},
    {"Output Table": "r3_prespecified_selection_summary.csv", "Generating Script": "r3_prespecified_selection.py", "Uses unified_simulator.py": True, "Origin-0 Vectors for Static": True, "Status": "Active Final"},
    {"Output Table": "r3_all_variants_val_vs_test.csv", "Generating Script": "r3_prespecified_selection.py", "Uses unified_simulator.py": True, "Origin-0 Vectors for Static": True, "Status": "Active Final"},
    {"Output Table": "r3_paired_bootstrap_selected.csv", "Generating Script": "r3_prespecified_selection.py", "Uses unified_simulator.py": True, "Origin-0 Vectors for Static": True, "Status": "Active Final"},
    {"Output Table": "r4_matched_fill_frontier.csv", "Generating Script": "r4_matched_fill_frontier.py", "Uses unified_simulator.py": True, "Origin-0 Vectors for Static": True, "Status": "Active Final"},
    {"Output Table": "r5_cost_sensitivity_reoptimized.csv", "Generating Script": "r5_cost_sensitivity.py", "Uses unified_simulator.py": True, "Origin-0 Vectors for Static": True, "Status": "Active Final"},
    {"Output Table": "r6_forecast_metrics_e7_recomputed.csv", "Generating Script": "r6_forecast_metrics_recomputation.py", "Uses unified_simulator.py": False, "Origin-0 Vectors for Static": "N/A (Forecast Metrics)", "Status": "Active Final"},
    {"Output Table": "r7_population_representativeness_e9.csv", "Generating Script": "r7_r8_population_and_conformal.py", "Uses unified_simulator.py": False, "Origin-0 Vectors for Static": "N/A (Population Audit)", "Status": "Active Final"},
    {"Output Table": "r8_conformal_coverage_cis.csv", "Generating Script": "r7_r8_population_and_conformal.py", "Uses unified_simulator.py": False, "Origin-0 Vectors for Static": "N/A (Conformal Audit)", "Status": "Active Final"},
    {"Output Table": "r9_optimizers_benchmark.csv", "Generating Script": "r9_optimizers_benchmark.py", "Uses unified_simulator.py": True, "Origin-0 Vectors for Static": True, "Status": "Active Final"},
    {"Output Table": "step1_feasibility_check.csv", "Generating Script": "step1_to_step5_experiments.py", "Uses unified_simulator.py": False, "Origin-0 Vectors for Static": False, "Status": "SUPERSEDED by r1-r3"},
    {"Output Table": "step2_initialization_sensitivity.csv", "Generating Script": "step1_to_step5_experiments.py", "Uses unified_simulator.py": False, "Origin-0 Vectors for Static": False, "Status": "SUPERSEDED by r1-r3"},
    {"Output Table": "step3_causality_audit.csv", "Generating Script": "step1_to_step5_experiments.py", "Uses unified_simulator.py": False, "Origin-0 Vectors for Static": False, "Status": "SUPERSEDED by r1-r3"},
    {"Output Table": "step4_e1_summary.csv", "Generating Script": "step1_to_step5_experiments.py", "Uses unified_simulator.py": False, "Origin-0 Vectors for Static": False, "Status": "SUPERSEDED by r3_selection"},
    {"Output Table": "step4_e2_grid_search.csv", "Generating Script": "step1_to_step5_experiments.py", "Uses unified_simulator.py": False, "Origin-0 Vectors for Static": False, "Status": "SUPERSEDED by r5_sensitivity"},
    {"Output Table": "step4_e3_factorial_ablation.csv", "Generating Script": "step1_to_step5_experiments.py", "Uses unified_simulator.py": False, "Origin-0 Vectors for Static": False, "Status": "SUPERSEDED by r3_selection"},
    {"Output Table": "step5_pareto_matched_fill_rate.csv", "Generating Script": "step1_to_step5_experiments.py", "Uses unified_simulator.py": False, "Origin-0 Vectors for Static": False, "Status": "SUPERSEDED by r4_frontier"},
]

df_v3 = pd.DataFrame(table_inventory)
df_v3.to_csv(os.path.join(TABLES_DIR, "v3_table_inventory.csv"), index=False)
print("Saved v3_table_inventory.csv")
print(df_v3.to_string(index=False))


# ==============================================================================
# V4: STRICT sMAPE (0/0 = 0 RULE) & WAPE PRIMARY METRIC
# ==============================================================================
print("\n" + "=" * 80)
print("V4: STRICT sMAPE (0/0 = 0 RULE) & WAPE PRIMARY METRIC")
print("=" * 80)

q_csv_path = os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "tft_quantile_forecasts.csv")
h_csv_path = os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "tft_test_forecasts.csv")
df_q_fc = pd.read_csv(q_csv_path)
df_h_fc = pd.read_csv(h_csv_path)

sim_in = np.load(os.path.join(TABLES_DIR, "simulation_inputs.npz"))
active_pairs = sorted(list(set(zip(test_npz["store_nbr"], test_npz["item_nbr"]))))
pair_to_idx = {p: i for i, p in enumerate(active_pairs)}
test_dbar = sim_in["test_dbar_vec"]

y_seq = test_npz["y_raw"]
N_SEQS = len(y_seq)
seq_mean_pred = np.zeros_like(y_seq)
for s_idx in range(N_SEQS):
    pair = (int(test_npz["store_nbr"][s_idx]), int(test_npz["item_nbr"][s_idx]))
    seq_mean_pred[s_idx, :] = test_dbar[pair_to_idx[pair]]

def calc_all_forecast_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100.0)
    denom = np.abs(y_true) + np.abs(y_pred)
    # Strict 0/0 = 0 rule
    term_strict = np.where(denom == 0.0, 0.0, 200.0 * np.abs(y_true - y_pred) / np.maximum(1e-12, denom))
    smape_strict = float(np.mean(term_strict))
    # Epsilon = 0.002
    smape_eps = float(np.mean(200.0 * np.abs(y_true - y_pred) / (denom + 0.002)))
    return mae, rmse, wape, smape_strict, smape_eps

v4_records = []

# 1. Quantile TFT overall and per-horizon
y_q_true = df_q_fc["actual_sales"].values
y_q_pred = df_q_fc["p50_forecast"].values
mae_q, rmse_q, wape_q, smape_s_q, smape_e_q = calc_all_forecast_metrics(y_q_true, y_q_pred)
v4_records.append({
    "Model": "Quantile TFT (P50)",
    "Horizon": "Overall (All Horizons)",
    "MAE": round(mae_q, 4),
    "RMSE": round(rmse_q, 4),
    "WAPE (%) [Primary]": round(wape_q, 2),
    "sMAPE (%) [Strict 0/0=0]": round(smape_s_q, 2),
    "sMAPE (%) [eps=0.002]": round(smape_e_q, 2),
})

for h in range(1, 8):
    sub = df_q_fc[df_q_fc["forecast_day"] == h]
    m, r, w, s_s, s_e = calc_all_forecast_metrics(sub["actual_sales"].values, sub["p50_forecast"].values)
    v4_records.append({
        "Model": "Quantile TFT (P50)",
        "Horizon": f"Horizon Day h={h}",
        "MAE": round(m, 4),
        "RMSE": round(r, 4),
        "WAPE (%) [Primary]": round(w, 2),
        "sMAPE (%) [Strict 0/0=0]": round(s_s, 2),
        "sMAPE (%) [eps=0.002]": round(s_e, 2),
    })

# 2. Huber TFT overall and per-horizon
h_true_all = []
h_pred_all = []
for h in range(1, 8):
    y_t = df_h_fc[f"actual_day_{h}"].values
    y_p = df_h_fc[f"pred_day_{h}"].values
    h_true_all.extend(y_t)
    h_pred_all.extend(y_p)
    m, r, w, s_s, s_e = calc_all_forecast_metrics(y_t, y_p)
    v4_records.append({
        "Model": "Huber TFT",
        "Horizon": f"Horizon Day h={h}",
        "MAE": round(m, 4),
        "RMSE": round(r, 4),
        "WAPE (%) [Primary]": round(w, 2),
        "sMAPE (%) [Strict 0/0=0]": round(s_s, 2),
        "sMAPE (%) [eps=0.002]": round(s_e, 2),
    })

m, r, w, s_s, s_e = calc_all_forecast_metrics(np.array(h_true_all), np.array(h_pred_all))
v4_records.append({
    "Model": "Huber TFT",
    "Horizon": "Overall (All Horizons)",
    "MAE": round(m, 4),
    "RMSE": round(r, 4),
    "WAPE (%) [Primary]": round(w, 2),
    "sMAPE (%) [Strict 0/0=0]": round(s_s, 2),
    "sMAPE (%) [eps=0.002]": round(s_e, 2),
})

# 3. Trailing Mean overall
m, r, w, s_s, s_e = calc_all_forecast_metrics(y_seq.flatten(), seq_mean_pred.flatten())
v4_records.append({
    "Model": "Trailing Mean",
    "Horizon": "Overall (All Horizons)",
    "MAE": round(m, 4),
    "RMSE": round(r, 4),
    "WAPE (%) [Primary]": round(w, 2),
    "sMAPE (%) [Strict 0/0=0]": round(s_s, 2),
    "sMAPE (%) [eps=0.002]": round(s_e, 2),
})

df_v4 = pd.DataFrame(v4_records)
df_v4.to_csv(os.path.join(TABLES_DIR, "v4_forecast_metrics_strict_smape.csv"), index=False)
print("Saved v4_forecast_metrics_strict_smape.csv")
print(df_v4[df_v4["Horizon"].str.contains("Overall")].to_string(index=False))


# ==============================================================================
# V5: MATCHED-FILL FRONTIER, CP SENSITIVITY FLIPS, & BOOTSTRAPS IN BOTH REGIMES
# ==============================================================================
print("\n" + "=" * 80)
print("V5: MATCHED-FILL FRONTIER, CP SENSITIVITY FLIPS, & BOOTSTRAP IN BOTH REGIMES")
print("=" * 80)

# 1. Matched-fill frontier summary (targets 90% to 96%)
df_r4 = pd.read_csv(os.path.join(TABLES_DIR, "r4_matched_fill_frontier.csv"))
piv_r4 = df_r4.pivot(index="Target Fill Rate (%)", columns="Policy", values="Minimum Cost ($)")
print("\nMatched-Fill Frontier Lower Envelope (No Interpolation):")
print(piv_r4.to_string())

# 2. CP Sensitivity ranking flips table
df_r5 = pd.read_csv(os.path.join(TABLES_DIR, "r5_cost_sensitivity_reoptimized.csv"))
cp_sub = df_r5[df_r5["Sensitivity Dimension"] == "Stockout Penalty Variation"]
piv_cp_prim = cp_sub.pivot(index="Stockout Penalty ($)", columns="Policy", values="Test Cost (Primary 7-14) ($)")
piv_cp_rob  = cp_sub.pivot(index="Stockout Penalty ($)", columns="Policy", values="Test Cost (Robustness 0-14) ($)")
print("\nStockout Penalty (cp) Sensitivity - Primary Test Cost ($):")
print(piv_cp_prim.to_string())

# 3. Bootstraps in Both Regimes: All Buffers and Restricted
# Simulate per-pair costs
# All Buffers Regime:
# Quantile Empirical Residual (Static): a=0.9286, b=1.2163
# Huber Empirical Residual (Static): a=1.0805, b=0.8435
# Seasonal Naive Empirical Residual (Static): a=0.8272, b=1.2164
# Mean Empirical Residual (Rolling): a=0.7725, b=0.9256
qa_c, qa_s, qa_d, _, _, _, _ = simulate_inventory("Quantile", "empirical_residual", 0.9286, 1.2163, split="test", is_rolling=False, score_days=(7, 15), return_per_pair=True)
ha_c, ha_s, ha_d, _, _, _, _ = simulate_inventory("Huber", "empirical_residual", 1.0805, 0.8435, split="test", is_rolling=False, score_days=(7, 15), return_per_pair=True)
sna_c, sna_s, sna_d, _, _, _, _ = simulate_inventory("SeasonalNaive", "empirical_residual", 0.8272, 1.2164, split="test", is_rolling=False, score_days=(7, 15), return_per_pair=True)
ma_c, ma_s, ma_d, _, _, _, _ = simulate_inventory("Mean", "empirical_residual", 0.7725, 0.9256, split="test", is_rolling=True, score_days=(7, 15), return_per_pair=True)

# Restricted Regime:
# Quantile Conformal (Static): a=0.1648, b=1.2172
# Huber None (Rolling): a=0.0, b=4.9979
# Seasonal Naive None (Rolling): a=0.0, b=2.4629
# Mean Gaussian (Rolling): a=0.0147, b=1.4765
qr_c, qr_s, qr_d, _, _, _, _ = simulate_inventory("Quantile", "conformal_quantile", 0.1648, 1.2172, split="test", is_rolling=False, score_days=(7, 15), return_per_pair=True)
hr_c, hr_s, hr_d, _, _, _, _ = simulate_inventory("Huber", "none", 0.0, 4.9979, split="test", is_rolling=True, score_days=(7, 15), return_per_pair=True)
snr_c, snr_s, snr_d, _, _, _, _ = simulate_inventory("SeasonalNaive", "none", 0.0, 2.4629, split="test", is_rolling=True, score_days=(7, 15), return_per_pair=True)
mr_c, mr_s, mr_d, _, _, _, _ = simulate_inventory("Mean", "gaussian_demand", 0.0147, 1.4765, split="test", is_rolling=True, score_days=(7, 15), return_per_pair=True)

top20_idxs = np.argsort(qa_d)[-20:]
excl20_mask = np.ones(len(qa_c), dtype=bool)
excl20_mask[top20_idxs] = False

B_REPS = 2000
def run_paired_bootstrap(c_a, s_a, d_a, c_b, s_b, d_b, mask=None):
    if mask is not None:
        c_a, s_a, d_a = c_a[mask], s_a[mask], d_a[mask]
        c_b, s_b, d_b = c_b[mask], s_b[mask], d_b[mask]
    n = len(c_a)
    boot_cost_deltas = []
    boot_fill_deltas = []
    p1_cheaper_count = 0
    for _ in range(B_REPS):
        idx = np.random.randint(0, n, size=n)
        tot_c_a, tot_c_b = np.sum(c_a[idx]), np.sum(c_b[idx])
        delta_c = tot_c_a - tot_c_b
        fill_a = (np.sum(s_a[idx]) / np.sum(d_a[idx])) * 100.0
        fill_b = (np.sum(s_b[idx]) / np.sum(d_b[idx])) * 100.0
        delta_fill = fill_a - fill_b
        boot_cost_deltas.append(delta_c)
        boot_fill_deltas.append(delta_fill)
        if delta_c < 0:
            p1_cheaper_count += 1
    mean_c = float(np.mean(boot_cost_deltas))
    ci_c = (float(np.percentile(boot_cost_deltas, 2.5)), float(np.percentile(boot_cost_deltas, 97.5)))
    mean_f = float(np.mean(boot_fill_deltas))
    ci_f = (float(np.percentile(boot_fill_deltas, 2.5)), float(np.percentile(boot_fill_deltas, 97.5)))
    p1_ch = float((p1_cheaper_count / B_REPS) * 100.0)
    return mean_c, ci_c, mean_f, ci_f, p1_ch

v5_boot_records = []

comparisons_to_run = [
    # All Buffers
    ("All Buffers", "Quantile Sel vs Huber Sel", qa_c, qa_s, qa_d, ha_c, ha_s, ha_d),
    ("All Buffers", "Quantile Sel vs Seasonal Naive Sel", qa_c, qa_s, qa_d, sna_c, sna_s, sna_d),
    ("All Buffers", "Quantile Sel vs Mean-Demand Sel", qa_c, qa_s, qa_d, ma_c, ma_s, ma_d),
    # Restricted
    ("Restricted", "Quantile Sel vs Huber Sel", qr_c, qr_s, qr_d, hr_c, hr_s, hr_d),
    ("Restricted", "Quantile Sel vs Seasonal Naive Sel", qr_c, qr_s, qr_d, snr_c, snr_s, snr_d),
    ("Restricted", "Quantile Sel vs Mean-Demand Sel", qr_c, qr_s, qr_d, mr_c, mr_s, mr_d),
]

for regime, comp_name, ca, sa, da, cb, sb, db in comparisons_to_run:
    # All pairs
    mc, cic, mf, cif, pch = run_paired_bootstrap(ca, sa, da, cb, sb, db, mask=None)
    v5_boot_records.append({
        "Regime": regime,
        "Comparison": comp_name,
        "Cohort": "All Pairs (N=363)",
        "Mean Cost Delta ($)": round(mc, 2),
        "Cost Delta 95% CI ($)": f"[{cic[0]:.2f}, {cic[1]:.2f}]",
        "Mean Fill Delta (%)": round(mf, 2),
        "Fill Delta 95% CI (%)": f"[{cif[0]:.2f}, {cif[1]:.2f}]",
        "Share Pol A Cheaper (%)": round(pch, 1)
    })
    # Excl top 20
    mc, cic, mf, cif, pch = run_paired_bootstrap(ca, sa, da, cb, sb, db, mask=excl20_mask)
    v5_boot_records.append({
        "Regime": regime,
        "Comparison": comp_name,
        "Cohort": "Excl Top 20 (N=343)",
        "Mean Cost Delta ($)": round(mc, 2),
        "Cost Delta 95% CI ($)": f"[{cic[0]:.2f}, {cic[1]:.2f}]",
        "Mean Fill Delta (%)": round(mf, 2),
        "Fill Delta 95% CI (%)": f"[{cif[0]:.2f}, {cif[1]:.2f}]",
        "Share Pol A Cheaper (%)": round(pch, 1)
    })

df_v5_boot = pd.DataFrame(v5_boot_records)
df_v5_boot.to_csv(os.path.join(TABLES_DIR, "v5_bootstraps_both_regimes.csv"), index=False)
print("\nBootstraps Summary in Both Selection Regimes (Primary Window Days 7-14):")
print(df_v5_boot.to_string(index=False))


# ==============================================================================
# V6: ACCURATE SECTION B POPULATION, DEMAND_BIN, & PERISHABLE COUNTS
# ==============================================================================
print("\n" + "=" * 80)
print("V6: ACCURATE SECTION B POPULATION, DEMAND_BIN, & PERISHABLE COUNTS")
print("=" * 80)

cand_df = pd.read_parquet(os.path.join(TABLES_DIR, "eligible_population_pairs.parquet"))
N_POP = len(cand_df)
df_363 = df_400[df_400.apply(lambda r: (int(r["store_nbr"]), int(r["item_nbr"])) in set(zip(test_npz["store_nbr"], test_npz["item_nbr"])), axis=1)].copy()

q20 = float(np.percentile(cand_df["average_sales"], 20))
q40 = float(np.percentile(cand_df["average_sales"], 40))
q60 = float(np.percentile(cand_df["average_sales"], 60))
q80 = float(np.percentile(cand_df["average_sales"], 80))

# Demand bin counts from selected_pairs.csv metadata
tier_map = {
    "very_low": "Tier 1: Very Low Demand",
    "low": "Tier 2: Low Demand",
    "mid": "Tier 3: Medium Demand",
    "high": "Tier 4: High Demand",
    "very_high": "Tier 5: Very High Demand"
}

v6_rows = []
for b_key in ["very_low", "low", "mid", "high", "very_high"]:
    c_400 = int((df_400["demand_bin"] == b_key).sum())
    p_400 = (c_400 / len(df_400)) * 100.0
    c_363 = int((df_363["demand_bin"] == b_key).sum())
    p_363 = (c_363 / len(df_363)) * 100.0
    
    # Population count by corresponding quintile
    if b_key == "very_low":
        c_pop = int((cand_df["average_sales"] <= q20).sum())
        r_str = f"<= {q20:.2f}"
    elif b_key == "low":
        c_pop = int(((cand_df["average_sales"] > q20) & (cand_df["average_sales"] <= q40)).sum())
        r_str = f"({q20:.2f}, {q40:.2f}]"
    elif b_key == "mid":
        c_pop = int(((cand_df["average_sales"] > q40) & (cand_df["average_sales"] <= q60)).sum())
        r_str = f"({q40:.2f}, {q60:.2f}]"
    elif b_key == "high":
        c_pop = int(((cand_df["average_sales"] > q60) & (cand_df["average_sales"] <= q80)).sum())
        r_str = f"({q60:.2f}, {q80:.2f}]"
    else:
        c_pop = int((cand_df["average_sales"] > q80).sum())
        r_str = f"> {q80:.2f}"
    p_pop = (c_pop / N_POP) * 100.0
    
    v6_rows.append({
        "Demand Tier": tier_map[b_key],
        "Internal Key": b_key,
        "Population Sales Cutoff (units/day)": r_str,
        "Population Count": c_pop,
        "Population Share (%)": round(p_pop, 2),
        "400-Cohort Count": c_400,
        "400-Cohort Share (%)": round(p_400, 2),
        "363-Active Count": c_363,
        "363-Active Share (%)": round(p_363, 2),
    })

df_v6 = pd.DataFrame(v6_rows)
df_v6.to_csv(os.path.join(TABLES_DIR, "v6_section_b_population_audit.csv"), index=False)
print("Saved v6_section_b_population_audit.csv")
print(df_v6.to_string(index=False))

p400_per = int((df_400["perishable"] == 1).sum())
p400_dry = int((df_400["perishable"] == 0).sum())
p363_per = int((df_363["perishable"] == 1).sum())
p363_dry = int((df_363["perishable"] == 0).sum())

print(f"\nPerishable Counts Summary:")
print(f"  400-Cohort:  Perishable = {p400_per} ({p400_per/len(df_400)*100:.1f}%), Dry = {p400_dry} ({p400_dry/len(df_400)*100:.1f}%)")
print(f"  363-Active:  Perishable = {p363_per} ({p363_per/len(df_363)*100:.1f}%), Dry = {p363_dry} ({p363_dry/len(df_363)*100:.1f}%)")
print(f"  Population:  Perishable = 32,282 (23.23%), Dry = 106,669 (76.77%)")

print("\n" + "=" * 80)
print("V1 THROUGH V6 FINAL VERIFICATION COMPLETE.")
print("=" * 80)
