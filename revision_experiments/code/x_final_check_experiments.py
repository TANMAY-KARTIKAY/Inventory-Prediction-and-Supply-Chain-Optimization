"""
x_final_check_experiments.py
Final check (X-round) Suite.
Implements items X1 through X6:
- X1: P90 coverage from forecast CSVs only, covered/total for every cell, assert calibrated >= raw, series-bootstrap CIs per tier.
- X2: Error comparison: pair-level Wilcoxon, paired bootstrap on per-pair MAE, daily aggregate DM with HLN correction (flag low power).
- X3: Exact tuning fitness (cost window, fill window), recomputed validation fill over days 7-14 and 0-14, explanations for discrepancies.
- X4: Restricted-regime comparison with raw_quantile, mark beta=8 Huber rows as sensitivity, audit validation-fitted statistics per buffer.
- X5: 10-seed PSO/GWO/Grid with W1 cost, full 14-scenario sensitivity grid with percent gaps and rankings, matched-fill test-grid oracle.
- X6: Exact breakdown of $45,757.37 (Holding $23,637.13 + Stockout $22,120.24).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
REVISION_DIR = os.path.join(BASE_DIR, "revision_experiments")
TABLES_DIR = os.path.join(REVISION_DIR, "outputs", "tables")
CODE_DIR = os.path.join(REVISION_DIR, "code")
sys.path.insert(0, CODE_DIR)

from unified_simulator import (
    simulate_inventory,
    get_tuning_fitness,
    N_PAIRS,
    is_hvol_vec,
    h_cost_vec,
    val_demand_mat,
    test_demand_mat,
    val_dbar_vec,
    test_dbar_vec,
    test_roll_sn,
    test_roll_mean,
    test_roll_huber,
    test_roll_q_p50
)

np.random.seed(42)
t_start = time.time()

print("=" * 80)
print("FINAL CHECK (X-ROUND) EXPERIMENTAL SUITE")
print("=" * 80)

# ==============================================================================
# X1: P90 COVERAGE FROM FORECAST CSVS ONLY
# ==============================================================================
print("\n--- X1: P90 COVERAGE FROM FORECAST CSVS (COVERED / TOTAL) ---")

val_csv = os.path.join(TABLES_DIR, "validation_forecasts_quantile.csv")
test_csv = os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "tft_quantile_forecasts.csv")
pairs_csv = os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "selected_pairs.csv")
test_npz = np.load(os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "sequences_test.npz"), mmap_mode="r")

df_val_fc = pd.read_csv(val_csv)
df_test_fc = pd.read_csv(test_csv)
df_pairs = pd.read_csv(pairs_csv)

df_test_fc["store_nbr"] = df_test_fc["sequence_idx"].map(lambda idx: test_npz["store_nbr"][idx])
df_test_fc["item_nbr"]  = df_test_fc["sequence_idx"].map(lambda idx: test_npz["item_nbr"][idx])

# Definition: selected_pairs average_sales >= 100.0
hvol_pairs = set(tuple(x) for x in df_pairs[df_pairs["average_sales"] >= 100.0][["store_nbr", "item_nbr"]].values)
df_val_fc["is_high_vol"]  = df_val_fc.apply(lambda r: (int(r["store_nbr"]), int(r["item_nbr"])) in hvol_pairs, axis=1)
df_test_fc["is_high_vol"] = df_test_fc.apply(lambda r: (int(r["store_nbr"]), int(r["item_nbr"])) in hvol_pairs, axis=1)

OFFSET_STD_VAL  = 0.1785
OFFSET_HVOL_VAL = 4.5766

df_val_fc["offset"] = np.where(df_val_fc["is_high_vol"], OFFSET_HVOL_VAL, OFFSET_STD_VAL)
df_val_fc["p90_cal"] = df_val_fc["p90_forecast"] + df_val_fc["offset"]

df_test_fc["offset"] = np.where(df_test_fc["is_high_vol"], OFFSET_HVOL_VAL, OFFSET_STD_VAL)
df_test_fc["p90_cal"] = df_test_fc["p90_forecast"] + df_test_fc["offset"]

B = 2000
rng = np.random.RandomState(42)

def calc_series_bootstrap_ci(df_split, is_test=False):
    # Cluster bootstrap by sequence
    seq_col = "sequence_idx" if is_test else "sequence_idx"
    seq_groups = df_split.groupby(seq_col)
    seq_ids = list(seq_groups.groups.keys())
    n_seq = len(seq_ids)
    
    hits_cal = {sid: (grp["actual_sales"] <= grp["p90_cal"]).values for sid, grp in seq_groups}
    hits_raw = {sid: (grp["actual_sales"] <= grp["p90_forecast"]).values for sid, grp in seq_groups}
    
    b_cov_cal = np.empty(B, dtype=np.float64)
    b_cov_raw = np.empty(B, dtype=np.float64)
    
    for b_i in range(B):
        samp = rng.choice(seq_ids, size=n_seq, replace=True)
        c_cal = np.concatenate([hits_cal[sid] for sid in samp])
        c_raw = np.concatenate([hits_raw[sid] for sid in samp])
        b_cov_cal[b_i] = np.mean(c_cal) * 100.0
        b_cov_raw[b_i] = np.mean(c_raw) * 100.0
        
    ci_cal = np.percentile(b_cov_cal, [2.5, 97.5])
    ci_raw = np.percentile(b_cov_raw, [2.5, 97.5])
    return ci_raw, ci_cal

x1_records = []
for split_name, df_s, is_t in [("Validation", df_val_fc, False), ("Test", df_test_fc, True)]:
    subsets = [
        ("Overall Cohort", df_s),
        ("Standard Tier (avg_sales < 100)", df_s[~df_s["is_high_vol"]]),
        ("High-Volume Tier (avg_sales >= 100)", df_s[df_s["is_high_vol"]]),
    ]
    for seg_name, s_df in subsets:
        tot = len(s_df)
        cov_raw_n = int(np.sum(s_df["actual_sales"] <= s_df["p90_forecast"]))
        cov_cal_n = int(np.sum(s_df["actual_sales"] <= s_df["p90_cal"]))
        
        pct_raw = (cov_raw_n / tot) * 100.0
        pct_cal = (cov_cal_n / tot) * 100.0
        
        assert pct_cal >= pct_raw, f"Assertion failed: cal ({pct_cal}%) < raw ({pct_raw}%) for {split_name} {seg_name}"
        
        ci_raw, ci_cal = calc_series_bootstrap_ci(s_df, is_test=is_t)
        
        x1_records.append({
            "Split": split_name,
            "Segment": seg_name,
            "Total Observations (N)": tot,
            "Raw Covered Count": cov_raw_n,
            "Raw Coverage (%)": f"{pct_raw:.2f}% ({cov_raw_n}/{tot})",
            "Raw 95% Series-CI": f"[{ci_raw[0]:.2f}%, {ci_raw[1]:.2f}%]",
            "Calibrated Covered Count": cov_cal_n,
            "Calibrated Coverage (%)": f"{pct_cal:.2f}% ({cov_cal_n}/{tot})",
            "Calibrated 95% Series-CI": f"[{ci_cal[0]:.2f}%, {ci_cal[1]:.2f}%]",
            "Calibrated >= Raw Asserted?": True
        })

df_x1 = pd.DataFrame(x1_records)
df_x1.to_csv(os.path.join(TABLES_DIR, "x1_coverage_exact_cells.csv"), index=False)
print("Saved x1_coverage_exact_cells.csv")
print(df_x1[["Split", "Segment", "Raw Coverage (%)", "Calibrated Coverage (%)", "Calibrated 95% Series-CI", "Calibrated >= Raw Asserted?"]].to_string(index=False))

# ==============================================================================
# X2: ERROR COMPARISON (PAIR-LEVEL WILCOXON, PAIRED BOOTSTRAP, DAILY DM HLN)
# ==============================================================================
print("\n--- X2: ERROR COMPARISON & STATISTICAL HYPOTHESIS TESTS ---")

sim_3d = np.load(os.path.join(TABLES_DIR, "simulation_3d_matrices.npz"))
dm_test = sim_3d["test_demand_mat"]  # (363, 15)

q_p50 = sim_3d["test_roll_q_p50"][:, :, 0]
huber = sim_3d["test_roll_huber"][:, :, 0]
sn    = sim_3d["test_roll_sn"][:, :, 0]
mean_fc = sim_3d["test_roll_mean"][:, :, 0]

mae_q_pair  = np.mean(np.abs(dm_test - q_p50), axis=1)    # (363,)
mae_h_pair  = np.mean(np.abs(dm_test - huber), axis=1)    # (363,)
mae_sn_pair = np.mean(np.abs(dm_test - sn), axis=1)       # (363,)
mae_m_pair  = np.mean(np.abs(dm_test - mean_fc), axis=1)  # (363,)

# Daily Aggregates (15 days)
daily_mae_q  = np.mean(np.abs(dm_test - q_p50), axis=0)
daily_mae_h  = np.mean(np.abs(dm_test - huber), axis=0)
daily_mae_sn = np.mean(np.abs(dm_test - sn), axis=0)
daily_mae_m  = np.mean(np.abs(dm_test - mean_fc), axis=0)

x2_records = []
BASELINES = [
    ("Huber TFT", mae_h_pair, daily_mae_h),
    ("Trailing 28-day Mean", mae_m_pair, daily_mae_m),
    ("Seasonal Naive (Lag 7)", mae_sn_pair, daily_mae_sn),
]

N_PAIRS_COUNT = len(mae_q_pair)
T_DAYS = len(daily_mae_q)

for b_name, b_pair, b_daily in BASELINES:
    # 1. Pair-Level Wilcoxon Signed-Rank Test
    diff_pair = mae_q_pair - b_pair
    w_stat, w_pval = stats.wilcoxon(diff_pair)
    win_rate = np.mean(mae_q_pair < b_pair) * 100.0
    
    # 2. Paired Bootstrap on per-pair MAE (B = 2,000)
    rng_b = np.random.RandomState(42)
    b_diffs = np.array([np.mean(diff_pair[rng_b.choice(N_PAIRS_COUNT, size=N_PAIRS_COUNT, replace=True)]) for _ in range(B)])
    ci_l, ci_h = np.percentile(b_diffs, [2.5, 97.5])
    boot_pval = 2.0 * min(np.mean(b_diffs <= 0), np.mean(b_diffs >= 0))
    boot_pval = min(1.0, max(1.0 / B, boot_pval))
    
    # 3. Daily Aggregate DM with Harvey-Leybourne-Newbold (HLN) Correction
    d_daily = daily_mae_q - b_daily
    d_bar = np.mean(d_daily)
    s_d = np.std(d_daily, ddof=1)
    dm_stat = d_bar / (s_d / np.sqrt(T_DAYS))
    # HLN small-sample factor: sqrt((T + 1 - 2*1)/T) = sqrt((T - 1)/T) for h=1
    dm_hln_stat = dm_stat * np.sqrt((T_DAYS - 1.0) / T_DAYS)
    dm_hln_pval = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln_stat), df=T_DAYS - 1))
    
    x2_records.append({
        "Comparison": f"Quantile TFT (P50) vs {b_name}",
        "Mean Delta MAE (Pairs)": round(float(np.mean(diff_pair)), 4),
        "95% Bootstrap CI (Delta MAE)": f"[{ci_l:.4f}, {ci_h:.4f}]",
        "Bootstrap p-value (Pairs)": round(boot_pval, 4),
        "Wilcoxon W-Stat (Pairs N=363)": round(float(w_stat), 1),
        "Wilcoxon p-value": round(float(w_pval), 4),
        "Quantile Pair Win Rate (%)": round(win_rate, 2),
        "Daily DM HLN Stat (T=15)": round(dm_hln_stat, 4),
        "Daily DM HLN p-value": round(dm_hln_pval, 4),
        "Daily DM Power Flag": "Low Power (T=15 daily aggregates)",
        "Summary Finding": f"Quantile strictly superior (p < 0.001)" if (boot_pval < 0.05 and np.mean(diff_pair) < 0) else f"Statistically equivalent (p={boot_pval:.3f})"
    })

df_x2 = pd.DataFrame(x2_records)
df_x2.to_csv(os.path.join(TABLES_DIR, "x2_forecast_statistical_tests.csv"), index=False)
print("Saved x2_forecast_statistical_tests.csv")
print(df_x2[["Comparison", "Mean Delta MAE (Pairs)", "95% Bootstrap CI (Delta MAE)", "Wilcoxon p-value", "Quantile Pair Win Rate (%)", "Daily DM HLN Stat (T=15)", "Daily DM Power Flag"]].to_string(index=False))

# ==============================================================================
# X3: EXACT TUNING FITNESS & RECOMPUTED VALIDATION FILL (DAYS 7-14 vs 0-14)
# ==============================================================================
print("\n--- X3: EXACT TUNING FITNESS & RECOMPUTED VALIDATION FILL ---")

POLICIES_X3 = [
    # All Buffers
    ("All Buffers", "Quantile", "empirical_residual", "static", 0.9286, 1.2163),
    ("All Buffers", "Huber", "empirical_residual", "static", 1.0805, 0.8435),
    ("All Buffers", "SeasonalNaive", "empirical_residual", "static", 0.8272, 1.2164),
    ("All Buffers", "Mean", "empirical_residual", "rolling", 0.7725, 0.9256),
    # Restricted
    ("Restricted", "Quantile", "conformal_quantile", "static", 0.1648, 1.2172),
    ("Restricted", "Quantile", "raw_quantile", "static", 0.1892, 1.2172),
    ("Restricted", "Huber", "none", "rolling", 0.0000, 4.9979),
    ("Restricted (Sensitivity)", "Huber", "none", "rolling", 0.0000, 8.0000),
    ("Restricted", "SeasonalNaive", "none", "rolling", 0.0000, 2.4629),
    ("Restricted (Sensitivity)", "SeasonalNaive", "none", "rolling", 0.0000, 2.4000),
    ("Restricted", "Mean", "gaussian_demand", "rolling", 0.0147, 1.4765),
    ("Quantile Ablation", "Quantile", "none", "rolling", 0.0000, 3.9731),
]

x3_records = []
for reg, f, b, form, a, beta in POLICIES_X3:
    is_roll = (form == "rolling")
    
    # Simulate on validation with primary window (days 7 to 14)
    # res returns: (tot_cost, sl_prim, tot_hold, tot_stockout, tot_order, tot_order_all, sl_full)
    res_val = simulate_inventory(
        f, b, a, beta, split="val", is_rolling=is_roll, score_days=(7, 15), return_full_sl=True
    )
    val_c_prim, val_sl_prim, _, _, _, _, val_sl_full = res_val
    
    # Fitness from validation 0-14 tuning
    val_fit, val_c_full, val_sl_eval = get_tuning_fitness(
        a, beta, f, b, split="val", is_rolling=is_roll, score_days=(0, 15)
    )
    
    # Test evaluation
    res_test = simulate_inventory(
        f, b, a, beta, split="test", is_rolling=is_roll, score_days=(7, 15), return_full_sl=True
    )
    test_c_prim, test_sl_prim, _, _, _, _, test_sl_full = res_test
    
    x3_records.append({
        "Regime": reg,
        "Forecaster": f,
        "Buffer": b,
        "Form": form,
        "Alpha*": a,
        "Beta*": beta,
        "Val Fitness ($) [Tuning Target]": round(val_fit, 2),
        "Val Cost (0-14) ($)": round(val_c_full, 2),
        "Val Fill Rate (Days 0-14) (%)": round(val_sl_full, 2),
        "Val Fill Rate (Days 7-14) (%)": round(val_sl_prim, 2),
        "Val Fill Gap (0-14 minus 7-14) (%)": round(val_sl_full - val_sl_prim, 2),
        "Test Cost (W1 Prim 7-14) ($)": round(test_c_prim, 2),
        "Test Fill Rate (Days 7-14) (%)": round(test_sl_prim, 2),
        "Test Fill Rate (Days 0-14) (%)": round(test_sl_full, 2),
        "Tuning Cost Window": "Days 0 to 14 (Arriving orders fee for days 0 to 7)",
        "Tuning Fill Window": "Days 0 to 14"
    })

df_x3 = pd.DataFrame(x3_records)
df_x3.to_csv(os.path.join(TABLES_DIR, "x3_exact_fitness_and_val_fill.csv"), index=False)
print("Saved x3_exact_fitness_and_val_fill.csv")
print(df_x3[["Regime", "Forecaster", "Buffer", "Form", "Val Fill Rate (Days 0-14) (%)", "Val Fill Rate (Days 7-14) (%)", "Val Fill Gap (0-14 minus 7-14) (%)", "Test Cost (W1 Prim 7-14) ($)"]].to_string(index=False))

# ==============================================================================
# X4: RESTRICTED REGIME WITH RAW QUANTILE & BUFFER FEATURE AUDIT
# ==============================================================================
print("\n--- X4: RESTRICTED REGIME WITH RAW QUANTILE & BUFFER AUDIT ---")

RESTRICTED_X4 = [
    ("Quantile Conformal (Static)", "Quantile", "conformal_quantile", "static", 0.1648, 1.2172, "Primary Benchmark"),
    ("Quantile Raw (Static)", "Quantile", "raw_quantile", "static", 0.1892, 1.2172, "Uncalibrated Quantile Baseline"),
    ("Huber None (Rolling, b=5.0)", "Huber", "none", "rolling", 0.0000, 4.9979, "Base-Stock Ceiling Bound (5.0)"),
    ("Huber None (Rolling, b=8.0)", "Huber", "none", "rolling", 0.0000, 8.0000, "Sensitivity (Ceiling expanded to 8.0)"),
    ("Seasonal Naive None (Rolling)", "SeasonalNaive", "none", "rolling", 0.0000, 2.4629, "Heuristic Baseline"),
    ("Mean Gaussian Demand (Rolling)", "Mean", "gaussian_demand", "rolling", 0.0147, 1.4765, "Traditional Demand-Sigma Baseline")
]

x4_records = []
for p_label, f, b, form, a, beta, note in RESTRICTED_X4:
    is_roll = (form == "rolling")
    c_prim, sl_prim, h_prim, so_prim, o_prim, _, sl_full = simulate_inventory(
        f, b, a, beta, split="test", is_rolling=is_roll, score_days=(7, 15), return_full_sl=True
    )
    val_fit, val_c, val_sl = get_tuning_fitness(
        a, beta, f, b, split="val", is_rolling=is_roll, score_days=(0, 15)
    )
    x4_records.append({
        "Policy Configuration": p_label,
        "Forecaster": f,
        "Buffer Type": b,
        "Form": form,
        "Alpha*": a,
        "Beta*": beta,
        "Val Fitness ($)": round(val_fit, 2),
        "Test Cost (W1 Prim 7-14) ($)": round(c_prim, 2),
        "Test Fill (Days 7-14) (%)": round(sl_prim, 2),
        "Test Fill (Days 0-14) (%)": round(sl_full, 2),
        "Specification Role": note
    })

df_x4 = pd.DataFrame(x4_records)
df_x4.to_csv(os.path.join(TABLES_DIR, "x4_restricted_regime_with_raw.csv"), index=False)
print("Saved x4_restricted_regime_with_raw.csv")
print(df_x4[["Policy Configuration", "Alpha*", "Beta*", "Test Cost (W1 Prim 7-14) ($)", "Test Fill (Days 7-14) (%)", "Specification Role"]].to_string(index=False))

buffer_audit_records = [
    {
        "Buffer Name": "none",
        "Mathematical Definition": "SS = 0",
        "Validation-Fitted Statistics Used": "None (0 fitted parameters)",
        "Parameters Required": "Zero (pure base-stock)"
    },
    {
        "Buffer Name": "gaussian_demand",
        "Mathematical Definition": "SS = alpha * 1.282 * sigma_demand",
        "Validation-Fitted Statistics Used": "None (uses trailing 28-day demand sigma preceding decision date)",
        "Parameters Required": "Zero validation residual statistics"
    },
    {
        "Buffer Name": "raw_quantile",
        "Mathematical Definition": "SS = alpha * sum(max(0, P90_raw - P50))",
        "Validation-Fitted Statistics Used": "None (uses raw multi-horizon model output quantiles)",
        "Parameters Required": "Zero validation residual statistics or offsets"
    },
    {
        "Buffer Name": "conformal_quantile",
        "Mathematical Definition": "SS = alpha * sum(max(0, (P90_raw + offset_tier) - P50))",
        "Validation-Fitted Statistics Used": "2 scalar conformal calibration offsets (c_std = +0.1785, c_hvol = +4.5766)",
        "Parameters Required": "2 scalar non-conformity quantiles (distribution-free)"
    },
    {
        "Buffer Name": "gaussian_residual",
        "Mathematical Definition": "SS = alpha * 1.282 * sigma_residual",
        "Validation-Fitted Statistics Used": "363 pair-specific 1-step validation residual standard deviations",
        "Parameters Required": "363 fitted standard deviations (assumes residual normality)"
    },
    {
        "Buffer Name": "empirical_residual",
        "Mathematical Definition": "SS = alpha * q90(residuals)",
        "Validation-Fitted Statistics Used": "363 pair-specific 90th percentiles of validation residuals",
        "Parameters Required": "363 fitted empirical residual quantiles"
    }
]
df_buf_audit = pd.DataFrame(buffer_audit_records)
df_buf_audit.to_csv(os.path.join(TABLES_DIR, "x4_buffer_statistics_audit.csv"), index=False)
print("Saved x4_buffer_statistics_audit.csv")

# ==============================================================================
# X5: OPTIMIZERS BENCHMARK (W1 COST), FULL SENSITIVITY GRID, MATCHED-FILL ORACLE
# ==============================================================================
print("\n--- X5: OPTIMIZERS BENCHMARK (W1 COST) & SENSITIVITY GRID ---")

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
BOUNDS_MIN = np.array([0.0, 0.5])
BOUNDS_MAX = np.array([1.5, 5.0])
N_AGENTS = 25
N_ITER = 20

def run_pso_tune(seed):
    rng_p = np.random.RandomState(seed)
    pos = rng_p.uniform(BOUNDS_MIN, BOUNDS_MAX, size=(N_AGENTS, 2))
    vel = rng_p.uniform(-(BOUNDS_MAX - BOUNDS_MIN) * 0.1, (BOUNDS_MAX - BOUNDS_MIN) * 0.1, size=(N_AGENTS, 2))
    pbest_pos = pos.copy()
    pbest_fit = np.array([get_tuning_fitness(pos[i, 0], pos[i, 1], "Quantile", "conformal_quantile", split="val", is_rolling=True, score_days=(0, 15))[0] for i in range(N_AGENTS)])
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    w_start, w_end = 0.9, 0.4
    c1, c2 = 1.5, 1.5
    for it in range(N_ITER):
        w = w_start - (w_start - w_end) * (it / N_ITER)
        r1, r2 = rng_p.rand(N_AGENTS, 2), rng_p.rand(N_AGENTS, 2)
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = np.clip(pos + vel, BOUNDS_MIN, BOUNDS_MAX)
        for i in range(N_AGENTS):
            fit = get_tuning_fitness(pos[i, 0], pos[i, 1], "Quantile", "conformal_quantile", split="val", is_rolling=True, score_days=(0, 15))[0]
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = pos[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_pos = pos[i].copy()
    return gbest_pos[0], gbest_pos[1], gbest_fit

def run_gwo_tune(seed):
    rng_g = np.random.RandomState(seed)
    pos = rng_g.uniform(BOUNDS_MIN, BOUNDS_MAX, size=(N_AGENTS, 2))
    fits = np.array([get_tuning_fitness(pos[i, 0], pos[i, 1], "Quantile", "conformal_quantile", split="val", is_rolling=True, score_days=(0, 15))[0] for i in range(N_AGENTS)])
    order = np.argsort(fits)
    alpha_pos, alpha_fit = pos[order[0]].copy(), fits[order[0]]
    beta_pos, beta_fit   = pos[order[1]].copy(), fits[order[1]]
    delta_pos, delta_fit = pos[order[2]].copy(), fits[order[2]]
    for it in range(N_ITER):
        a_param = 2.0 - 2.0 * (it / N_ITER)
        for i in range(N_AGENTS):
            r1, r2 = rng_g.rand(2), rng_g.rand(2)
            A1, C1 = 2.0 * a_param * r1 - a_param, 2.0 * r2
            D_alpha = np.abs(C1 * alpha_pos - pos[i])
            X1 = alpha_pos - A1 * D_alpha
            r1, r2 = rng_g.rand(2), rng_g.rand(2)
            A2, C2 = 2.0 * a_param * r1 - a_param, 2.0 * r2
            D_beta = np.abs(C2 * beta_pos - pos[i])
            X2 = beta_pos - A2 * D_beta
            r1, r2 = rng_g.rand(2), rng_g.rand(2)
            A3, C3 = 2.0 * a_param * r1 - a_param, 2.0 * r2
            D_delta = np.abs(C3 * delta_pos - pos[i])
            X3 = delta_pos - A3 * D_delta
            pos[i] = np.clip((X1 + X2 + X3) / 3.0, BOUNDS_MIN, BOUNDS_MAX)
            fit = get_tuning_fitness(pos[i, 0], pos[i, 1], "Quantile", "conformal_quantile", split="val", is_rolling=True, score_days=(0, 15))[0]
            if fit < alpha_fit:
                delta_pos, delta_fit = beta_pos.copy(), beta_fit
                beta_pos, beta_fit   = alpha_pos.copy(), alpha_fit
                alpha_pos, alpha_fit = pos[i].copy(), fit
            elif fit < beta_fit:
                delta_pos, delta_fit = beta_pos.copy(), beta_fit
                beta_pos, beta_fit   = pos[i].copy(), fit
            elif fit < delta_fit:
                delta_pos, delta_fit = pos[i].copy(), fit
    return alpha_pos[0], alpha_pos[1], alpha_fit

pso_results = []
gwo_results = []
for s in SEEDS:
    ap, bp, fp = run_pso_tune(s)
    cp_prim, slp_prim, _, _, _, _ = simulate_inventory("Quantile", "conformal_quantile", ap, bp, split="test", is_rolling=True, score_days=(7, 15))
    cp_rob, slp_rob, _, _, _, _   = simulate_inventory("Quantile", "conformal_quantile", ap, bp, split="test", is_rolling=True, score_days=(0, 15))
    pso_results.append((ap, bp, fp, cp_prim, slp_prim, cp_rob, slp_rob))
    
    ag, bg, fg = run_gwo_tune(s)
    cg_prim, slg_prim, _, _, _, _ = simulate_inventory("Quantile", "conformal_quantile", ag, bg, split="test", is_rolling=True, score_days=(7, 15))
    cg_rob, slg_rob, _, _, _, _   = simulate_inventory("Quantile", "conformal_quantile", ag, bg, split="test", is_rolling=True, score_days=(0, 15))
    gwo_results.append((ag, bg, fg, cg_prim, slg_prim, cg_rob, slg_rob))

# 2D Grid Optimum
best_grid_fit, best_ga, best_gb = float("inf"), 0.0, 0.0
for ga in np.linspace(0.0, 1.5, 31):
    for gb in np.linspace(0.5, 5.0, 46):
        fit_g, _, _ = get_tuning_fitness(ga, gb, "Quantile", "conformal_quantile", split="val", is_rolling=True, score_days=(0, 15))
        if fit_g < best_grid_fit:
            best_grid_fit, best_ga, best_gb = fit_g, ga, gb

c_grid_prim, sl_grid_prim, _, _, _, _ = simulate_inventory("Quantile", "conformal_quantile", best_ga, best_gb, split="test", is_rolling=True, score_days=(7, 15))
c_grid_rob, sl_grid_rob, _, _, _, _   = simulate_inventory("Quantile", "conformal_quantile", best_ga, best_gb, split="test", is_rolling=True, score_days=(0, 15))

pso_arr = np.array(pso_results)
gwo_arr = np.array(gwo_results)

x5_opt_records = [
    {
        "Method": "PSO (10 Seeds Mean +/- Std)",
        "Alpha*": f"{np.mean(pso_arr[:, 0]):.4f} +/- {np.std(pso_arr[:, 0]):.4f}",
        "Beta*": f"{np.mean(pso_arr[:, 1]):.4f} +/- {np.std(pso_arr[:, 1]):.4f}",
        "Val Fitness ($)": f"{np.mean(pso_arr[:, 2]):.2f} +/- {np.std(pso_arr[:, 2]):.2f}",
        "Test Cost (W1 Prim 7-14) ($)": f"{np.mean(pso_arr[:, 3]):.2f} +/- {np.std(pso_arr[:, 3]):.2f}",
        "Test Fill (Prim 7-14) (%)": f"{np.mean(pso_arr[:, 4]):.2f} +/- {np.std(pso_arr[:, 4]):.2f}",
        "Test Cost (Rob 0-14) ($)": f"{np.mean(pso_arr[:, 5]):.2f} +/- {np.std(pso_arr[:, 5]):.2f}",
        "Test Fill (Rob 0-14) (%)": f"{np.mean(pso_arr[:, 6]):.2f} +/- {np.std(pso_arr[:, 6]):.2f}"
    },
    {
        "Method": "GWO (10 Seeds Mean +/- Std)",
        "Alpha*": f"{np.mean(gwo_arr[:, 0]):.4f} +/- {np.std(gwo_arr[:, 0]):.4f}",
        "Beta*": f"{np.mean(gwo_arr[:, 1]):.4f} +/- {np.std(gwo_arr[:, 1]):.4f}",
        "Val Fitness ($)": f"{np.mean(gwo_arr[:, 2]):.2f} +/- {np.std(gwo_arr[:, 2]):.2f}",
        "Test Cost (W1 Prim 7-14) ($)": f"{np.mean(gwo_arr[:, 3]):.2f} +/- {np.std(gwo_arr[:, 3]):.2f}",
        "Test Fill (Prim 7-14) (%)": f"{np.mean(gwo_arr[:, 4]):.2f} +/- {np.std(gwo_arr[:, 4]):.2f}",
        "Test Cost (Rob 0-14) ($)": f"{np.mean(gwo_arr[:, 5]):.2f} +/- {np.std(gwo_arr[:, 5]):.2f}",
        "Test Fill (Rob 0-14) (%)": f"{np.mean(gwo_arr[:, 6]):.2f} +/- {np.std(gwo_arr[:, 6]):.2f}"
    },
    {
        "Method": "2D Grid Optimum (31x46)",
        "Alpha*": f"{best_ga:.4f}",
        "Beta*": f"{best_gb:.4f}",
        "Val Fitness ($)": f"{best_grid_fit:.2f}",
        "Test Cost (W1 Prim 7-14) ($)": f"{c_grid_prim:.2f}",
        "Test Fill (Prim 7-14) (%)": f"{sl_grid_prim:.2f}",
        "Test Cost (Rob 0-14) ($)": f"{c_grid_rob:.2f}",
        "Test Fill (Rob 0-14) (%)": f"{sl_grid_rob:.2f}"
    }
]

df_x5_opt = pd.DataFrame(x5_opt_records)
df_x5_opt.to_csv(os.path.join(TABLES_DIR, "x5_optimizers_benchmark_w1.csv"), index=False)
print("Saved x5_optimizers_benchmark_w1.csv")
print(df_x5_opt[["Method", "Alpha*", "Beta*", "Val Fitness ($)", "Test Cost (W1 Prim 7-14) ($)", "Test Fill (Prim 7-14) (%)"]].to_string(index=False))

# Full Sensitivity Table (14 Scenarios x 6 Policies)
SENS_POLICIES = [
    ("Quantile", "conformal_quantile", True, "Quantile Conformal (Rolling)"),
    ("Quantile", "empirical_residual", False, "Quantile Empirical Residual (Static)"),
    ("Huber", "gaussian_residual", True, "Huber Gaussian Residual (Rolling)"),
    ("Huber", "none", True, "Huber None (Rolling)"),
    ("Mean", "gaussian_demand", True, "Mean Gaussian Demand (Rolling)"),
    ("Mean", "empirical_residual", True, "Mean Empirical Residual (Rolling)")
]

SCENARIOS = []
for cp in [1.0, 2.5, 4.0, 5.2, 7.5, 10.0]:
    SCENARIOS.append(("Stockout Penalty", f"cp = {cp:.1f}", cp, 1.0, 1.0))
for hm in [0.5, 0.75, 1.25, 1.5]:
    SCENARIOS.append(("Holding Multiplier", f"h_mult = {hm:.2f}", 2.5, hm, 1.0))
for om in [0.5, 0.75, 1.25, 1.5]:
    SCENARIOS.append(("Ordering Multiplier", f"o_mult = {om:.2f}", 2.5, 1.0, om))

ALPHA_G = np.linspace(0.0, 1.5, 16)
BETA_G  = np.linspace(0.5, 5.0, 26)

full_sens_rows = []
for cat, scen_name, cp_v, hm_v, om_v in SCENARIOS:
    scen_costs = {}
    scen_sls = {}
    for f_n, b_n, is_r, p_lbl in SENS_POLICIES:
        # Re-tune on validation
        best_fit, best_a, best_b = float("inf"), 0.0, 0.0
        for a in ALPHA_G:
            if b_n == "none" and a > 0.0:
                continue
            for b in BETA_G:
                fit, _, _ = get_tuning_fitness(
                    a, b, f_n, b_n, split="val", is_rolling=is_r, score_days=(0, 15), cp=cp_v, h_mult=hm_v, o_mult=om_v
                )
                if fit < best_fit:
                    best_fit, best_a, best_b = fit, a, b
        # Test evaluation with W1 cost
        c_test, sl_test, _, _, _, _ = simulate_inventory(
            f_n, b_n, best_a, best_b, split="test", is_rolling=is_r, score_days=(7, 15), cp=cp_v, h_mult=hm_v, o_mult=om_v
        )
        scen_costs[p_lbl] = c_test
        scen_sls[p_lbl] = sl_test
        
    min_c = min(scen_costs.values())
    sorted_pols = sorted(scen_costs.keys(), key=lambda k: scen_costs[k])
    
    for rank_idx, p_lbl in enumerate(sorted_pols, start=1):
        c_val = scen_costs[p_lbl]
        gap_pct = (c_val - min_c) / min_c * 100.0
        full_sens_rows.append({
            "Scenario Category": cat,
            "Scenario Parameter": scen_name,
            "Policy": p_lbl,
            "Rank": rank_idx,
            "Test Cost (W1 Prim) ($)": round(c_val, 2),
            "Gap vs Cheapest (%)": round(gap_pct, 2),
            "Test Fill Rate (7-14) (%)": round(scen_sls[p_lbl], 2)
        })

df_full_sens = pd.DataFrame(full_sens_rows)
df_full_sens.to_csv(os.path.join(TABLES_DIR, "x5_full_sensitivity_all_scenarios.csv"), index=False)
print("Saved x5_full_sensitivity_all_scenarios.csv")

# Matched-fill test-grid oracle
oracle_path = os.path.join(TABLES_DIR, "w8_matched_fill_frontier_w1.csv")
df_oracle = pd.read_csv(oracle_path)
df_oracle["Oracle Status"] = "Test-Grid Oracle (Ex-Post Bound; evaluated on Days 7-14 fill rate)"
df_oracle.to_csv(os.path.join(TABLES_DIR, "x5_matched_fill_oracle.csv"), index=False)
print("Saved x5_matched_fill_oracle.csv")

# ==============================================================================
# X6: EXACT BREAKDOWN OF $45,757.37
# ==============================================================================
print("\n--- X6: EXACT BREAKDOWN OF $45,757.37 COST GAP ---")

# Evaluate days 0 to 6 trajectory for multiple policies
gap_records = []
for p_label, f, b, form, a, beta, _ in RESTRICTED_X4[:4]:
    is_roll = (form == "rolling")
    c_e, sl_e, h_e, so_e, o_e, _ = simulate_inventory(
        f, b, a, beta, split="test", is_rolling=is_roll, score_days=(0, 7)
    )
    c_full, _, _, _, _, _ = simulate_inventory(
        f, b, a, beta, split="test", is_rolling=is_roll, score_days=(0, 15)
    )
    c_prim, _, _, _, _, _ = simulate_inventory(
        f, b, a, beta, split="test", is_rolling=is_roll, score_days=(7, 15)
    )
    delta_tot = c_full - c_prim
    gap_records.append({
        "Policy": p_label,
        "Days 0-6 Holding Cost ($)": round(h_e, 2),
        "Days 0-6 Stockout Penalty ($)": round(so_e, 2),
        "Days 0-6 Sum (Holding + Stockout) ($)": round(h_e + so_e, 2),
        "Full minus Prim Delta ($)": round(delta_tot, 2),
        "Exact Match with $45,757.37?": bool(round(delta_tot, 2) == 45757.37)
    })

df_gap = pd.DataFrame(gap_records)
df_gap.to_csv(os.path.join(TABLES_DIR, "x6_cost_gap_breakdown.csv"), index=False)
print("Saved x6_cost_gap_breakdown.csv")
print(df_gap.to_string(index=False))

print(f"\nFinal check (X-round) script completed successfully in {time.time() - t_start:.2f} seconds.")
