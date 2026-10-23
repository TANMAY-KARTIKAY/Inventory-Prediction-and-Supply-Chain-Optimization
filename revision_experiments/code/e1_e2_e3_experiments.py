"""
e1_e2_e3_experiments.py
Phase 2 Experiments E1, E2, E3:
- E1: Validation Policy Tuning (10 seeds each for all policies via PSO & GWO) & One-Shot Test Evaluation
- E2: Exhaustive 2D Grid Reference & Fitness Surface Landscape Mapping
- E3: Factorial Ablation (No SS vs Classical Gaussian vs Raw Quantile vs Conformal Quantile)

Loads simulation_inputs.npz directly for instant execution.
"""

import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t_start = time.time()

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
TABLES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("EXPERIMENTS E1, E2, E3: VALIDATION TUNING, GRID SEARCH & FACTORIAL ABLATION")
print("=" * 80)

# Load precomputed simulation matrices
sim_npz_path = os.path.join(TABLES_DIR, "simulation_inputs.npz")
print(f"Loading simulation inputs from: {sim_npz_path}")
sim_data = np.load(sim_npz_path)

val_demand_mat  = sim_data["val_demand_mat"]
val_p50_mat     = sim_data["val_p50_mat"]
val_p90_mat     = sim_data["val_p90_mat"]
val_p90_raw_mat = sim_data["val_p90_raw_mat"]
val_huber_mat   = sim_data["val_huber_mat"]
val_dbar_vec    = sim_data["val_dbar_vec"]
val_sigma_vec   = sim_data["val_sigma_vec"]
val_sig_res_vec = sim_data["val_sig_res_vec"]

test_demand_mat  = sim_data["test_demand_mat"]
test_p50_mat     = sim_data["test_p50_mat"]
test_p90_mat     = sim_data["test_p90_mat"]
test_p90_raw_mat = sim_data["test_p90_raw_mat"]
test_huber_mat   = sim_data["test_huber_mat"]
test_dbar_vec    = sim_data["test_dbar_vec"]
test_sigma_vec   = sim_data["test_sigma_vec"]
test_sig_res_vec = sim_data["val_sig_res_vec"]  # causal validation residuals

h_cost_vec = sim_data["h_cost_vec"]
N_PAIRS = len(h_cost_vec)

UNIT_HOLDING_DRY    = 0.10
UNIT_HOLDING_PERISH = 0.15
STOCKOUT_PENALTY    = 2.50
FIXED_ORDER_COST    = 8.00
LEAD_TIME_DAYS      = 7

# ==============================================================================
# FAST VECTORIZED SIMULATOR
# ==============================================================================
def fast_sim(alpha, beta, policy_name, split="val"):
    if split == "val":
        demand_mat = val_demand_mat
        p50_mat = val_p50_mat
        p90_mat = val_p90_mat
        p90_raw_mat = val_p90_raw_mat
        huber_mat = val_huber_mat
        dbar_vec = val_dbar_vec
        sigma_vec = val_sigma_vec
        sig_res_vec = val_sig_res_vec
    elif split == "test":
        demand_mat = test_demand_mat
        p50_mat = test_p50_mat
        p90_mat = test_p90_mat
        p90_raw_mat = test_p90_raw_mat
        huber_mat = test_huber_mat
        dbar_vec = test_dbar_vec
        sigma_vec = test_sigma_vec
        sig_res_vec = test_sig_res_vec
    else:
        raise ValueError(f"Unknown split: {split}")

    if policy_name in ["Mean-demand Gaussian", "Classical Gaussian"]:
        lead_exp = dbar_vec * LEAD_TIME_DAYS
        ss = alpha * 1.282 * sigma_vec
    elif policy_name in ["Huber TFT + Gaussian", "Huber TFT"]:
        lead_exp = np.sum(huber_mat[:, :LEAD_TIME_DAYS], axis=1)
        ss = alpha * 1.282 * sig_res_vec
    elif policy_name in ["Quantile TFT", "Quantile Conformal", "Quantile Default", "Quantile TFT + Conformal PSO"]:
        lead_exp = np.sum(p50_mat[:, :LEAD_TIME_DAYS], axis=1)
        ss = alpha * np.sum(np.maximum(0.0, p90_mat[:, :LEAD_TIME_DAYS] - p50_mat[:, :LEAD_TIME_DAYS]), axis=1)
    elif policy_name == "Quantile Raw":
        lead_exp = np.sum(p50_mat[:, :LEAD_TIME_DAYS], axis=1)
        ss = alpha * np.sum(np.maximum(0.0, p90_raw_mat[:, :LEAD_TIME_DAYS] - p50_mat[:, :LEAD_TIME_DAYS]), axis=1)
    elif policy_name in ["No Safety Stock", "No SS"]:
        lead_exp = np.sum(p50_mat[:, :LEAD_TIME_DAYS], axis=1)
        ss = np.zeros(N_PAIRS, dtype=np.float64)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    s = lead_exp + ss
    S = s + (beta * dbar_vec)

    # Standardized Causal Initial Inventory
    on_hand = 0.70 * LEAD_TIME_DAYS * dbar_vec
    pipeline = np.zeros((N_PAIRS, LEAD_TIME_DAYS), dtype=np.float64)

    tot_hold = 0.0
    tot_stockout = 0.0
    tot_order = 0.0
    tot_sales = 0.0
    tot_demand = float(np.sum(demand_mat))

    for t in range(15):
        d_t = demand_mat[:, t]
        arrivals = pipeline[:, 0]
        on_hand += arrivals

        sales = np.minimum(on_hand, d_t)
        stockout = np.maximum(0.0, d_t - on_hand)
        on_hand = np.maximum(0.0, on_hand - d_t)

        tot_sales += float(np.sum(sales))
        tot_stockout += float(np.sum(stockout * STOCKOUT_PENALTY))
        tot_hold += float(np.sum(on_hand * h_cost_vec))

        inv_pos = on_hand + np.sum(pipeline[:, 1:], axis=1)
        order_mask = (inv_pos <= s)
        order_qty = np.where(order_mask, np.maximum(0.0, S - inv_pos), 0.0)

        tot_order += float(np.sum(order_mask) * FIXED_ORDER_COST)

        pipeline[:, :-1] = pipeline[:, 1:]
        pipeline[:, -1] = order_qty

    sl = (tot_sales / tot_demand) * 100.0 if tot_demand > 0 else 100.0
    tot_cost = tot_hold + tot_stockout + tot_order

    return tot_cost, sl, tot_hold, tot_stockout, tot_order

def val_objective(position, policy_name):
    alpha_val, beta_val = float(position[0]), float(position[1])
    tot_cost, sl, h_c, s_c, o_c = fast_sim(alpha_val, beta_val, policy_name, split="val")
    penalty = 0.0
    if sl < 95.0:
        penalty = 50000.0 * ((95.0 - sl) ** 2)
    fitness = tot_cost + penalty
    return fitness, tot_cost, sl, h_c, s_c, o_c

# ==============================================================================
# OPTIMIZERS: PSO & GWO
# ==============================================================================
BOUNDS_MIN = np.array([0.0, 0.5])
BOUNDS_MAX = np.array([1.5, 5.0])
N_AGENTS = 25
N_ITER = 20

def run_pso(policy_name, seed):
    np.random.seed(seed)
    W, C1, C2 = 0.72, 1.49, 1.49
    pos = np.random.uniform(BOUNDS_MIN, BOUNDS_MAX, (N_AGENTS, 2))
    vel = np.zeros((N_AGENTS, 2))

    pbest_pos = pos.copy()
    pbest_val = np.full(N_AGENTS, float("inf"))

    gbest_pos = None
    gbest_val = float("inf")
    gbest_cost = None
    gbest_sl = None

    history = []

    for it in range(N_ITER):
        for i in range(N_AGENTS):
            fit, cost, sl, _, _, _ = val_objective(pos[i], policy_name)
            if fit < pbest_val[i]:
                pbest_val[i] = fit
                pbest_pos[i] = pos[i].copy()
            if fit < gbest_val:
                gbest_val = fit
                gbest_pos = pos[i].copy()
                gbest_cost = cost
                gbest_sl = sl

        history.append(gbest_cost)
        r1 = np.random.uniform(0, 1, (N_AGENTS, 2))
        r2 = np.random.uniform(0, 1, (N_AGENTS, 2))
        vel = W * vel + C1 * r1 * (pbest_pos - pos) + C2 * r2 * (gbest_pos - pos)
        pos = np.clip(pos + vel, BOUNDS_MIN, BOUNDS_MAX)

    return {
        "alpha": gbest_pos[0],
        "beta": gbest_pos[1],
        "val_fitness": gbest_val,
        "val_cost": gbest_cost,
        "val_sl": gbest_sl,
        "history": history
    }

def run_gwo(policy_name, seed):
    np.random.seed(seed)
    pos = np.random.uniform(BOUNDS_MIN, BOUNDS_MAX, (N_AGENTS, 2))

    alpha_pos, beta_pos, delta_pos = None, None, None
    alpha_fit, beta_fit, delta_fit = float("inf"), float("inf"), float("inf")
    alpha_cost, alpha_sl = None, None

    for i in range(N_AGENTS):
        fit, cost, sl, _, _, _ = val_objective(pos[i], policy_name)
        if fit < alpha_fit:
            delta_fit, delta_pos = beta_fit, copy.copy(beta_pos)
            beta_fit, beta_pos = alpha_fit, copy.copy(alpha_pos)
            alpha_fit, alpha_pos = fit, pos[i].copy()
            alpha_cost, alpha_sl = cost, sl
        elif fit < beta_fit:
            delta_fit, delta_pos = beta_fit, copy.copy(beta_pos)
            beta_fit, beta_pos = fit, pos[i].copy()
        elif fit < delta_fit:
            delta_fit, delta_pos = fit, pos[i].copy()

    history = []

    for it in range(1, N_ITER + 1):
        a = 2.0 - it * (2.0 / N_ITER)
        for i in range(N_AGENTS):
            r1, r2 = np.random.rand(2), np.random.rand(2)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - pos[i])
            X1 = alpha_pos - A1 * D_alpha

            r1, r2 = np.random.rand(2), np.random.rand(2)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = np.abs(C2 * beta_pos - pos[i])
            X2 = beta_pos - A2 * D_beta

            r1, r2 = np.random.rand(2), np.random.rand(2)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = np.abs(C3 * delta_pos - pos[i])
            X3 = delta_pos - A3 * D_delta

            new_pos = np.clip((X1 + X2 + X3) / 3.0, BOUNDS_MIN, BOUNDS_MAX)
            pos[i] = new_pos

            fit, cost, sl, _, _, _ = val_objective(pos[i], policy_name)
            if fit < alpha_fit:
                alpha_fit, alpha_pos = fit, pos[i].copy()
                alpha_cost, alpha_sl = cost, sl
            elif fit < beta_fit:
                beta_fit, beta_pos = fit, pos[i].copy()
            elif fit < delta_fit:
                delta_fit, delta_pos = fit, pos[i].copy()

        history.append(alpha_cost)

    return {
        "alpha": alpha_pos[0],
        "beta": alpha_pos[1],
        "val_fitness": alpha_fit,
        "val_cost": alpha_cost,
        "val_sl": alpha_sl,
        "history": history
    }

# ==============================================================================
# EXPERIMENT E2: EXHAUSTIVE 2D GRID SEARCH (VALIDATION FITNESS SURFACE)
# ==============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT E2: EXHAUSTIVE 2D GRID SEARCH ON VALIDATION SPLIT")
print("=" * 80)

alpha_grid = np.linspace(0.0, 1.5, 31)  # step 0.05
beta_grid  = np.linspace(0.5, 5.0, 46)  # step 0.10
print(f"Grid dimensions: {len(alpha_grid)} x {len(beta_grid)} = {len(alpha_grid)*len(beta_grid)} evaluations")

grid_results = []
Z_fitness = np.zeros((len(beta_grid), len(alpha_grid)))
Z_cost    = np.zeros((len(beta_grid), len(alpha_grid)))
Z_sl      = np.zeros((len(beta_grid), len(alpha_grid)))

grid_best_fit = float("inf")
grid_best_param = None

for j, b_val in enumerate(beta_grid):
    for i, a_val in enumerate(alpha_grid):
        fit, c_val, sl_val, _, _, _ = val_objective([a_val, b_val], "Quantile TFT")
        Z_fitness[j, i] = fit
        Z_cost[j, i] = c_val
        Z_sl[j, i] = sl_val

        grid_results.append({
            "alpha": round(a_val, 4),
            "beta": round(b_val, 4),
            "val_cost": round(c_val, 2),
            "val_sl": round(sl_val, 2),
            "val_fitness": round(fit, 2)
        })

        if fit < grid_best_fit:
            grid_best_fit = fit
            grid_best_param = (a_val, b_val, c_val, sl_val)

df_grid = pd.DataFrame(grid_results)
df_grid.to_csv(os.path.join(TABLES_DIR, "e2_grid_search_summary.csv"), index=False)

print(f">> True Global Optimum on Validation Grid (Quantile TFT):")
print(f"   Alpha* = {grid_best_param[0]:.4f}, Beta* = {grid_best_param[1]:.4f}")
print(f"   Val Cost = ${grid_best_param[2]:,.2f}, Val Fill Rate = {grid_best_param[3]:.2f}%")

# Evaluate Grid Optimum on Test Split
t_cost_grid, t_sl_grid, t_h_grid, t_so_grid, t_ord_grid = fast_sim(grid_best_param[0], grid_best_param[1], "Quantile TFT", split="test")
print(f">> One-Shot Test Evaluation of Grid Optimum:")
print(f"   Test Cost = ${t_cost_grid:,.2f}, Test Fill Rate = {t_sl_grid:.2f}% (Holding: ${t_h_grid:,.2f}, SO: ${t_so_grid:,.2f}, Ord: ${t_ord_grid:,.2f})")

# Plot 2D Fitness Surface Landscape
fig, ax = plt.subplots(figsize=(10, 7))
A_mesh, B_mesh = np.meshgrid(alpha_grid, beta_grid)
cp = ax.contourf(A_mesh, B_mesh, Z_fitness, levels=40, cmap="viridis_r")
cbar = fig.colorbar(cp, ax=ax)
cbar.set_label("Validation Fitness ($ + Penalty)", fontsize=11)

cs = ax.contour(A_mesh, B_mesh, Z_sl, levels=[95.0], colors=["red"], linewidths=2.5, linestyles=["--"])
ax.clabel(cs, inline=True, fmt="SL = 95.0%%", fontsize=10)

ax.plot(grid_best_param[0], grid_best_param[1], marker="*", color="gold", markersize=18, markeredgecolor="black", label=f"True Grid Optimum ({grid_best_param[0]:.2f}, {grid_best_param[1]:.2f})")
ax.plot(0.10, 3.4879, marker="X", color="crimson", markersize=12, label="Legacy Test-Tuned Point (0.10, 3.49)")

ax.set_title("Validation Fitness Landscape: Quantile (s, S) Policy Search", fontsize=13, fontweight="bold")
ax.set_xlabel("Safety Stock Multiplier (Alpha)", fontsize=11)
ax.set_ylabel("Order-Up-To Cycle Coverage Days (Beta)", fontsize=11)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.3)
fig_path = os.path.join(FIGURES_DIR, "e2_fitness_surface_validation.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=200)
plt.close()
print(f"[OK] Saved Validation Fitness Landscape Figure to: {fig_path}")

# ==============================================================================
# EXPERIMENT E1: VALIDATION TUNING (10 SEEDS) & TEST EVALUATION
# ==============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT E1: TUNING EVERY POLICY ON VALIDATION (10 SEEDS EACH) & TEST EVALUATION")
print("=" * 80)

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
POLICIES = [
    "Quantile TFT",
    "Huber TFT + Gaussian",
    "Mean-demand Gaussian"
]

e1_seed_records = []
e1_summary_records = []

# Legacy Oracle Row (Condition 6)
e1_summary_records.append({
    "Policy": "Quantile TFT + Conformal PSO",
    "Optimizer": "PSO (Legacy Oracle)",
    "Alpha*": 0.1000,
    "Beta*": 3.4879,
    "Val Cost ($)": None,
    "Val Fill Rate (%)": None,
    "Test Cost ($)": 136378.10,
    "Test Holding ($)": 103576.04,
    "Test Stockout ($)": 21690.06,
    "Test Ordering ($)": 11112.00,
    "Test Fill Rate (%)": 96.41,
    "Misses 95% on Test?": False,
    "Evaluation Note": "oracle (uses test information)"
})

for p_name in POLICIES:
    for opt_name in ["PSO", "GWO"]:
        print(f"--- Tuning {p_name} using {opt_name} across {len(SEEDS)} seeds ---")
        best_seed_val_fit = float("inf")
        best_seed_res = None

        for seed in SEEDS:
            if opt_name == "PSO":
                res = run_pso(p_name, seed)
            else:
                res = run_gwo(p_name, seed)

            a_opt, b_opt = res["alpha"], res["beta"]
            v_cost, v_sl = res["val_cost"], res["val_sl"]

            t_cost, t_sl, t_h, t_so, t_ord = fast_sim(a_opt, b_opt, p_name, split="test")

            e1_seed_records.append({
                "Policy": p_name,
                "Optimizer": opt_name,
                "Seed": seed,
                "Alpha*": round(a_opt, 4),
                "Beta*": round(b_opt, 4),
                "Val Cost ($)": round(v_cost, 2),
                "Val Fill Rate (%)": round(v_sl, 2),
                "Test Cost ($)": round(t_cost, 2),
                "Test Fill Rate (%)": round(t_sl, 2),
                "Test Holding ($)": round(t_h, 2),
                "Test Stockout ($)": round(t_so, 2),
                "Test Ordering ($)": round(t_ord, 2),
                "Misses 95% on Test?": bool(t_sl < 95.0)
            })

            if res["val_fitness"] < best_seed_val_fit:
                best_seed_val_fit = res["val_fitness"]
                best_seed_res = (a_opt, b_opt, v_cost, v_sl, t_cost, t_sl, t_h, t_so, t_ord)

        e1_summary_records.append({
            "Policy": p_name,
            "Optimizer": opt_name,
            "Alpha*": round(best_seed_res[0], 4),
            "Beta*": round(best_seed_res[1], 4),
            "Val Cost ($)": round(best_seed_res[2], 2),
            "Val Fill Rate (%)": round(best_seed_res[3], 2),
            "Test Cost ($)": round(best_seed_res[4], 2),
            "Test Holding ($)": round(best_seed_res[6], 2),
            "Test Stockout ($)": round(best_seed_res[7], 2),
            "Test Ordering ($)": round(best_seed_res[8], 2),
            "Test Fill Rate (%)": round(best_seed_res[5], 2),
            "Misses 95% on Test?": bool(best_seed_res[5] < 95.0),
            "Evaluation Note": "fair_val_tuned (frozen on validation)"
        })

df_e1_seeds = pd.DataFrame(e1_seed_records)
df_e1_seeds.to_csv(os.path.join(TABLES_DIR, "e1_validation_tuned_seeds_detail.csv"), index=False)

df_e1_summary = pd.DataFrame(e1_summary_records)
df_e1_summary.to_csv(os.path.join(TABLES_DIR, "e1_validation_tuned_evaluation.csv"), index=False)

print("\n" + "=" * 80)
print("E1 VALIDATION-TUNED FINAL TEST BENCHMARK SUMMARY:")
print(df_e1_summary[["Policy", "Optimizer", "Alpha*", "Beta*", "Val Cost ($)", "Val Fill Rate (%)", "Test Cost ($)", "Test Fill Rate (%)", "Misses 95% on Test?", "Evaluation Note"]].to_string(index=False))
print("=" * 80)

# ==============================================================================
# EXPERIMENT E3: FACTORIAL ABLATION
# ==============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT E3: FACTORIAL ABLATION (VALIDATION TUNED -> TEST EVALUATED)")
print("=" * 80)

ABLATION_CONFIGS = [
    # 1. No Safety Stock (alpha=0, tune beta only)
    ("No Safety Stock", "No SS (alpha=0)", 0.0),
    # 2. Classical Mean-demand Gaussian
    ("Mean-demand Gaussian", "Gaussian SS (Mean)", None),
    # 3. Huber TFT + Gaussian
    ("Huber TFT + Gaussian", "Gaussian SS (Huber Residual)", None),
    # 4. Raw Quantile TFT (No conformal offset)
    ("Quantile Raw", "Quantile Empirical (Uncalibrated)", None),
    # 5. Full Conformal Quantile TFT
    ("Quantile TFT", "Quantile Conformal (Calibrated)", None),
]

e3_records = []

# Include Legacy Oracle Row (Condition 6)
e3_records.append({
    "Ablation Component": "Legacy Quantile PSO",
    "Forecaster": "Quantile TFT",
    "Safety Stock Buffer": "Conformal Quantile (Test Tuned)",
    "Alpha*": 0.1000,
    "Beta*": 3.4879,
    "Val Cost ($)": None,
    "Val Fill Rate (%)": None,
    "Test Cost ($)": 136378.10,
    "Test Holding ($)": 103576.04,
    "Test Stockout ($)": 21690.06,
    "Test Ordering ($)": 11112.00,
    "Test Fill Rate (%)": 96.41,
    "Misses 95% on Test?": False,
    "Evaluation Note": "oracle (uses test information)"
})

for pol_id, comp_name, fixed_a in ABLATION_CONFIGS:
    print(f"Running Factorial Ablation for: {comp_name}...")
    best_fit = float("inf")
    best_params = None

    for seed in SEEDS:
        np.random.seed(seed)
        if fixed_a is not None:
            # Optimize beta only
            pos_b = np.random.uniform(BOUNDS_MIN[1], BOUNDS_MAX[1], N_AGENTS)
            vel_b = np.zeros(N_AGENTS)
            pbest_pos_b = pos_b.copy()
            pbest_val_b = np.full(N_AGENTS, float("inf"))
            gbest_b = None
            gbest_val_b = float("inf")
            gbest_cost_b, gbest_sl_b = None, None

            for it in range(N_ITER):
                for i in range(N_AGENTS):
                    fit, c_val, sl_val, _, _, _ = val_objective([fixed_a, pos_b[i]], pol_id)
                    if fit < pbest_val_b[i]:
                        pbest_val_b[i] = fit
                        pbest_pos_b[i] = pos_b[i]
                    if fit < gbest_val_b:
                        gbest_val_b = fit
                        gbest_b = pos_b[i]
                        gbest_cost_b = c_val
                        gbest_sl_b = sl_val

                r1, r2 = np.random.uniform(0, 1, N_AGENTS), np.random.uniform(0, 1, N_AGENTS)
                vel_b = 0.72 * vel_b + 1.49 * r1 * (pbest_pos_b - pos_b) + 1.49 * r2 * (gbest_b - pos_b)
                pos_b = np.clip(pos_b + vel_b, BOUNDS_MIN[1], BOUNDS_MAX[1])

            if gbest_val_b < best_fit:
                best_fit = gbest_val_b
                best_params = (fixed_a, gbest_b, gbest_cost_b, gbest_sl_b)
        else:
            pso_res = run_pso(pol_id, seed)
            if pso_res["val_fitness"] < best_fit:
                best_fit = pso_res["val_fitness"]
                best_params = (pso_res["alpha"], pso_res["beta"], pso_res["val_cost"], pso_res["val_sl"])

    a_star, b_star, v_cost, v_sl = best_params
    t_cost, t_sl, t_h, t_so, t_ord = fast_sim(a_star, b_star, pol_id, split="test")

    if "Mean" in pol_id:
        f_name = "Mean-Demand Baseline"
    elif "Huber" in pol_id:
        f_name = "Huber TFT"
    else:
        f_name = "Quantile TFT"

    e3_records.append({
        "Ablation Component": comp_name,
        "Forecaster": f_name,
        "Safety Stock Buffer": comp_name,
        "Alpha*": round(a_star, 4),
        "Beta*": round(b_star, 4),
        "Val Cost ($)": round(v_cost, 2),
        "Val Fill Rate (%)": round(v_sl, 2),
        "Test Cost ($)": round(t_cost, 2),
        "Test Holding ($)": round(t_h, 2),
        "Test Stockout ($)": round(t_so, 2),
        "Test Ordering ($)": round(t_ord, 2),
        "Test Fill Rate (%)": round(t_sl, 2),
        "Misses 95% on Test?": bool(t_sl < 95.0),
        "Evaluation Note": "fair_val_tuned (frozen on validation)"
    })

df_e3 = pd.DataFrame(e3_records)
df_e3.to_csv(os.path.join(TABLES_DIR, "e3_factorial_ablation.csv"), index=False)

print("\n" + "=" * 80)
print("E3 FACTORIAL ABLATION FINAL EVALUATION SUMMARY:")
print(df_e3[["Ablation Component", "Forecaster", "Alpha*", "Beta*", "Val Cost ($)", "Val Fill Rate (%)", "Test Cost ($)", "Test Fill Rate (%)", "Misses 95% on Test?"]].to_string(index=False))
print("=" * 80)

elapsed = time.time() - t_start
print(f"\nAll experiments E1, E2, E3 completed successfully in {elapsed:.2f} seconds.")
