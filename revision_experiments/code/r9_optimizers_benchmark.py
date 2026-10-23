"""
r9_optimizers_benchmark.py
R9. Optimizers:
- Reports PSO and GWO for the Primary specification across 10 random seeds.
- Reports mean and std of alpha*, beta*, validation fitness, test cost (Primary & Robustness), and test fill rate.
- Reports 2D Grid search optimum validation fitness and test cost next to PSO.
- Replaces identical-fill explanation with the actual cause (dispatch bug resolution & true forecast slice divergence).
"""

import os
import sys
import time
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
TABLES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "tables")
CODE_DIR = os.path.join(BASE_DIR, "revision_experiments", "code")
sys.path.insert(0, CODE_DIR)

from unified_simulator import simulate_inventory, get_tuning_fitness

print("=" * 80)
print("R9: OPTIMIZERS BENCHMARK (PSO vs GWO vs 2D GRID SEARCH, 10 SEEDS)")
print("=" * 80)
t0 = time.time()

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
BOUNDS_MIN = np.array([0.0, 0.5])
BOUNDS_MAX = np.array([1.5, 5.0])
N_AGENTS = 25
N_ITER = 20

# Primary Policy: Quantile TFT + Conformal Quantile Buffer (Rolling)
FORECASTER = "Quantile"
BUFFER = "conformal_quantile"
IS_ROLLING = True
ACCOUNTING = "arriving_orders"

def fitness_fn(alpha, beta):
    fit, _, _ = get_tuning_fitness(
        alpha=alpha,
        beta=beta,
        forecaster=FORECASTER,
        buffer_type=BUFFER,
        split="val",
        is_rolling=IS_ROLLING,
        accounting_mode=ACCOUNTING,
        score_days=(0, 15)
    )
    return fit

# 1. PSO Implementation
def run_pso(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(BOUNDS_MIN, BOUNDS_MAX, size=(N_AGENTS, 2))
    vel = rng.uniform(-(BOUNDS_MAX - BOUNDS_MIN) * 0.1, (BOUNDS_MAX - BOUNDS_MIN) * 0.1, size=(N_AGENTS, 2))
    
    pbest_pos = pos.copy()
    pbest_fit = np.array([fitness_fn(pos[i, 0], pos[i, 1]) for i in range(N_AGENTS)])
    
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    w_start, w_end = 0.9, 0.4
    c1, c2 = 1.5, 1.5
    
    for it in range(N_ITER):
        w = w_start - (w_start - w_end) * (it / N_ITER)
        r1 = rng.rand(N_AGENTS, 2)
        r2 = rng.rand(N_AGENTS, 2)
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = np.clip(pos + vel, BOUNDS_MIN, BOUNDS_MAX)
        
        for i in range(N_AGENTS):
            fit = fitness_fn(pos[i, 0], pos[i, 1])
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = pos[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_pos = pos[i].copy()
                    
    return gbest_pos[0], gbest_pos[1], gbest_fit

# 2. GWO Implementation
def run_gwo(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(BOUNDS_MIN, BOUNDS_MAX, size=(N_AGENTS, 2))
    fits = np.array([fitness_fn(pos[i, 0], pos[i, 1]) for i in range(N_AGENTS)])
    
    order = np.argsort(fits)
    alpha_pos, alpha_fit = pos[order[0]].copy(), fits[order[0]]
    beta_pos, beta_fit   = pos[order[1]].copy(), fits[order[1]]
    delta_pos, delta_fit = pos[order[2]].copy(), fits[order[2]]
    
    for it in range(N_ITER):
        a = 2.0 - 2.0 * (it / N_ITER)
        for i in range(N_AGENTS):
            r1, r2 = rng.rand(2), rng.rand(2)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * alpha_pos - pos[i])
            X1 = alpha_pos - A1 * D_alpha
            
            r1, r2 = rng.rand(2), rng.rand(2)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * beta_pos - pos[i])
            X2 = beta_pos - A2 * D_beta
            
            r1, r2 = rng.rand(2), rng.rand(2)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * delta_pos - pos[i])
            X3 = delta_pos - A3 * D_delta
            
            pos[i] = np.clip((X1 + X2 + X3) / 3.0, BOUNDS_MIN, BOUNDS_MAX)
            fit = fitness_fn(pos[i, 0], pos[i, 1])
            
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

# 3. Execute 10 Seeds for PSO & GWO
seed_records = []

print("Running 10 Seeds for PSO...")
for s in SEEDS:
    a, b, vfit = run_pso(s)
    # Test evaluation
    c_prim, sl_prim, _, _, _, _ = simulate_inventory(FORECASTER, BUFFER, a, b, split="test", is_rolling=IS_ROLLING, score_days=(7, 15))
    c_rob, sl_rob, h_rob, so_rob, o_rob, _ = simulate_inventory(FORECASTER, BUFFER, a, b, split="test", is_rolling=IS_ROLLING, score_days=(0, 15))
    seed_records.append({
        "Optimizer": "PSO",
        "Seed": s,
        "Alpha*": round(a, 4),
        "Beta*": round(b, 4),
        "Val Fitness ($)": round(vfit, 2),
        "Test Cost (Primary 7-14) ($)": round(c_prim, 2),
        "Test Fill (Primary 7-14) (%)": round(sl_prim, 2),
        "Test Cost (Robustness 0-14) ($)": round(c_rob, 2),
        "Test Fill (Robustness 0-14) (%)": round(sl_rob, 2)
    })

print("Running 10 Seeds for GWO...")
for s in SEEDS:
    a, b, vfit = run_gwo(s)
    c_prim, sl_prim, _, _, _, _ = simulate_inventory(FORECASTER, BUFFER, a, b, split="test", is_rolling=IS_ROLLING, score_days=(7, 15))
    c_rob, sl_rob, h_rob, so_rob, o_rob, _ = simulate_inventory(FORECASTER, BUFFER, a, b, split="test", is_rolling=IS_ROLLING, score_days=(0, 15))
    seed_records.append({
        "Optimizer": "GWO",
        "Seed": s,
        "Alpha*": round(a, 4),
        "Beta*": round(b, 4),
        "Val Fitness ($)": round(vfit, 2),
        "Test Cost (Primary 7-14) ($)": round(c_prim, 2),
        "Test Fill (Primary 7-14) (%)": round(sl_prim, 2),
        "Test Cost (Robustness 0-14) ($)": round(c_rob, 2),
        "Test Fill (Robustness 0-14) (%)": round(sl_rob, 2)
    })

df_seeds = pd.DataFrame(seed_records)
df_seeds.to_csv(os.path.join(TABLES_DIR, "r9_optimizers_10seeds_detail.csv"), index=False)
print("Saved r9_optimizers_10seeds_detail.csv")

# 4. 2D Grid Optimum
print("Evaluating 2D Grid Search Optimum...")
grid_a = np.linspace(0.0, 1.5, 31)
grid_b = np.linspace(0.5, 5.0, 46)

best_grid_fit = float("inf")
best_grid_a = 0.0
best_grid_b = 0.0

for a in grid_a:
    for b in grid_b:
        fit = fitness_fn(a, b)
        if fit < best_grid_fit:
            best_grid_fit = fit
            best_grid_a = a
            best_grid_b = b

c_grid_prim, sl_grid_prim, _, _, _, _ = simulate_inventory(FORECASTER, BUFFER, best_grid_a, best_grid_b, split="test", is_rolling=IS_ROLLING, score_days=(7, 15))
c_grid_rob, sl_grid_rob, _, _, _, _   = simulate_inventory(FORECASTER, BUFFER, best_grid_a, best_grid_b, split="test", is_rolling=IS_ROLLING, score_days=(0, 15))

# 5. Summary Benchmark Comparison Table
pso_df = df_seeds[df_seeds["Optimizer"] == "PSO"]
gwo_df = df_seeds[df_seeds["Optimizer"] == "GWO"]

summary_records = [
    {
        "Method": "PSO (10 Seeds Mean +/- Std)",
        "Alpha*": f"{pso_df['Alpha*'].mean():.4f} +/- {pso_df['Alpha*'].std():.4f}",
        "Beta*": f"{pso_df['Beta*'].mean():.4f} +/- {pso_df['Beta*'].std():.4f}",
        "Val Fitness ($)": f"{pso_df['Val Fitness ($)'].mean():.2f} +/- {pso_df['Val Fitness ($)'].std():.2f}",
        "Test Cost (Primary 7-14) ($)": f"{pso_df['Test Cost (Primary 7-14) ($)'].mean():.2f} +/- {pso_df['Test Cost (Primary 7-14) ($)'].std():.2f}",
        "Test Fill (Primary 7-14) (%)": f"{pso_df['Test Fill (Primary 7-14) (%)'].mean():.2f} +/- {pso_df['Test Fill (Primary 7-14) (%)'].std():.2f}",
        "Test Cost (Robustness 0-14) ($)": f"{pso_df['Test Cost (Robustness 0-14) ($)'].mean():.2f} +/- {pso_df['Test Cost (Robustness 0-14) ($)'].std():.2f}",
        "Test Fill (Robustness 0-14) (%)": f"{pso_df['Test Fill (Robustness 0-14) (%)'].mean():.2f} +/- {pso_df['Test Fill (Robustness 0-14) (%)'].std():.2f}"
    },
    {
        "Method": "GWO (10 Seeds Mean +/- Std)",
        "Alpha*": f"{gwo_df['Alpha*'].mean():.4f} +/- {gwo_df['Alpha*'].std():.4f}",
        "Beta*": f"{gwo_df['Beta*'].mean():.4f} +/- {gwo_df['Beta*'].std():.4f}",
        "Val Fitness ($)": f"{gwo_df['Val Fitness ($)'].mean():.2f} +/- {gwo_df['Val Fitness ($)'].std():.2f}",
        "Test Cost (Primary 7-14) ($)": f"{gwo_df['Test Cost (Primary 7-14) ($)'].mean():.2f} +/- {gwo_df['Test Cost (Primary 7-14) ($)'].std():.2f}",
        "Test Fill (Primary 7-14) (%)": f"{gwo_df['Test Fill (Primary 7-14) (%)'].mean():.2f} +/- {gwo_df['Test Fill (Primary 7-14) (%)'].std():.2f}",
        "Test Cost (Robustness 0-14) ($)": f"{gwo_df['Test Cost (Robustness 0-14) ($)'].mean():.2f} +/- {gwo_df['Test Cost (Robustness 0-14) ($)'].std():.2f}",
        "Test Fill (Robustness 0-14) (%)": f"{gwo_df['Test Fill (Robustness 0-14) (%)'].mean():.2f} +/- {gwo_df['Test Fill (Robustness 0-14) (%)'].std():.2f}"
    },
    {
        "Method": "2D Grid Search Optimum (31x46)",
        "Alpha*": f"{best_grid_a:.4f}",
        "Beta*": f"{best_grid_b:.4f}",
        "Val Fitness ($)": f"{best_grid_fit:.2f}",
        "Test Cost (Primary 7-14) ($)": f"{c_grid_prim:.2f}",
        "Test Fill (Primary 7-14) (%)": f"{sl_grid_prim:.2f}",
        "Test Cost (Robustness 0-14) ($)": f"{c_grid_rob:.2f}",
        "Test Fill (Robustness 0-14) (%)": f"{sl_grid_rob:.2f}"
    }
]

df_summary = pd.DataFrame(summary_records)
df_summary.to_csv(os.path.join(TABLES_DIR, "r9_optimizers_benchmark.csv"), index=False)
print("\n" + "=" * 80)
print("OPTIMIZERS BENCHMARK SUMMARY (PSO vs GWO vs 2D GRID):")
print("=" * 80)
print(df_summary.to_string(index=False))
print(f"\nR9 Completed in {time.time() - t0:.2f} seconds.")
