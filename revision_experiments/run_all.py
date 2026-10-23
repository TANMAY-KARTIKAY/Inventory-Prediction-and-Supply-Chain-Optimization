"""
run_all.py
Master standalone orchestration script for Round 3 (Final Experiment Round).
Executes the complete locked experimental suite end-to-end without deep model retraining.
Reproduces all numbers, tables, figures, and statistical tests reported in RESULTS_SUMMARY.md and DISCREPANCIES.md.
"""

import os
import sys
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "code")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

STEPS = [
    ("R1 & R2: Simulator Module Reconciliation & Scoring Invariance Audit", "r1_r2_reconciliation.py"),
    ("R3: Pre-Specified Validation Selection & Paired Bootstrap", "r3_prespecified_selection.py"),
    ("R4: Monotone Lower Envelope Matched-Fill Frontier", "r4_matched_fill_frontier.py"),
    ("R5: Cost Parameter Sensitivity Re-Optimization", "r5_cost_sensitivity.py"),
    ("R6: Forecast Horizon Metrics & Error Decomposition", "r6_forecast_metrics_recomputation.py"),
    ("R7 & R8: Population Representativeness & Conformal Evidence", "r7_r8_population_and_conformal.py"),
    ("R9: Optimizers 10-Seed Benchmark vs 2D Grid Optimum", "r9_optimizers_benchmark.py"),
    ("V1 to V6: Final Verification Suite", "v_final_verification.py"),
    ("Round V7 (Final): Fee Accounting, Mean Reconciliation, Bounds & Bootstraps Suite", "w_round_v7_experiments.py"),
    ("Final Check (X-Round): Complete Audit, Sensitivity & Verification Suite", "x_final_check_experiments.py")
]

def main():
    print("=" * 80)
    print("REVISION EXPERIMENTS: MASTER PIPELINE (ROUND V7 FINAL LOCKED SPECIFICATION)")
    print(f"Working Directory: {SCRIPT_DIR}")
    print(f"Python Executable: {sys.executable}")
    print("=" * 80)
    t_master_start = time.time()
    
    for idx, (title, script_name) in enumerate(STEPS, start=1):
        script_path = os.path.join(CODE_DIR, script_name)
        print(f"\n[{idx}/{len(STEPS)}] Executing {title} ({script_name})...")
        t_step_start = time.time()
        
        res = subprocess.run([sys.executable, script_path], cwd=SCRIPT_DIR, capture_output=True, text=True)
        
        if res.returncode != 0:
            print(f"ERROR executing {script_name}!")
            print(res.stderr)
            sys.exit(res.returncode)
        else:
            print(f"Completed {script_name} successfully in {time.time() - t_step_start:.2f}s.")
            for line in res.stdout.strip().splitlines()[-5:]:
                print(f"   | {line}")

    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS SUCCESSFULLY EXECUTED AND LOCKED IN {time.time() - t_master_start:.2f} SECONDS.")
    print("All outputs generated in revision_experiments/outputs/tables and revision_experiments/outputs/figures.")
    print("=" * 80)

if __name__ == "__main__":
    main()
