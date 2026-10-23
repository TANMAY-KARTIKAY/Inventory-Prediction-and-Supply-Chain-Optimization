# Multi-Horizon Quantile Demand Forecasting & Conformalized Swarm Inventory Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: Locked](https://img.shields.io/badge/Reproducibility-Verified%20(<8%20min)-success.svg)]()

This repository contains the complete, reproducible research codebase, simulation-optimization framework, model checkpoints, and manuscript materials for:

> **"Multi-Horizon Quantile Demand Forecasting and Conformalized Swarm Inventory Optimization in Retail Supply Chains"**  
> *Tanmay Kartikay, et al.*

---

## Abstract & System Architecture

Modern retail supply chains operate under severe non-stationary demand volatility, intermittent consumption spikes, lead-time uncertainty, and asymmetric operational penalty structures. Traditional single-point forecasting combined with deterministic safety stock formulas fails to protect against tail-risk shortages while inflating inventory holding capital.

This project implements an end-to-end, distribution-free computational framework:
1. **Multi-Horizon Quantile Forecasting:** A customized Temporal Fusion Transformer (TFT) that predicts multi-step demand across nominal quantiles $q \in \{0.10, 0.50, 0.90\}$ over a 16-day operational planning horizon under a specialized multi-quantile pinball loss.
2. **Distribution-Free Conformal Calibration:** A split conformal prediction post-processor that adjusts quantile intervals against empirical validation residuals, guaranteeing nominal finite-sample coverage ($\ge 90\%$) under minimal exchangeability assumptions.
3. **Continuous $(s, S)$ Simulation-Optimization:** A realistic rolling-horizon inventory simulator executing continuous $(s, S)$ replenishment policies under deterministic lead times, batch ordering constraints, carrying costs, fixed replenishment fees, and non-linear stockout penalties.
4. **Metaheuristic Swarm Search:** Particle Swarm Optimization (PSO) and Grey Wolf Optimizer (GWO) algorithms that navigate non-convex, simulation-evaluated cost surfaces to identify Pareto-optimal $(s, S)$ policy parameters.

```
+----------------------------------------------------------------------------------------------------+
|                                    END-TO-END PIPELINE ARCHITECTURE                                |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Raw Multi-Store Retail Time Series]                                                              |
|                     |                                                                              |
|                     v                                                                              |
|  [Temporal Feature Engineering]  --> Static metadata, rolling lags, holiday/oil covariates         |
|                     |                                                                              |
|                     v                                                                              |
|  [Multi-Horizon Quantile TFT]   --> Predicts Q10, Q50, Q90 over 16-day lead horizon                |
|                     |                                                                              |
|                     v                                                                              |
|  [Conformal Calibration Layer]   --> Post-hoc residual conformity bounds (guaranteed 90% coverage)   |
|                     |                                                                              |
|                     v                                                                              |
|  [Swarm Simulation-Optimization]--> PSO & GWO swarm agents navigate 16-day inventory simulator    |
|                     |                                                                              |
|                     v                                                                              |
|  [Locked Policy Evaluation]      --> Optimal (s*, S*) policy minimizing total supply chain costs   |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## Key Experimental Results

All numbers and tables are audited, verified, and locked against empirical outputs in `revision_experiments/outputs/tables/`:

- **Forecasting Precision & Coverage:**
  - Quantile TFT outperforms standard LSTM, GRU, and PatchTST baselines in Weighted Quantile Loss (WQL) across all 16 forecast horizons.
  - Conformal calibration successfully recalibrates empirical test coverage from uncalibrated intervals to nominal **90.0%** coverage with an improved Winkler interval score.
- **Inventory Policy Economics:**
  - **22.8%** reduction in total inventory operating costs relative to conventional static $(s, S)$ heuristic benchmarks.
  - **71.4%** reduction in stockout occurrences across active retail series.
  - **Optimizer Dynamics:** GWO exhibits faster initial convergence within fewer function evaluations, while PSO attains marginally superior holding cost precision in steady-state exploitation.

---

## Repository Structure

```
.
├── revision_experiments/               # Master audited experiment suite
│   ├── run_all.py                      # Master reproduction orchestrator (<8 min run)
│   ├── code/                           # Standalone reproduction modules (R1-R9, V1-V6, W-V7, X)
│   │   ├── e0_fair_simulator.py        # Canonical rolling-horizon inventory simulator
│   │   ├── r1_r2_reconciliation.py     # Simulator audit & scoring invariance
│   │   ├── r3_prespecified_selection.py# Paired bootstrap & validation-locked selection
│   │   ├── r4_matched_fill_frontier.py # Monotone lower envelope matched-fill frontier
│   │   ├── r5_cost_sensitivity.py      # Penalty parameter re-optimization
│   │   ├── r6_forecast_metrics_recomputation.py # Horizon metrics & error decomposition
│   │   ├── r7_r8_population_and_conformal.py    # Population representativeness & conformal evidence
│   │   ├── r9_optimizers_benchmark.py  # 10-seed PSO/GWO benchmark vs 2D grid optimum
│   │   ├── w_round_v7_experiments.py   # Fee accounting, mean reconciliation & bootstrap suite
│   │   └── x_final_check_experiments.py# Master sensitivity & validation verification
│   ├── outputs/
│   │   ├── tables/                     # Ground-truth CSV tables (Tables 2-9 and appendices)
│   │   └── figures/                    # High-resolution convergence & frontier figures
│   ├── RESULTS_SUMMARY.md              # Exhaustive numerical ledger matching manuscript
│   └── DISCREPANCIES.md                # Audit trail of reconciled metrics
├── Multi-Horizon Quantile Forecasting(process)/
│   ├── tft_model.py                    # PyTorch Temporal Fusion Transformer architecture
│   ├── dataset.py                      # PyTorch sequence dataset & batch loaders
│   ├── tft_quantile_champion.pth       # Trained champion model weights (1.05 MB)
│   ├── target_sales_scaler.joblib      # Robust unit sales scaler
│   ├── sequences_test.npz              # Test split tensors (9.98 MB)
│   ├── sequences_validation.npz        # Validation split tensors (10.18 MB)
│   ├── selected_pairs.csv              # Active 363 store-item evaluation cohort
│   └── tft_quantile_forecasts.csv      # Out-of-sample quantile predictions
├── Program/                            # Data preparation & preprocessing scripts
│   ├── pipeline(1-5).py                # Subsampling, cleaning, and covariate generation
│   ├── prepare_chronological_splits(6).py # Time-series train/val/test splits
│   ├── scale_splits(7).py              # Feature scaling
│   └── generate_sequences_multivariate(8).py # 3D temporal sequence tensor generation
├── Phase_4_Inventory_Optimization/     # Policy optimization analysis & dashboards
├── paper/                              # Manuscript drafts, LaTeX source, and figures
│   ├── manuscript.md                   # Full research paper manuscript
│   └── manuscript_clean.docx           # Publication-ready submission document
├── requirements.txt                    # Python runtime dependencies
├── .gitignore                          # Git exclude rules for multi-gigabyte raw dumps
└── README.md                           # Project documentation
```

---

## Quickstart & Full Reproduction

### 1. Environment Installation

Ensure Python 3.10 or higher is installed:

```bash
git clone https://github.com/TANMAY-KARTIKAY/Inventory-Prediction-and-Supply-Chain-Optimization.git
cd Inventory-Prediction-and-Supply-Chain-Optimization

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Complete Experimental Suite (< 8 minutes)

The entire experimental suite—including optimizer benchmarks across 10 random seeds, conformal recalibration, matched-fill trade-off frontiers, and cost sensitivity re-optimizations—is executed with a single command:

```bash
python revision_experiments/run_all.py
```

This updates and locks all outputs in:
- `revision_experiments/outputs/tables/`
- `revision_experiments/outputs/figures/`

All generated outputs precisely reproduce Tables 2 through 9 and supplementary appendices in the paper.

---

## Data Availability & Training From Scratch

To allow immediate verification without downloading massive data, **all test and validation tensors, scalers, and trained champion weights are pre-packaged within this repository** (~35 MB total).

To retrain the complete TFT model from raw transaction records:
1. Download the raw CSV dataset from [Kaggle: Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data).
2. Place `train.csv`, `items.csv`, `stores.csv`, `oil.csv`, and `holidays_events.csv` in `RAW(1)/`.
3. Execute the preprocessing pipeline:
   ```bash
   python Program/pipeline(1-5).py
   python Program/prepare_chronological_splits(6).py
   python Program/scale_splits(7).py
   python Program/generate_sequences_multivariate(8).py
   ```
4. Train the Quantile Temporal Fusion Transformer:
   ```bash
   python "Multi-Horizon Quantile Forecasting(process)/build_full_thesis.py"
   ```

---

## Citation

If you use this codebase, models, or simulation-optimization framework in your research, please cite:

```bibtex
@article{kartikay2026retail,
  title={Multi-Horizon Quantile Demand Forecasting and Conformalized Swarm Inventory Optimization in Retail Supply Chains},
  author={Kartikay, Tanmay and Collaborators},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
