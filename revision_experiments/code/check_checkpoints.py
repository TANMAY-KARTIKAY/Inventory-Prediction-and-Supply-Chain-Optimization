import os
import torch
import numpy as np

# Load training data to compute cardinalities
train_path = r"C:\Users\tanma\OneDrive\Desktop\project(1)\Multi-Horizon Quantile Forecasting(process)\sequences_train.npz"
tr = np.load(train_path, mmap_mode="r")

static_cols = [
    "store_nbr", "item_nbr", "family", "class",
    "perishable", "city", "state", "type", "cluster"
]

cat_categories = {}
cat_cardinalities = []
for col in static_cols:
    uniques = np.unique(tr[col])
    cat_categories[col] = uniques
    cat_cardinalities.append(len(uniques))
    print(f"  {col}: {len(uniques)} unique")

# Check Quantile checkpoint
q_ckpt_path = r"C:\Users\tanma\OneDrive\Desktop\project(1)\Multi-Horizon Quantile Forecasting(process)\tft_quantile_champion.pth"
print("\n=== Quantile Checkpoint ===")
q_ckpt = torch.load(q_ckpt_path, map_location="cpu", weights_only=False)
print("Keys:", list(q_ckpt.keys()))
if "model_state_dict" in q_ckpt:
    state_dict = q_ckpt["model_state_dict"]
else:
    state_dict = q_ckpt
for k in ["static_embeddings.0.weight", "forecast_head.weight", "encoder_lstm.weight_ih_l0"]:
    if k in state_dict:
        print(f"  {k}: shape {state_dict[k].shape}")

# Check Huber checkpoint
h_ckpt_path = r"C:\Users\tanma\OneDrive\Desktop\project(1)\MODLE TESTING\tft_best_champion.pth"
print("\n=== Huber Checkpoint ===")
h_ckpt = torch.load(h_ckpt_path, map_location="cpu", weights_only=False)
if isinstance(h_ckpt, dict):
    print("Keys:", list(h_ckpt.keys()))
    if "model_state_dict" in h_ckpt:
        h_state_dict = h_ckpt["model_state_dict"]
    else:
        h_state_dict = h_ckpt
else:
    h_state_dict = h_ckpt

for k in ["static_embeddings.0.weight", "forecast_head.weight", "encoder_lstm.weight_ih_l0"]:
    if k in h_state_dict:
        print(f"  {k}: shape {h_state_dict[k].shape}")
