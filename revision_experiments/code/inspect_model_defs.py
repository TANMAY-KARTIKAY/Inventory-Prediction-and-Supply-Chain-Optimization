import json

def inspect_nb(nb_path, keywords):
    print(f"=== Inspecting {nb_path} ===")
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                src = "".join(cell["source"])
                for kw in keywords:
                    if kw in src:
                        print(f"Found '{kw}' in cell {i+1} (length {len(src)}):")
                        lines = src.split("\n")
                        matching_lines = [f"  L{ln}: {line}" for ln, line in enumerate(lines) if kw in line]
                        print("\n".join(matching_lines[:10]))
                        break
    except Exception as e:
        print(f"Error reading {nb_path}: {e}")

inspect_nb(r"C:\Users\tanma\OneDrive\Desktop\project(1)\TFT\Multi-Horizon Quantile Forecasting\Untitled12.ipynb", ["class ", "tft_quantile_champion", "Quantile", "predict", "inverse_transform"])
inspect_nb(r"C:\Users\tanma\OneDrive\Desktop\project(1)\MODLE TESTING\Untitled.ipynb", ["class ", "tft_best_champion", "predict", "inverse_transform"])
inspect_nb(r"C:\Users\tanma\OneDrive\Desktop\project(1)\TFT\Huber TFT imp\Untitled.ipynb", ["class ", "tft_best_champion", "Huber"])
