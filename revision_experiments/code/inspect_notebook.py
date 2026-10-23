import json
import os
import sys

nb_path = r"C:\Users\tanma\OneDrive\Desktop\project(1)\Multi-Horizon Quantile Forecasting(process)\MHQF optm.ipynb"
out_path = r"C:\Users\tanma\OneDrive\Desktop\project(1)\revision_experiments\logs\notebook_raw_cells.txt"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(out_path, "w", encoding="utf-8") as out_f:
    for idx, cell in enumerate(nb["cells"]):
        cell_type = cell["cell_type"]
        src = "".join(cell["source"])
        out_f.write(f"\n{'='*30} CELL {idx+1} ({cell_type}) {'='*30}\n")
        out_f.write(f"Total source length: {len(src)} chars, {len(cell['source'])} lines\n\n")
        out_f.write(src)
        out_f.write("\n\n")
        if "outputs" in cell:
            for out in cell["outputs"]:
                if out.get("output_type") == "stream":
                    out_f.write("--- OUTPUT STREAM (" + out.get("name", "") + ") ---\n")
                    out_f.write("".join(out.get("text", [])))
                    out_f.write("\n")
                elif "data" in out and "text/plain" in out["data"]:
                    out_f.write("--- OUTPUT PLAIN ---\n")
                    out_f.write("".join(out["data"]["text/plain"]))
                    out_f.write("\n")

print(f"Dumped notebook cells to {out_path}")
