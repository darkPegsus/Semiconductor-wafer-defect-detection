"""Generate Markdown summary table for WM-811K roundtrip reconstruction audit."""

import json
from pathlib import Path

report_path = Path("tests/data/roundtrip_sources/roundtrip_report.json")
with open(report_path) as f:
    data = json.load(f)

print(f"Total test cases evaluated: {len(data)}\n")

# Group by pattern
patterns = ["Near-full", "Scratch", "Loc", "Center", "Edge-Ring", "none", "Donut", "Edge-Loc", "Random"]

print("| Defect Pattern | Palette | Original Grid | Reconstructed Grid | Total Cells | Exact Matches | Mismatches | Reconstruction Accuracy | Per-Category Mismatches (0 / 1 / 2) | Orientation Preserved |")
print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for p in patterns:
    p_rows = [r for r in data if r["pattern"] == p]
    for r in p_rows:
        pat = r["pattern"]
        pal = r["palette_name"].split(" (")[0]
        orig = f"{r['original_dimensions'][0]}×{r['original_dimensions'][1]}"
        recon = f"{r['reconstructed_dimensions'][0]}×{r['reconstructed_dimensions'][1]}"
        tot = f"{r['total_cells']:,}"
        match = f"{r['matching_cells']:,}"
        mism = f"{r['mismatched_cells']}"
        acc = f"{r['reconstruction_accuracy_pct']:.2f}%"
        c0 = r["per_category_mismatches"]["cat_0_exterior"]
        c1 = r["per_category_mismatches"]["cat_1_normal"]
        c2 = r["per_category_mismatches"]["cat_2_defect"]
        cats = f"0={c0}, 1={c1}, 2={c2}"
        orient = "✅ Preserved" if r["orientation_preserved"] else "❌ Failed"
        print(f"| **{pat}** | {pal} | {orig} | {recon} | {tot} | {match} | {mism} | **{acc}** | {cats} | {orient} |")
