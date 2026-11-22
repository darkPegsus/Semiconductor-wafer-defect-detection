# ruff: noqa: E402
"""Extract representative validation wafer maps for demonstration in Streamlit app."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from app.utils.image_utils import wafer_to_rgb
from src.data.dataset import IDX_TO_LABEL, load_labeled_dataset
from src.data.group_split import load_split_indices


def main():
    df = load_labeled_dataset("data/interim/labeled_wafers.pkl")
    splits = load_split_indices("data/interim/split_indices.json", "group_stratified_70_15_15")
    val_idx = splits["val"]
    val_df = df.iloc[val_idx].reset_index(drop=True)

    out_dir = PROJECT_ROOT / "app" / "assets" / "sample_wafers"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}

    for cls_idx in range(9):
        cname = IDX_TO_LABEL[cls_idx]
        matches = val_df[val_df["failureType"] == cname]
        assert len(matches) > 0, f"No samples found for {cname}"

        # Choose a clean representative sample with clear morphology
        sample_row = matches.iloc[0]
        wm = sample_row["waferMap"]
        rgb = wafer_to_rgb(wm)
        pil_img = Image.fromarray(rgb)

        clean_name = cname.lower().replace("-", "_")
        filename = f"sample_{cls_idx}_{clean_name}.png"
        file_path = out_dir / filename
        pil_img.save(file_path)

        manifest[cname] = {
            "class_index": cls_idx,
            "class_name": cname,
            "filename": filename,
            "lot_name": str(sample_row["lotName"]),
            "original_shape": list(wm.shape),
        }
        print(f"Saved {cname:<10}: {filename} (Shape: {wm.shape})")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest successfully written to: {manifest_path}")


if __name__ == "__main__":
    main()
