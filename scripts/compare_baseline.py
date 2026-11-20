"""
Side-by-side page of new plots against the legacy baseline (left: baseline,
right: new), worst first by mean absolute pixel difference.

    python scripts/compare_baseline.py NEW_DIR [BASELINE_DIR]   # default data/legacy_baseline
    -> NEW_DIR/compare_baseline.html
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def difference(new, base):
    """Mean absolute RGB difference in [0, 1], baseline resized to the new image."""
    a = Image.open(new).convert("RGB")
    b = Image.open(base).convert("RGB").resize(a.size)
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean() / 255)


def main(new_dir, base_dir="data/legacy_baseline"):
    new_dir, base_dir = Path(new_dir), Path(base_dir)
    new, base = {f.name for f in new_dir.glob("*.png")}, {f.name for f in base_dir.glob("*.png")}
    rows = sorted(((difference(new_dir / n, base_dir / n), n) for n in new & base), reverse=True)
    rel = os.path.relpath(base_dir, new_dir)
    page = ["<!doctype html><meta charset=utf-8><title>New vs baseline</title>",
            "<style>body{font-family:sans-serif;margin:16px}.pair{display:flex;gap:8px}"
            ".pair img{width:50%;border:1px solid #ccc}</style>",
            f"<h1>New vs baseline: {len(rows)} pairs, worst first</h1>",
            f"<p>Only in baseline: {sorted(base - new) or 'none'}<br>Only in new: {sorted(new - base) or 'none'}</p>"]
    page += [f"<h3>{score:.4f} &nbsp; {n}</h3><div class=pair><img src='{rel}/{n}'><img src='{n}'></div>"
             for score, n in rows]
    (new_dir / "compare_baseline.html").write_text("\n".join(page))
    for score, n in rows[:15]:
        print(f"{score:.4f}  {n}")


if __name__ == "__main__":
    main(*sys.argv[1:])
