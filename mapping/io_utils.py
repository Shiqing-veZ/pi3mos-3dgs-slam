from pathlib import Path
from typing import Iterable

import numpy as np


def save_ply(path_no_ext: str, points: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path_no_ext).with_suffix(".ply")
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors)
    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
