"""Small, dependency-light drawing helpers for B-mode images, boxes and masks."""

from __future__ import annotations

import numpy as np


def show_gray(ax, img, title: str | None = None, cmap: str = "gray") -> None:
    ax.imshow(np.asarray(img), cmap=cmap, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def draw_box(ax, xyxy, color: str = "#00e5ff", label: str | None = None, lw: float = 1.6) -> None:
    from matplotlib.patches import Rectangle

    x0, y0, x1, y1 = xyxy
    ax.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=lw)
    )
    if label:
        ax.text(x0, y0 - 3, label, color=color, fontsize=8, weight="bold")


def overlay_mask(gray, mask, color=(0, 229, 255), alpha: float = 0.4) -> np.ndarray:
    """Return an RGB image with ``mask`` tinted over the grayscale ``gray`` (both HxW)."""
    g = np.asarray(gray, dtype=np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    rgb = np.stack([g, g, g], axis=-1)
    m = np.asarray(mask).astype(bool)
    col = np.array(color, dtype=np.float32) / 255.0
    rgb[m] = (1 - alpha) * rgb[m] + alpha * col
    return np.clip(rgb, 0, 1)


def montage(frames, ncols: int = 8, pad: int = 2, fill: float = 0.0) -> np.ndarray:
    """Tile a stack of 2D frames ``(N, H, W)`` into a single image for quick inspection."""
    frames = np.asarray(frames, dtype=np.float32)
    n, h, w = frames.shape
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    canvas = np.full((nrows * (h + pad) - pad, ncols * (w + pad) - pad), fill, dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, ncols)
        canvas[r * (h + pad): r * (h + pad) + h, c * (w + pad): c * (w + pad) + w] = frames[i]
    return canvas
