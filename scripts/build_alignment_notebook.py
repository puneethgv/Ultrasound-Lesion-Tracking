"""Generate ``notebooks/03_mask_alignment_check.ipynb``.

Cross-checks where the lesion annotations land, answering "are the masks applied correctly?".
For each case, one row of three panels:

  A  native RF pre-scan of the spline's acquisition/frame + GUI box + spline mapped into the RF grid
     with the inverse QuantUS warp — two independent annotations, compared in the same grid
  B  canonical RF fan image + mask (``register_case``, source='rf') — exactly what the model trains on
  C  raw DICOM + spline under the *old* assumption (spline pixels == DICOM-crop pixels), for reference

Key point: the spline pickles are QuantUS ROI exports drawn on QuantUS's own scan-conversion of the RF
(``Scan Name``), with ``Frame`` indexing the RF frames — not DICOM pixels (see usloc.labels.quantus).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "03_mask_alignment_check.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells: list = []

cells.append(md(
    "# Mask-alignment verification (QuantUS spline ↔ RF ↔ GUI box)\n"
    "\n"
    "The `FLL_ROI/*.pkl` splines are **QuantUS ROI exports** (`Spline X/Y`, `Scan Name`, "
    "`Phantom Name`, `Frame`). QuantUS draws ROIs on *its own* scan-conversion of the Clarius RF "
    "acquisition named by `Scan Name` (fixed settings: lines padded to 2928 samples, sector = "
    "2·radius degrees, start depth = depth/4, height 500 px), and `Frame` indexes the **RF frames**. "
    "They are **not** DICOM pixels — the earlier pipeline treated them as DICOM-crop pixels and "
    "indexed `0_0.dcm` with `Frame`, which misplaced the masks.\n"
    "\n"
    "Fix (`usloc.labels.quantus` + `register_case(source='rf')`): invert the QuantUS warp to put the "
    "spline in the RF pre-scan grid `(line, sample)` — the same grid as the independent **GUI box** — "
    "then render image and mask from that RF frame with one shared scan-conversion grid.\n"
    "\n"
    "Per case: **A** RF pre-scan + GUI box + inverted spline · **B** canonical RF fan + mask (model "
    "input) · **C** DICOM + spline under the old assumption."
))

cells.append(code(
    "%matplotlib inline\n"
    "import glob, warnings; warnings.simplefilter('ignore')\n"
    "import numpy as np, matplotlib.pyplot as plt, pydicom\n"
    "from usloc import config as C\n"
    "from usloc.data.discover import find_case_dir, list_dicoms\n"
    "from usloc.io import read_dicom\n"
    "from usloc.recon import rf_to_bmode\n"
    "from usloc.labels import load_boxes, load_rf_case, rasterize_spline, register_case\n"
    "from usloc.deid import ultrasound_region\n"
    "from usloc.viz import show_gray, draw_box, overlay_mask\n"
    "plt.rcParams.update({'figure.dpi':110,'font.size':9,'axes.titlesize':8,'figure.facecolor':'white'})\n"
    "\n"
    "def gui_box(case, kind='large'):\n"
    "    bs = load_boxes(glob.glob(str(C.GUI_ROOT/'gui_*'/f'{case}.xlsx'))[0])\n"
    "    return next((b for b in bs if b.roi_kind==kind), bs[0])\n"
))

cells.append(code(
    "def alignment_row(case):\n"
    "    rc = load_rf_case(case)                                 # spline's own RF acquisition\n"
    "    fr = int(np.clip(rc.spline.frame, 0, rc.n_frames-1))    # Frame indexes RF frames\n"
    "    prescan = rf_to_bmode(rc.rf[fr].astype(np.float32), axis=1)   # (lines, samples)\n"
    "    box = gui_box(case); h0,v0,h1,v1 = box.xyxy             # h = sample, v = line\n"
    "    s = register_case(case)                                 # canonical RF fan + aligned mask\n"
    "    # old assumption, for reference: spline pixels == DICOM-crop pixels, Frame indexes 0_0.dcm\n"
    "    d = [p for p in list_dicoms(find_case_dir(case)) if p.name.startswith('0_0')][0]\n"
    "    _, frames = read_dicom(d, deidentify=True)\n"
    "    reg = ultrasound_region(pydicom.dcmread(str(d))); sf = min(rc.spline.frame, frames.shape[0]-1)\n"
    "    old = rasterize_spline(np.asarray(rc.spline.x)+reg.x0, np.asarray(rc.spline.y)+reg.y0, frames.shape[1:])\n"
    "\n"
    "    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))\n"
    "    show_gray(ax[0], prescan.T, f'{case}  A: RF pre-scan (frame {fr}) — box vs inverted spline')\n"
    "    draw_box(ax[0], (v0,h0,v1,h1), '#00e5ff', 'GUI box')\n"
    "    pl = rc.prescan_polygon\n"
    "    ax[0].plot(np.r_[pl[:,0], pl[0,0]], np.r_[pl[:,1], pl[0,1]], '-', color='#ff00c8', lw=1.4)\n"
    "    ax[1].imshow(overlay_mask(s.image, s.mask, (255,0,200), 0.45)); ax[1].set_xticks([]); ax[1].set_yticks([])\n"
    "    ax[1].set_title('B: canonical RF fan + mask (model input)')\n"
    "    ax[2].imshow(overlay_mask(frames[sf], old, (255,160,0), 0.5), aspect='auto'); ax[2].set_xticks([]); ax[2].set_yticks([])\n"
    "    ax[2].set_title(f'C: DICOM + OLD mapping (frame {sf}) — for reference')\n"
    "    plt.tight_layout(); plt.show()\n"
    "\n"
    "cases = ['CEUS017', 'CEUS003', 'UKHCEUS003', 'UKDCEUS029', 'CEUS014']\n"
    "for c in cases:\n"
    "    try:\n"
    "        alignment_row(c)\n"
    "    except Exception as e:\n"
    "        print(f'{c}: skipped ({type(e).__name__}: {e})')\n"
))

cells.append(md(
    "## How to read this\n"
    "- **A** is the decisive check: the GUI box and the spline are independent annotations; after "
    "inverting the QuantUS warp both sit in the same RF grid and should outline the same lesion. "
    "Run `python scripts/validate_alignment.py --gallery 24` for the numbers over **all** spline cases "
    "(IoU per case, flagged outliers).\n"
    "- **B** is what the model now trains on: image and mask are rendered from the same RF frame "
    "through one shared grid, so they are aligned by construction (the fan geometry itself is "
    "approximate, which affects realism, not alignment).\n"
    "- **C** shows the previous mapping. Wherever its mask misses the lesion seen in B, the old "
    "training labels were wrong — retrain nnU-Net / YOLO / the classifier on the re-exported data.\n"
    "- Raw `*_rf.raw` files are now parsed with their Clarius header + per-frame timestamps "
    "(`read_clarius_raw`); the old flat read shifted frame *k* by 10 + 4k samples."
))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python (usloc)", "language": "python", "name": "usloc"}
nb.metadata["language_info"] = {"name": "python"}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB_PATH))
print("wrote", NB_PATH)
