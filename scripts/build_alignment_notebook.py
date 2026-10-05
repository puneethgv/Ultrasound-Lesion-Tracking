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

cells.append(md(
    "## Aligned vs mis-aligned — full-cohort review galleries\n"
    "\n"
    "Every spline case is scored by the **IoU between the QuantUS-inverted spline and the independent "
    "GUI box** (both in the RF grid) and sorted into tiers. In each panel the **mask is magenta** and "
    "the GUI boxes are **red (large) / cyan (small)** on the canonical RF fan — so a clinician can see "
    "directly whether the mask sits on the lesion the box marks. **Green titles = aligned, red/amber = "
    "needs review.**\n"
    "\n"
    "- ✅ **Aligned, high confidence** — IoU ≥ 0.5\n"
    "- ✅ **Aligned, acceptable** — 0.3 ≤ IoU < 0.5\n"
    "- ⚠️ **Flagged, low IoU** (< 0.3, same acquisition) — mask may be misplaced, **or** box & spline "
    "mark *different* lesions (common in multifocal metastasis)\n"
    "- ⚠️ **Different acquisition** — spline's `Scan Name` ≠ `raw_0_0`, so the box is not comparable "
    "(boxes omitted; judge the mask alone)\n"
    "- ⚠️ **No RF** — spline exists but no `raw_*` acquisition found (drops out of the RF pipeline)"
))
cells.append(code(
    "import pandas as pd\n"
    "from pathlib import Path\n"
    "from usloc.data.cohort import build_cohort\n"
    "from usloc.labels.register import DEFAULT_SCAN\n"
    "\n"
    "def boxes_rf(case):\n"
    "    out = {}\n"
    "    for b in load_boxes(glob.glob(str(C.GUI_ROOT/'gui_*'/f'{case}.xlsx'))[0]):\n"
    "        if None in (b.h1, b.h2, b.v1, b.v2): continue\n"
    "        h0,v0,h1,v1 = b.xyxy; out.setdefault(b.roi_kind, (v0,h0,v1,h1))  # (line0,sample0,line1,sample1)\n"
    "    return out\n"
    "def _iou(a, b):\n"
    "    ix = max(0., min(a[2],b[2]) - max(a[0],b[0])); iy = max(0., min(a[3],b[3]) - max(a[1],b[1]))\n"
    "    it = ix*iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it\n"
    "    return it/u if u > 0 else 0.\n"
    "\n"
    "# score + cache a render frame for every spline case (loads RF cines — takes a few minutes)\n"
    "cache, rows = {}, []\n"
    "for _, row in build_cohort().query('has_spline').iterrows():\n"
    "    case = row['case']\n"
    "    try: rc = load_rf_case(case)\n"
    "    except Exception: rc = None\n"
    "    if rc is None or rc.prescan_polygon is None:\n"
    "        rows.append((case, row['class'], row['site'], 'no_rf', float('nan'))); continue\n"
    "    bx = boxes_rf(case); pl = rc.prescan_polygon\n"
    "    sb = (pl[:,0].min(), pl[:,1].min(), pl[:,0].max(), pl[:,1].max())\n"
    "    ref = bx.get('large', bx.get('small')); iou = _iou(sb, ref) if ref else float('nan')\n"
    "    if not bx: cat = 'no_box'\n"
    "    elif Path(rc.spline.scan_name).stem != DEFAULT_SCAN: cat = 'diff_acq'\n"
    "    elif iou >= 0.5: cat = 'aligned_high'\n"
    "    elif iou >= 0.3: cat = 'aligned_ok'\n"
    "    else: cat = 'flagged_low'\n"
    "    rows.append((case, row['class'], row['site'], cat, iou))\n"
    "    fr = int(np.clip(rc.spline.frame, 0, rc.n_frames-1))\n"
    "    cache[case] = (rc.frame_image(fr), rc.mask, rc.grid.forward, bx, iou, fr, rc.spline.scan_name)\n"
    "status = pd.DataFrame(rows, columns=['case','class','site','category','iou'])\n"
    "print(status['category'].value_counts().to_string())\n"
    "print('\\naligned (IoU>=0.3):', int(status.category.isin(['aligned_high','aligned_ok']).sum()),\n"
    "      '/ ', len(status), 'spline cases')\n"
))
cells.append(code(
    "def render_group(cases, title, show_boxes=True, color='black', ncol=5):\n"
    "    if not cases:\n"
    "        print('(none)'); return\n"
    "    nrow = int(np.ceil(len(cases)/ncol))\n"
    "    fig, ax = plt.subplots(nrow, ncol, figsize=(3.4*ncol, 3.0*nrow), squeeze=False)\n"
    "    for a in ax.ravel(): a.axis('off')\n"
    "    for a, case in zip(ax.ravel(), cases):\n"
    "        image, mask, forward, bx, iou, fr, scan = cache[case]\n"
    "        a.imshow(overlay_mask(image, mask, (255,0,200), 0.4)); a.axis('off')\n"
    "        if show_boxes:\n"
    "            for kind, c in (('large','#ff3d00'), ('small','#00e5ff')):\n"
    "                if kind in bx:\n"
    "                    l0,s0,l1,s1 = bx[kind]\n"
    "                    ls = np.r_[np.linspace(l0,l1,20), np.full(20,l1), np.linspace(l1,l0,20), np.full(20,l0)]\n"
    "                    ss = np.r_[np.full(20,s0), np.linspace(s0,s1,20), np.full(20,s1), np.linspace(s1,s0,20)]\n"
    "                    x,y = forward(ls, ss); a.plot(x, y, '-', color=c, lw=1.0)\n"
    "        tt = case + (f'  IoU={iou:.2f}' if iou == iou else '')\n"
    "        if not show_boxes: tt += f'  [{scan}]'\n"
    "        a.set_title(tt, fontsize=8, color=color)\n"
    "    fig.suptitle(title, fontweight='bold'); plt.tight_layout(); plt.show()\n"
    "\n"
    "def grp(cat, asc=False):\n"
    "    return list(status[status.category == cat].sort_values('iou', ascending=asc)['case'])\n"
))
cells.append(md("### ✅ Aligned — high confidence (IoU ≥ 0.5)"))
cells.append(code("render_group(grp('aligned_high'), 'ALIGNED — high confidence (IoU >= 0.5)', color='#1a7f37')"))
cells.append(md("### ✅ Aligned — acceptable (0.3 ≤ IoU < 0.5)"))
cells.append(code("render_group(grp('aligned_ok'), 'ALIGNED — acceptable (0.3 <= IoU < 0.5)', color='#1a7f37')"))
cells.append(md("### ⚠️ Flagged — low IoU (< 0.3), same acquisition — **please review**"))
cells.append(code("render_group(grp('flagged_low', asc=True), 'FLAGGED — low IoU (mask may be wrong, or a different lesion)', color='#d1242f')"))
cells.append(md("### ⚠️ Flagged — spline on a different acquisition than the boxes (box not comparable)"))
cells.append(code("render_group(grp('diff_acq'), 'FLAGGED — different acquisition than the GUI boxes', show_boxes=False, color='#bf8700')"))
cells.append(md("### ⚠️ No RF acquisition found (excluded from the RF pipeline)"))
cells.append(code("print('\\n'.join(status[status.category=='no_rf']['case']) or '(none)')"))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python (usloc)", "language": "python", "name": "usloc"}
nb.metadata["language_info"] = {"name": "python"}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB_PATH))
print("wrote", NB_PATH)
