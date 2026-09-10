"""Generate ``notebooks/02_training_method1_nnunet.ipynb``.

A "Training" notebook that documents Method 1 (nnU-Net) and visualizes:
  * the data that actually goes into the model, separated into TRAIN and VAL, with GT lesion overlays;
  * what comes straight out of the model — predicted segmentations vs ground truth, on the held-out
    Dresden val set and the Halle external test.

Reads the nnU-Net dataset / predictions written under ``data/derived/nnunet`` (regenerate them with
scripts/export_nnunet + nnUNetv2_train + scripts/eval_nnunet before executing this notebook).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "02_training_method1_nnunet.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells: list = []

cells.append(md(
    "# Training\n"
    "\n"
    "Localizing the focal liver lesion is framed as **dense segmentation** (lesion vs liver), which "
    "generalizes far better than box detection on these small, low-contrast lesions. Each method gets "
    "its own section; below is **Method 1: nnU-Net**."
))

cells.append(md(
    "## Method 1 — nnU-Net (2D lesion segmentation)\n"
    "\n"
    "[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) is a self-configuring segmentation framework — it "
    "inspects the dataset and picks the architecture, patch size, normalization and augmentation "
    "automatically. It is the standard strong baseline for medical image segmentation.\n"
    "\n"
    "**How we run it here**\n"
    "1. **Build examples** — for each patient we take the annotated `0_0.dcm` frame, de-identify it "
    "(keep only the ultrasound sector), and pair it with the lesion mask rasterized from the spline "
    "contour (`usloc.labels.register_case`). One frame per patient keeps nnU-Net's internal 5-fold CV "
    "patient-clean.\n"
    "2. **Export to nnU-Net format** (`usloc.datasets.export_nnunet`): Dresden → `imagesTr/labelsTr`, "
    "Halle → `imagesTs/labelsTs` (locked external test). Binary labels: `background=0, lesion=1`.\n"
    "3. **Plan & preprocess** — nnU-Net configures an **8-stage 2D U-Net** (patch 448×896, z-score "
    "normalization, batch 4).\n"
    "4. **Train** fold 0 for 100 epochs (`torch.compile` disabled for this GPU).\n"
    "5. **Evaluate** — nnU-Net's held-out Dresden val split, and inference on the Halle external test.\n"
    "\n"
    "**Results so far (fold 0):** peak validation Dice ≈ **0.16** (the model overfits after ~epoch 21 "
    "given only ~83 unique training images); **Halle external mean Dice ≈ 0.08, detection rate ≈ 17%** "
    "(Dice>0.1). Localization is genuinely data-limited — honest baseline numbers to improve on "
    "(5-fold ensemble, tighter/higher-res ROI, more data)."
))

cells.append(code(
    "%matplotlib inline\n"
    "import json, warnings; warnings.simplefilter('ignore')\n"
    "import numpy as np, cv2, matplotlib.pyplot as plt\n"
    "from usloc import config as C\n"
    "from usloc.viz import overlay_mask\n"
    "plt.rcParams.update({'figure.dpi':110,'font.size':9,'axes.titlesize':9,'figure.facecolor':'white'})\n"
    "\n"
    "NN = C.DERIVED/'nnunet'\n"
    "RAW = NN/'raw'/'Dataset001_Lesion'\n"
    "split = json.load(open(NN/'preprocessed'/'Dataset001_Lesion'/'splits_final.json'))[0]\n"
    "train_cases, val_cases = split['train'], split['val']\n"
    "test_cases = sorted(p.name[:-9] for p in (RAW/'imagesTs').glob('*_0000.png'))\n"
    "print(f'fold-0  train={len(train_cases)}  val={len(val_cases)}  |  Halle test={len(test_cases)}')\n"
    "\n"
    "def gray(case, img_sub):\n"
    "    p = RAW/img_sub/f'{case}_0000.png'\n"
    "    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None\n"
    "def gt(case, lab_sub):\n"
    "    p = RAW/lab_sub/f'{case}.png'\n"
    "    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)\n"
    "    return (m>0).astype('uint8') if m is not None else None\n"
    "def pred(case, folder):\n"
    "    p = NN/folder/f'{case}.png'\n"
    "    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)\n"
    "    return (m>0).astype('uint8') if m is not None else None\n"
    "def dice(a,b):\n"
    "    a,b = a>0, b>0; d = a.sum()+b.sum(); return 1.0 if d==0 else float(2*(a&b).sum()/d)\n"
))

cells.append(md(
    "### The data that goes into the model\n"
    "This is exactly what nnU-Net sees: the de-identified B-mode frame with the lesion mask (cyan) + "
    "its bounding box (yellow), kept strictly separate between the **train** and **val** patients of "
    "fold 0."
))
cells.append(code(
    "from matplotlib.patches import Rectangle\n"
    "def show_gt_grid(cases, img_sub, lab_sub, title, n=5):\n"
    "    cases = cases[:n]\n"
    "    fig, ax = plt.subplots(1, len(cases), figsize=(3.1*len(cases), 3.3))\n"
    "    if len(cases)==1: ax=[ax]\n"
    "    for a, c in zip(ax, cases):\n"
    "        g, m = gray(c, img_sub), gt(c, lab_sub)\n"
    "        if g is None or m is None: a.axis('off'); continue\n"
    "        a.imshow(overlay_mask(g, m, (0,229,255), 0.45), aspect='auto')\n"
    "        ys,xs = np.where(m)\n"
    "        if len(xs):\n"
    "            a.add_patch(Rectangle((xs.min(),ys.min()), xs.max()-xs.min(), ys.max()-ys.min(),\n"
    "                                  fill=False, edgecolor='#ffee00', lw=1.3))\n"
    "        a.set_xticks([]); a.set_yticks([]); a.set_title(c, fontsize=8)\n"
    "    fig.suptitle(title, fontweight='bold'); plt.tight_layout(); plt.show()\n"
    "\n"
    "show_gt_grid(train_cases, 'imagesTr', 'labelsTr', 'TRAIN data (fold 0) — image + GT lesion mask/box', n=5)\n"
    "show_gt_grid(val_cases,  'imagesTr', 'labelsTr', 'VAL data (fold 0, held out) — image + GT lesion mask/box', n=5)\n"
))

cells.append(md(
    "### Straight out of the model — predicted segmentation\n"
    "For each case: the raw input, the ground-truth lesion, and the **model's predicted mask** "
    "(magenta). Per-case Dice is shown. First the held-out **Dresden val**, then the **Halle external** "
    "test (a different hospital the model never saw)."
))
cells.append(code(
    "def show_pred_grid(cases, img_sub, lab_sub, pred_folder, title, n=5):\n"
    "    cases = [c for c in cases if pred(c, pred_folder) is not None][:n]\n"
    "    fig, ax = plt.subplots(3, len(cases), figsize=(3.0*len(cases), 8.6))\n"
    "    for j, c in enumerate(cases):\n"
    "        g, m, pr = gray(c, img_sub), gt(c, lab_sub), pred(c, pred_folder)\n"
    "        col = ax[:, j] if len(cases)>1 else ax\n"
    "        col[0].imshow(g, cmap='gray', aspect='auto'); col[0].set_title(f'{c}\\ninput', fontsize=8)\n"
    "        col[1].imshow(overlay_mask(g, m, (0,229,255), 0.5), aspect='auto'); col[1].set_title('ground truth', fontsize=8)\n"
    "        col[2].imshow(overlay_mask(g, pr, (255,0,200), 0.5), aspect='auto')\n"
    "        col[2].set_title(f'prediction  (Dice={dice(pr,m):.2f})', fontsize=8)\n"
    "        for a in col: a.set_xticks([]); a.set_yticks([])\n"
    "    fig.suptitle(title, fontweight='bold', y=1.0); plt.tight_layout(); plt.show()\n"
    "\n"
    "show_pred_grid(val_cases, 'imagesTr', 'labelsTr', 'pred_val', 'Dresden VAL — GT vs nnU-Net prediction (best checkpoint)', n=5)\n"
    "show_pred_grid(test_cases, 'imagesTs', 'labelsTs', 'pred_halle', 'Halle EXTERNAL — GT vs nnU-Net prediction', n=5)\n"
))

cells.append(md(
    "### Takeaways\n"
    "- The **GT panels confirm the pipeline is correct** — masks/boxes sit on real lesions, cleanly "
    "split between train and val patients.\n"
    "- Predictions land on some Dresden val lesions but **collapse on Halle external** (mean Dice ≈ "
    "0.08, ~17% detected) — localization is limited by only ~140 unique lesions and subtle B-mode "
    "appearance, not by the pipeline.\n"
    "- **Next:** 5-fold ensemble, tighter/higher-resolution lesion ROI, patient-grouped multi-frame "
    "data, and RF/QUS channels. The stage-2 **benign/malignant classifier** already works "
    "(0.90 Dresden / 0.65 Halle) and pairs with this as a two-stage system."
))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python (usloc)", "language": "python", "name": "usloc"}
nb.metadata["language_info"] = {"name": "python"}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB_PATH))
print("wrote", NB_PATH)
