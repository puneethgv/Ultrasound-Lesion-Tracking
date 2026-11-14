# Ultrasound Lesion Tracking

Automatic **focal-liver-lesion detection, classification & segmentation** from contrast/RF ultrasound.
Given a liver ultrasound acquisition, the system **localizes** the lesion (bounding box + segmentation
mask) and **classifies** its type (FNH / hemangioma / metastasis).

The dataset ships **raw RF** alongside B-mode, so a core research question is whether RF adds value
over B-mode ("brightness"). Data comes from two sites — **TU Dresden** (train) and **Halle**
(external test) — the intended generalization protocol.

## Status

- ✅ Environment, package scaffold, data readers (RF/envelope/DICOM), cohort builder, label parsers.
- ✅ **De-identification pipeline** (`usloc/deid.py`) — region-based burned-in-PHI removal + header
  scrub; on by default in `read_dicom`; audited to cover 100% of the 490 cines.
- ✅ **Label↔image registration — corrected.** The `FLL_ROI` splines are **QuantUS ROI exports**:
  drawn on QuantUS's own scan-conversion of the RF acquisition (`Scan Name`), with `Frame` indexing RF
  frames — not DICOM pixels. The earlier DICOM-crop mapping misplaced the masks.
  `usloc/labels/quantus.py` inverts the QuantUS warp into the RF grid (where the GUI boxes live) and
  `register_case` (default `source='rf'`) renders image + mask from that RF frame through one grid.
  Validate on the data with `python scripts/validate_alignment.py --review` (see
  *Checking the labels and the results* below).
- ✅ Clarius `*.raw` now parsed with header + per-frame timestamps (`read_clarius_raw`); the old flat
  read shifted frame *k* by 10 + 4k samples.
- ✅ **Notebooks** (de-identified figures): `01_dataset_exploration.ipynb` (data tour; its Phase 4b
  still shows the old DICOM placement), `02_training_method1_nnunet.ipynb` (nnU-Net with the old and
  the corrected masks), `03_mask_alignment_check.ipynb` (mask vs GUI box for every case),
  `04_medsam2_zeroshot_vs_box.ipynb` (MedSAM2 with a point and with a box prompt).
- **nnU-Net** (2D, fold 0, 100 epochs), retrained on the corrected masks:

  | | old masks (DICOM) | corrected masks (RF) |
  |---|---|---|
  | Dresden val mean Dice (21 cases) | 0.052 | 0.352 |
  | Halle external mean Dice (36 cases) | 0.077 | 0.265 |
  | Halle cases with Dice > 0.1 | 17% | 44% |

  One fold and small sets, and the images changed together with the masks (DICOM crop → RF fan;
  83 → 80 training cases), so read the size of the gain as approximate.
- **MedSAM2** on Halle (no training): 0.21 Dice prompted with the image centre, 0.88 prompted with
  the true mask's bounding box. Filling that box scores about 0.88 by itself on a round lesion, so
  the second number mostly says that localization is what is missing.
- ⚠️ The benign/malignant **classifier** (0.90 Dresden val / 0.65 Halle external, per crop) and
  **YOLO** were trained with the old masks — re-export and retrain them.
- ⏳ Next: nnU-Net 5-fold ensemble, tighter/higher-res ROI, patient-grouped multi-frame, RF/QUS channels.

See the full plan in the project notes; each plan phase maps to a notebook phase for easy tracking.

## Dataset (summary)

| Class | Dresden (train+CV) | Halle (ext. test) | Total |
|---|---|---|---|
| FNH | 15 | 3 | 18 |
| Hemangioma | 41 | 16 | 57 |
| Metastasis | 66 | 22 | 88 |

Per case: DICOM cines (JPEG-LS, 800×800), raw RF (`*_rf.raw`/`*_rf_no_tgc.npy`) and pre-scan envelope
(`*_env.raw`), plus GUI bounding-box labels and (for 141 cases) FLL_ROI spline masks. Site is encoded
in the case-ID prefix (`UKH…` = Halle).

> ⚠️ **PHI:** DICOM frames contain **burned-in identifiers** (case ID, institution, date, operator).
> `read_dicom(deidentify=True)` (default) keeps only the declared ultrasound region and scrubs PHI
> header tags — the ultrasound sector is never altered (`usloc/deid.py`, `scripts/audit_deid.py`).
> All committed notebook figures are de-identified; the one raw "before" demo panel is pixelated so no
> identifier is legible.

## Setup

```bash
python3 -m venv --system-site-packages .venv     # reuses system torch (CUDA) / numpy / pandas
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name usloc --display-name "Python (usloc)"
```

Point the code at the dataset (defaults to the mounted drive):

```bash
export USLOC_DATA_ROOT=/path/to/Dresden_new
```

## Run the exploration notebook

```bash
python scripts/build_notebook.py     # regenerate the notebook
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=usloc notebooks/01_dataset_exploration.ipynb
```

## Checking the labels and the results

```bash
# mask vs GUI box for every case -> alignment_report.csv, plus one two-frame picture per flagged case
python scripts/validate_alignment.py --review

# score existing predictions case by case, next to two baselines that need no model
python scripts/eval_nnunet.py --dataset-id 2 --name LesionRF --pred <prediction folder> --skip-predict

# control for the classifier: the same model on crops next to the lesion
python scripts/train_classifier.py --task binary --region beside

# train without the flagged cases (the Halle test set stays complete)
python scripts/export_nnunet.py --dataset-id 3 --name LesionRFclean     --exclude-file $USLOC_DERIVED/alignment/alignment_report.csv
```

- `alignment_report.csv` says *how* a mask and its box differ (`relation`: inside a larger box, shifted,
  apart), on which frames they were drawn (`frame_gap`), and whether they agree once the box is moved
  with the tissue to the mask's frame (`iou_tracked`). A low IoU is a reason to look at a case, not a
  reason to drop it: the models train on the masks only.
- A redrawn ROI goes next to the original as `<case>_roi_2.pkl`; the highest number is used.
- ROIs must be drawn on the QuantUS picture the conversion expects. In a QuantUS from June 2026 or
  later, set `start_depth_factor` to `0.25` (see `usloc/labels/quantus.py`).

## Layout

```
src/usloc/      config, io (rf/env/dicom), recon (b-mode), data (cohort), labels (box/spline/checks),
                datasets (nnU-Net / YOLO export), classify, eval (scores + baselines), viz
scripts/        notebook builders, dataset export, training, evaluation, label checks
notebooks/      phased exploration
tests/          pytest unit tests (dataset-free)
configs/        data / split / recon config
```

## Tests

```bash
pytest            # dataset-free unit tests
```
