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
- ✅ **Phase-based exploration notebook** (`notebooks/01_dataset_exploration.ipynb`).
- ⏳ Next: resolve label↔image coordinate registration, dataset export, YOLO11-seg baseline, RF ablation.

See the full plan in the project notes; each plan phase maps to a notebook phase for easy tracking.

## Dataset (summary)

| Class | Dresden (train+CV) | Halle (ext. test) | Total |
|---|---|---|---|
| FNH | 15 | 3 | 18 |
| Hemangioma | 42 | 16 | 58 |
| Metastasis | 66 | 22 | 88 |

Per case: DICOM cines (JPEG-LS, 800×800), raw RF (`*_rf.raw`/`*_rf_no_tgc.npy`) and pre-scan envelope
(`*_env.raw`), plus GUI bounding-box labels and (for ~142 cases) FLL_ROI spline masks. Site is encoded
in the case-ID prefix (`UKH…` = Halle).

> ⚠️ **PHI:** DICOM frames contain **burned-in identifiers** (case ID, institution, date, operator).
> `read_dicom(deidentify=True)` (default) keeps only the declared ultrasound region and scrubs PHI
> header tags — the ultrasound sector is never altered (`usloc/deid.py`, `scripts/audit_deid.py`).
> Notebook outputs are also stripped from git by default (`nbstripout`) as defense-in-depth.

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

## Layout

```
src/usloc/      config, io (rf/env/dicom), recon (b-mode), data (cohort), labels (box/spline), viz
scripts/        build_notebook.py (and future data-prep / training entrypoints)
notebooks/      phased exploration
tests/          pytest unit tests (dataset-free)
configs/        data / split / recon config
```

## Tests

```bash
pytest            # dataset-free unit tests
```
