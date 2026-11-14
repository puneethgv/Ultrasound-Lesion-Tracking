"""Train the stage-2 lesion classifier on cropped lesions (torchvision ResNet18, ImageNet-pretrained).

Grayscale crops are replicated to 3 channels. Class imbalance is handled with a class-balanced
sampler. Reports balanced accuracy + confusion on val (Dresden) and the locked Halle test, per crop
and per patient (``usloc.classify.metrics``). The per-patient number is the one to quote: a patient
contributes up to 40 near-identical crops.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .metrics import balanced_accuracy, confusion, patient_level


def _make_loaders(data_dir: Path, imgsz: int, batch: int, workers: int = 4):
    import torch
    from torchvision import datasets, transforms

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), norm,
    ])
    tf_eval = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize((imgsz, imgsz)), transforms.ToTensor(), norm,
    ])
    sets = {s: datasets.ImageFolder(str(data_dir / s), transform=(tf_train if s == "train" else tf_eval))
            for s in ("train", "val", "test") if (data_dir / s).exists()}
    classes = sets["train"].classes
    # ImageFolder numbers the classes of each split by the folders it finds there. If a split lacks a
    # class folder its numbers would shift and every label after it would be wrong, so use train's.
    for s, ds in sets.items():
        if ds.classes == classes:
            continue
        extra = [c for c in ds.classes if c not in classes]
        if extra:
            raise ValueError(f"{s} has classes {extra} that train does not have")
        to_train = [classes.index(c) for c in ds.classes]
        ds.samples = [(p, to_train[y]) for p, y in ds.samples]
        ds.targets = [y for _, y in ds.samples]
        ds.classes, ds.class_to_idx = classes, {c: i for i, c in enumerate(classes)}

    # class-balanced sampler for train
    labels = [y for _, y in sets["train"].samples]
    freq = Counter(labels)
    weights = [1.0 / freq[y] for y in labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
    loaders = {
        "train": torch.utils.data.DataLoader(sets["train"], batch_size=batch, sampler=sampler,
                                             num_workers=workers),
    }
    for s in ("val", "test"):
        if s in sets:
            loaders[s] = torch.utils.data.DataLoader(sets[s], batch_size=batch, shuffle=False,
                                                     num_workers=workers)
    return loaders, classes


def _predict(model, loader, device):
    """Labels and class probabilities of every crop, in dataset order."""
    import torch

    model.eval()
    yt, pr = [], []
    with torch.no_grad():
        for x, y in loader:
            pr.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
            yt += y.tolist()
    return np.asarray(yt, dtype=int), np.concatenate(pr)


def _evaluate(model, loader, device, n_classes):
    yt, pr = _predict(model, loader, device)
    yp = pr.argmax(1)
    return balanced_accuracy(yt, yp, n_classes), confusion(yt, yp, n_classes)


def train_classifier(data_dir, *, epochs=30, imgsz=224, batch=64, lr=3e-4, device="cuda",
                     workers=4, pretrained=True):
    import torch
    import torch.nn as nn
    from torchvision import models

    data_dir = Path(data_dir)
    loaders, classes = _make_loaders(data_dir, imgsz, batch, workers)
    n = len(classes)
    device = device if torch.cuda.is_available() else "cpu"

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        sched.step()
        if "val" in loaders:
            vb, _ = _evaluate(model, loaders["val"], device, n)
            if vb > best_val:
                best_val, best_state = vb, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch {ep+1:02d}/{epochs}  val_balanced_acc={vb:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    results = {"classes": classes, "val_balanced_acc": best_val}
    for s in ("val", "test"):
        if s in loaders:
            yt, pr = _predict(model, loaders[s], device)
            yp = pr.argmax(1)
            ba, cm = balanced_accuracy(yt, yp, n), confusion(yt, yp, n)
            names = [Path(p).name for p, _ in loaders[s].dataset.samples]
            pat = patient_level(names, yt, pr, classes)
            results[s] = {"balanced_acc": ba, "confusion": cm.tolist(), "n_crops": int(len(yt)), **pat}
            print(f"\n{s.upper()}  per crop    balanced_acc={ba:.3f}  (n={len(yt)})"
                  f"\n  classes={classes}\n  confusion=\n{cm}")
            auc_txt = f"  auc={pat['patient_auc']:.3f}" if "patient_auc" in pat else ""
            print(f"{s.upper()}  per patient balanced_acc={pat['patient_balanced_acc']:.3f}"
                  f"  (n={pat['n_patients']}){auc_txt}\n  confusion=\n{np.array(pat['patient_confusion'])}")
    return model, results
