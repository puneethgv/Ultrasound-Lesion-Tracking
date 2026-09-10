"""Train the stage-2 lesion classifier on cropped lesions (torchvision ResNet18, ImageNet-pretrained).

Grayscale crops are replicated to 3 channels. Class imbalance is handled with a class-balanced
sampler. Reports balanced accuracy + confusion on val (Dresden) and the locked Halle test.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np


def _make_loaders(data_dir: Path, imgsz: int, batch: int):
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

    # class-balanced sampler for train
    labels = [y for _, y in sets["train"].samples]
    freq = Counter(labels)
    weights = [1.0 / freq[y] for y in labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
    loaders = {
        "train": torch.utils.data.DataLoader(sets["train"], batch_size=batch, sampler=sampler, num_workers=4),
    }
    for s in ("val", "test"):
        if s in sets:
            loaders[s] = torch.utils.data.DataLoader(sets[s], batch_size=batch, shuffle=False, num_workers=4)
    return loaders, classes


def _balanced_accuracy(y_true, y_pred, n_classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs = []
    for c in range(n_classes):
        m = y_true == c
        if m.any():
            accs.append((y_pred[m] == c).mean())
    return float(np.mean(accs)) if accs else 0.0


def _evaluate(model, loader, device, n_classes):
    import torch

    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            yp += out.argmax(1).cpu().tolist()
            yt += y.tolist()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    return _balanced_accuracy(yt, yp, n_classes), cm


def train_classifier(data_dir, *, epochs=30, imgsz=224, batch=64, lr=3e-4, device="cuda"):
    import torch
    import torch.nn as nn
    from torchvision import models

    data_dir = Path(data_dir)
    loaders, classes = _make_loaders(data_dir, imgsz, batch)
    n = len(classes)
    device = device if torch.cuda.is_available() else "cpu"

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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
            ba, cm = _evaluate(model, loaders[s], device, n)
            results[s] = {"balanced_acc": ba, "confusion": cm.tolist()}
            print(f"\n{s.upper()}  balanced_acc={ba:.3f}\n  classes={classes}\n  confusion=\n{cm}")
    return model, results
