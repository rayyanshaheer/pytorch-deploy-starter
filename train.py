"""Transfer-learning trainer for the Oxford Flowers-102 dataset.

Fine-tunes a pretrained ResNet-18 to classify 102 flower species. The backbone
is frozen and only the classifier head is trained, so this runs in minutes on a
CPU. Outputs:

    artifacts/flowers_resnet18.pt   trained weights
    artifacts/metrics.json          accuracy + training metadata

Run:
    python train.py --epochs 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import Flowers102
from torchvision.models import ResNet18_Weights

from labels import FLOWER_CLASSES

ARTIFACTS = Path("artifacts")
WEIGHTS_PATH = ARTIFACTS / "flowers_resnet18.pt"
METRICS_PATH = ARTIFACTS / "metrics.json"


def build_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    weights = ResNet18_Weights.DEFAULT
    tf = weights.transforms()
    # Flowers102 "train" and "val" splits are 1020 images each (10 per class).
    # We combine them for training and evaluate on a slice of the large test set.
    train_a = Flowers102(root="data", split="train", download=True, transform=tf)
    train_b = Flowers102(root="data", split="val", download=True, transform=tf)
    train_ds = torch.utils.data.ConcatDataset([train_a, train_b])
    test_ds = Flowers102(root="data", split="test", download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def build_model() -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(FLOWER_CLASSES))
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    correct = total = 0
    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += int((preds == labels).sum())
        total += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-batches", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, test_loader = build_loaders(args.batch_size)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        acc = evaluate(model, test_loader, device, args.eval_batches)
        print(f"epoch {epoch}: loss={running / len(train_loader):.4f} test_acc={acc:.4f}")

    train_seconds = round(time.time() - start, 1)
    final_acc = evaluate(model, test_loader, device, max_batches=10_000)

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    metrics = {
        "dataset": "Oxford Flowers-102",
        "architecture": "ResNet-18 (frozen backbone, fine-tuned head)",
        "num_classes": len(FLOWER_CLASSES),
        "test_accuracy": round(final_acc, 4),
        "epochs": args.epochs,
        "train_seconds": train_seconds,
        "device": str(device),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved weights to {WEIGHTS_PATH}")
    print(f"final test accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
