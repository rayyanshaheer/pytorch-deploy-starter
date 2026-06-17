"""Minimal PyTorch training loop you can adapt for your own dataset.

This is intentionally small and framework-pure so students can read it top to
bottom. It trains a tiny CNN on random tensors as a stand-in for real data,
then saves the weights to `model.pt`. Replace the dataset and model with your
own, then point `app/model.py` at the saved weights.

Run:
    python train.py
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TinyCNN(nn.Module):
    """A small convolutional network for 3x32x32 inputs."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def make_demo_loader(samples: int = 256, num_classes: int = 4) -> DataLoader:
    """Build a fake dataset so the loop runs without external downloads."""
    images = torch.randn(samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=32, shuffle=True)


def train(epochs: int = 3) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)
    loader = make_demo_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch {epoch}: loss={running_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "model.pt")
    print("saved weights to model.pt")


if __name__ == "__main__":
    train()
