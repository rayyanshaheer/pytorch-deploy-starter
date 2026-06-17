"""PyTorch model loading and inference logic.

Uses a pretrained ResNet-18 from torchvision so the starter works out of the
box without a training step. Swap `build_model` for your own architecture and
load your own weights when you move to a real project.
"""

from __future__ import annotations

import io
from functools import lru_cache

import torch
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights

# Standard ImageNet preprocessing pipeline shipped with the pretrained weights.
_weights = ResNet18_Weights.DEFAULT
_preprocess = _weights.transforms()
_categories = _weights.meta["categories"]


@lru_cache(maxsize=1)
def build_model() -> torch.nn.Module:
    """Load a pretrained ResNet-18 once and cache it for reuse."""
    model = models.resnet18(weights=_weights)
    model.eval()
    return model


def predict(image_bytes: bytes, top_k: int = 3) -> list[dict]:
    """Run inference on raw image bytes and return the top-k predictions.

    Args:
        image_bytes: Raw bytes of an image file (JPEG/PNG).
        top_k: Number of top predictions to return.

    Returns:
        A list of {"label": str, "confidence": float} dicts, highest first.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        logits = build_model()(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top = torch.topk(probs, k=min(top_k, probs.shape[0]))
    return [
        {"label": _categories[idx], "confidence": round(float(score), 4)}
        for score, idx in zip(top.values, top.indices, strict=False)
    ]
