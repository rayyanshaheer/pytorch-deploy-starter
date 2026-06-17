"""PyTorch model loading and inference for the Flowers-102 classifier.

Loads the fine-tuned ResNet-18 produced by `train.py` and classifies an image
into one of 102 flower species. If the trained weights are missing, it falls
back to the pretrained ImageNet head so the app still starts.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights

from labels import FLOWER_CLASSES

_WEIGHTS_PATH = Path("artifacts/flowers_resnet18.pt")
_imagenet_weights = ResNet18_Weights.DEFAULT
_preprocess = _imagenet_weights.transforms()


@lru_cache(maxsize=1)
def build_model() -> tuple[torch.nn.Module, list[str]]:
    """Load the fine-tuned flower classifier once and cache it.

    Returns the model and the list of class names it predicts over.
    """
    if _WEIGHTS_PATH.exists():
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, len(FLOWER_CLASSES))
        model.load_state_dict(torch.load(_WEIGHTS_PATH, map_location="cpu"))
        classes = FLOWER_CLASSES
    else:
        # Fallback: pretrained ImageNet model so the service still runs.
        model = models.resnet18(weights=_imagenet_weights)
        classes = _imagenet_weights.meta["categories"]

    model.eval()
    return model, classes


def predict(image_bytes: bytes, top_k: int = 3) -> list[dict]:
    """Run inference on raw image bytes and return the top-k predictions.

    Returns a list of {"label": str, "confidence": float} dicts, highest first.
    """
    model, classes = build_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _preprocess(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top = torch.topk(probs, k=min(top_k, probs.shape[0]))
    return [
        {"label": classes[idx], "confidence": round(float(score), 4)}
        for score, idx in zip(top.values, top.indices, strict=False)
    ]
