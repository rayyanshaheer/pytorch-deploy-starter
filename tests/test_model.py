"""Integration test for the real model when trained weights are present.

Skips automatically in CI where weights are not committed-as-checkout, so the
fast API tests still cover the service surface.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.model import predict


@pytest.mark.skipif(
    not Path("artifacts/flowers_resnet18.pt").exists(),
    reason="trained weights not available in this environment",
)
def test_predict_shape_and_confidence():
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), color=(200, 120, 160)).save(buffer, format="PNG")
    results = predict(buffer.getvalue(), top_k=5)

    assert len(results) == 5
    assert all(0.0 <= r["confidence"] <= 1.0 for r in results)
    # confidences must be sorted high -> low
    confidences = [r["confidence"] for r in results]
    assert confidences == sorted(confidences, reverse=True)
