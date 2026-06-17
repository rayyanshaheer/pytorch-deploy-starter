"""Tests for the inference API.

These stub the predict function so CI stays fast and offline-friendly.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from app import main

client = TestClient(main.app)


def _fake_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=(120, 200, 80)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_returns_predictions(monkeypatch):
    monkeypatch.setattr(
        main,
        "predict",
        lambda image_bytes, top_k=3: [{"label": "rose", "confidence": 0.97}],
    )
    response = client.post(
        "/predict",
        files={"file": ("flower.png", _fake_png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["predictions"][0]["label"] == "rose"
