"""Gradio web demo for the Flowers-102 classifier.

This file is the entry point for Hugging Face Spaces (Spaces runs `app.py`).
Run locally with:

    python app.py

Then open the printed local URL and drop in a flower photo.
"""

from __future__ import annotations

import io

import gradio as gr

from app.model import predict


def classify(image) -> dict:
    """Take a PIL image from Gradio and return label -> confidence for the UI."""
    if image is None:
        return {}
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    results = predict(buffer.getvalue(), top_k=5)
    return {item["label"]: item["confidence"] for item in results}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a flower photo"),
    outputs=gr.Label(num_top_classes=5, label="Top predictions"),
    title="Flower Classifier — PyTorch ResNet-18",
    description=(
        "A fine-tuned PyTorch ResNet-18 that recognizes 102 flower species "
        "(Oxford Flowers-102, ~86% test accuracy). Built as an open teaching "
        "resource for students learning to take a model from training to "
        "production. Code: github.com/rayyanshaheer/pytorch-deploy-starter"
    ),
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()
