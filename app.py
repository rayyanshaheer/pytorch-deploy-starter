"""Gradio web demo for the Flowers-102 classifier (Hugging Face Spaces entry point)."""

from __future__ import annotations

import io

import gradio as gr

from app.model import predict

_DESCRIPTION = (
    "A fine-tuned PyTorch ResNet-18 that recognizes 102 flower species "
    "(Oxford Flowers-102, ~86% test accuracy). Built as an open teaching "
    "resource for students learning to take a model from training to "
    "production. Code: github.com/rayyanshaheer/pytorch-deploy-starter"
)

def classify(image):
    if image is None:
        return {}
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    results = predict(buffer.getvalue(), top_k=5)
    return {item["label"]: item["confidence"] for item in results}

with gr.Blocks(title="Flower Classifier — PyTorch ResNet-18") as demo:
    gr.Markdown("# Flower Classifier — PyTorch ResNet-18")
    gr.Markdown(_DESCRIPTION)
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload a flower photo")
        out = gr.Label(num_top_classes=5, label="Top predictions")
    gr.Button("Classify").click(classify, inputs=inp, outputs=out, api_name=False)

if __name__ == "__main__":
    demo.launch(ssr_mode=False, show_api=False)
