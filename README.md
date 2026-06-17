# Flower Classifier — PyTorch, trained to production

An end-to-end, open-source PyTorch project that recognizes **102 flower species** and ships as a real service. It is built as a teaching resource for student developers who can train a model but get stuck taking it to production.

**What makes it complete:** a fine-tuned PyTorch model, a web demo, a REST API, a container, and CI — the full path from `train.py` to a running app.

**🌸 Live demo:** https://huggingface.co/spaces/rayyanshaheer/flower-classifier-pytorch

## Results

| | |
|---|---|
| Dataset | Oxford Flowers-102 (102 classes, 6,149 test images) |
| Model | ResNet-18, frozen backbone + fine-tuned classifier head |
| **Test accuracy** | **86.4%** |
| Training time | ~21 min on CPU (no GPU needed) |

Accuracy and metadata are written to `artifacts/metrics.json` by the trainer, so the numbers above are reproducible, not hand-typed.

## Try it

### Web demo (Gradio)
```bash
pip install -r requirements.txt
python app.py          # opens a local web UI; drop in a flower photo
```

### REST API (FastAPI)
```bash
uvicorn app.main:app --reload
curl -F "file=@flower.jpg" http://localhost:8000/predict
# or open http://localhost:8000/docs
```

### Docker
```bash
docker build -t flower-classifier .
docker run -p 8000:8000 flower-classifier
```

## What's inside

| Piece | File | Purpose |
|-------|------|---------|
| Training | `train.py` | Transfer-learning loop, saves weights + metrics |
| Model + inference | `app/model.py` | Loads fine-tuned weights, top-k prediction |
| Web demo | `app.py` | Gradio UI (also the Hugging Face Spaces entry point) |
| REST API | `app/main.py` | FastAPI `/predict` and `/health` |
| Labels | `labels.py` | 102 human-readable flower names |
| Tests | `tests/` | Fast API tests + a real-model integration test |
| Container | `Dockerfile` | CPU image that ships the trained weights |
| CI | `.github/workflows/ci.yml` | Lint + test on every push and PR |

## Reproduce the training
```bash
python train.py --epochs 8
```
This downloads Flowers-102, fine-tunes the head, and writes `artifacts/flowers_resnet18.pt` and `artifacts/metrics.json`.

## Deploy a live demo on Hugging Face Spaces (free)
1. Create a new Space → SDK: **Gradio**.
2. Push this repo to the Space (it already has `app.py` and `requirements.txt`).
3. The Space builds and serves the demo automatically.

## Why this exists

Most beginner PyTorch tutorials stop at `model.eval()`. Students rarely see the next mile: serving, containerizing, testing, and deploying. This repo shows that whole path on one small, readable project, pairing PyTorch with the cloud and DevOps workflow that ML courses usually skip.

Built by [Rayyan Shaheer](https://github.com/rayyanshaheer) as an open teaching resource for student developers.

## License
MIT — use it, fork it, teach with it.
