# PyTorch Deploy Starter

A small, production-shaped template for taking a **PyTorch model from notebook to a running API**. Built for students who can train a model but get stuck on the deployment side: serving, containerizing, CI, and shipping to the cloud.

Most beginner PyTorch tutorials stop at `model.eval()`. This repo starts there and shows the next mile: wrapping a model in a real HTTP service, testing it, containerizing it, and running it on a cloud instance.

## What's inside

| Piece | File | Why it matters |
|-------|------|----------------|
| Model + inference | `app/model.py` | Loads a pretrained ResNet-18, runs top-k inference |
| HTTP API | `app/main.py` | FastAPI service with `/predict` and `/health` |
| Training loop | `train.py` | Minimal, readable PyTorch training you can adapt |
| Tests | `tests/test_api.py` | Fast, offline tests for the API |
| Container | `Dockerfile` | CPU-only image that warms the model at build time |
| CI | `.github/workflows/ci.yml` | Lint + test on every push and PR |

## Quickstart

```bash
# 1. Install (CPU build)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt

# 2. Run the API
uvicorn app.main:app --reload

# 3. Try it
curl -F "file=@your-image.jpg" http://localhost:8000/predict
# or open http://localhost:8000/docs
```

## Run with Docker

```bash
docker build -t pytorch-deploy-starter .
docker run -p 8000:8000 pytorch-deploy-starter
```

## Deploy to AWS (quick path)

This runs on a free-tier-friendly CPU instance. No GPU required for inference on a small model.

1. Launch an EC2 instance (Ubuntu, t3.small is enough for ResNet-18 inference).
2. Install Docker: `sudo apt update && sudo apt install -y docker.io`.
3. Clone this repo and `docker build -t pytorch-deploy-starter .`.
4. `docker run -d -p 80:8000 pytorch-deploy-starter`.
5. Hit `http://<your-ec2-ip>/docs`.

> Security note: this starter ships **without authentication** so it stays easy to learn from. Before exposing it publicly, add an API key or put it behind a gateway, and restrict the security group to known IPs.

## Train your own model

```bash
python train.py        # writes model.pt
```

Then load `model.pt` inside `app/model.py` instead of the pretrained ResNet, and update the labels.

## Who built this

Created by [Rayyan Shaheer](https://github.com/rayyanshaheer) as a teaching resource for student developers moving from "I trained a model" to "my model is running in production." It pairs PyTorch with the cloud and DevOps workflow (Docker, CI/CD, AWS) that most ML courses skip.

## License

MIT — use it, fork it, teach with it.
