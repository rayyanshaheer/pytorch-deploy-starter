"""FastAPI service that serves PyTorch model predictions over HTTP.

Run locally:
    uvicorn app.main:app --reload

Then POST an image to /predict, or open http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.model import predict

app = FastAPI(
    title="PyTorch Deploy Starter",
    description="A minimal, production-shaped template for serving a PyTorch model.",
    version="1.0.0",
)

_ALLOWED_TYPES = {"image/jpeg", "image/png"}


@app.get("/health")
def health() -> dict:
    """Liveness probe used by Docker, CI, and cloud load balancers."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict:
    """Accept an image upload and return the model's top predictions."""
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    predictions = predict(image_bytes)
    return {"predictions": predictions}
