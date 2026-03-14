import base64
import binascii
import io

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

# Load trained model
model = tf.keras.models.load_model("pneumonia_model.keras")

app = FastAPI()

HTML_PREFIXES = (
    b"<!doctype html",
    b"<html",
)

# Function to preprocess image
def preprocess_image(image):

    image = image.resize((224,224))
    image = np.array(image)

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image


def load_uploaded_image(contents: bytes) -> Image.Image:
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    payload = contents.strip()

    lowered = payload[:128].lower()
    if any(lowered.startswith(prefix) for prefix in HTML_PREFIXES):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is an HTML page, not an image. The source path likely saved an error page instead of the X-ray.",
        )

    candidates = [payload]

    if payload.startswith(b"data:image/") and b"," in payload:
        _, _, payload = payload.partition(b",")

    if payload not in candidates:
        candidates.append(payload)

    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        decoded = None

    if decoded:
        candidates.append(decoded)

    for candidate in candidates:
        try:
            image = Image.open(io.BytesIO(candidate))
            image.load()
            return image.convert("RGB")
        except (UnidentifiedImageError, OSError):
            continue

    raise HTTPException(
        status_code=400,
        detail="Invalid image upload. Send a JPG/PNG file or base64 image data.",
    )


@app.get("/")
def home():
    return {"message": "Pneumonia Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = load_uploaded_image(contents)

    img = preprocess_image(image)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        result = "PNEUMONIA"
    else:
        result = "NORMAL"

    return {
        "prediction": result,
        "confidence": float(prediction)
    }
