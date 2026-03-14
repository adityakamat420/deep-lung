# Pneumonia Detection API

REST API that serves the `pneumonia_model.keras` model for chest X-ray classification.

---

## Project Structure

```
pneumonia_api/
├── app.py                  # Flask application
├── requirements.txt
├── Dockerfile
└── pneumonia_model.keras   # ← place your model here
```

---

## Setup & Run

### Local (Python)

```bash
pip install -r requirements.txt
python app.py
# Server starts at http://localhost:5000
```

### Docker

```bash
docker build -t pneumonia-api .
docker run -p 5000:5000 -v $(pwd)/pneumonia_model.keras:/app/pneumonia_model.keras pneumonia-api
```

### Custom model path / port

```bash
MODEL_PATH=/models/my_model.keras PORT=8080 python app.py
```

---

## API Reference

### `GET /`
Health check.

**Response**
```json
{ "status": "ok", "model": "pneumonia_model.keras" }
```

---

### `POST /predict`
Classify a chest X-ray image.

**Request**
- Content-Type: `multipart/form-data`
- Field: `image` — JPEG / PNG / BMP / TIFF file

**Response (200)**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.97,
  "scores": {
    "NORMAL":    0.03,
    "PNEUMONIA": 0.97
  }
}
```

**Error responses**

| Code | Reason |
|------|--------|
| 400  | No `image` field / empty filename |
| 415  | Unsupported file type |
| 500  | Internal model or image error |

---

## Flutter Integration (HTTP)

In Flutter, use the `http` or `dio` package to send a multipart request:

```
POST http://<your-server>:5000/predict
Content-Type: multipart/form-data
Body: image = <XRay.jpg>
```

The response JSON maps directly to a Dart model — parse `prediction` and `confidence` for your UI.

---

## Notes

- Input images are automatically resized to **224×224** and normalised to [0, 1].
- Both **sigmoid** (binary) and **softmax** (multi-class) output layers are supported automatically.
- For production, run behind HTTPS (e.g. nginx + Let's Encrypt) before exposing to Flutter.
