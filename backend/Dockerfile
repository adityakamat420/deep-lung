FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow / TF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=pneumonia_model.keras
ENV PORT=5000

EXPOSE 5000

# Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
