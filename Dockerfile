# Fall detection API — deploy on Render, Railway, Fly.io, Google Cloud Run, Azure Container Apps.
# Not suitable for Vercel serverless (TensorFlow + PyTorch exceed size/runtime limits).

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

# One worker: models stay loaded in memory (multiple workers = multiple copies of TF/YOLO).
CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 1 --timeout 300 --access-logfile - --error-logfile -"]
