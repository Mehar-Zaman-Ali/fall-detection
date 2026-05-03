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

RUN chmod +x docker/start.sh

ENV PYTHONUNBUFFERED=1
# Fly.io / Render set PORT at runtime (Fly matches internal_port, often 8080). Do not pin PORT here.

EXPOSE 8080

# Exec form must use an absolute path — `docker/start.sh` is not found via PATH.
CMD ["/app/docker/start.sh"]
