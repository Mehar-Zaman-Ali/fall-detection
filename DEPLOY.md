# Deploying this fall-detection API

## Why not Vercel serverless?

This service loads **TensorFlow**, **PyTorch (via Ultralytics)**, **OpenCV**, and **large model files** (`.keras`, `.pt`, `.joblib`). Vercel serverless functions have **strict size limits** and **short timeouts**, so this stack does **not** run reliably there.

**Practical options:**

1. **Deploy the Docker image** (this repo’s `Dockerfile`) to **Render**, **Railway**, **Fly.io**, **Google Cloud Run**, or **Azure Container Apps**.
2. Use **Vercel only for a static site or Next.js UI**, and set `NEXT_PUBLIC_API_URL` to your backend URL hosted on one of the platforms above.

---

## Docker (local test)

```bash
docker build -t fall-detection .
docker run --rm -p 5000:5000 -e BIND_HOST=0.0.0.0 fall-detection
```

Open `http://localhost:5000`. Health: `GET http://localhost:5000/api/health`.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `PORT` | Listen port (set automatically on Render/Railway/Fly). |
| `BIND_HOST` | Only used when running `python app.py` locally (default `127.0.0.1`). Docker/Gunicorn binds `0.0.0.0`. |
| `CORS_ORIGIN` | Default `*` for `/api/*`. Set to your frontend origin in production if needed. |

Ensure **`models/`** and **`yolov8n-pose.pt`** are in the image (already copied by `COPY . .`).

---

## POST API (camera / mobile)

**Single image** — `multipart/form-data`:

- URL: `POST /api/detect` (same behavior as `POST /predict`)
- Fields: **`file`** or **`image`** or **`photo`** or **`frame`** (one file)
- Optional: **`model`** = `cnn` or `yolo`

**Multiple images** — `multipart/form-data`:

- URL: `POST /api/detect/batch` (same as `POST /predict/batch`)
- Repeated fields: **`files`**, **`images`**, **`frames`**, or **`photos`**
- Optional: **`model`**, **`aggregate`** = `majority` | `any` | `all`

---

## Render (recommended path)

This repo includes **`render.yaml`** so you can use a [Blueprint](https://render.com/docs/infrastructure-as-code).

1. Push the project to **GitHub / GitLab / Bitbucket**.
2. In [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect the repository; Render will read `render.yaml` and create the **web service** (`fall-detection-api`).
4. Wait for the **first build** (Docker can take 15–30+ minutes). If the build or runtime **runs out of memory**, open the service → **Settings** → **Instance type** and choose a plan with **at least 2 GB RAM** (the Blueprint uses `plan: standard`; adjust as needed).
5. When the deploy is **Live**, open the service URL and check **`GET /api/health`** (should return `{"status":"ok",...}`).

**Environment variables** (optional, in the service **Environment** tab):

- `CORS_ORIGIN` — e.g. `https://your-frontend.com` (default is `*` for `/api/*`).

**Without Blueprint:** **New** → **Web Service** → connect repo → **Runtime: Docker**, **Dockerfile path** `./Dockerfile`, **Health check path** `/api/health`.

---

## Railway / Fly (outline)

1. Connect the Git repo.
2. Choose **Docker** build (Dockerfile at repo root).
3. Set **memory** to at least **2 GB** (TensorFlow + YOLO are heavy); increase if OOM.
4. Open the service URL and hit `/api/health`.
