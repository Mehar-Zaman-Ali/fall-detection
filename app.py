"""Fall Detection Web App - CNN and YOLOv8+Pose models side by side."""
import os
import uuid
import numpy as np
import cv2
import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
# Single-image UI uses modest limit; camera bursts use /predict/batch (raise if you send huge bursts).
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "static", "uploads")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
IMG_SIZE = 160
BASE_DIR = os.path.dirname(__file__)
CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "fall_image_classifier.keras")
POSE_RF_PATH = os.path.join(BASE_DIR, "models", "yolo_pose_rf_classifier.joblib")

cnn_model = None
pose_yolo = None
pose_rf = None

FEATURE_COLS = [
    "body_angle", "height_ratio", "shoulder_hip_dist", "hip_ankle_dist",
    "shoulder_angle", "hip_angle", "knee_angle_left", "knee_angle_right",
    "head_hip_ratio", "body_compactness",
]


def get_cnn():
    global cnn_model
    if cnn_model is None:
        from tensorflow import keras

        cnn_model = keras.models.load_model(CNN_MODEL_PATH)
        print(f"CNN model loaded from {CNN_MODEL_PATH}")
    return cnn_model


def get_pose_models():
    global pose_yolo, pose_rf
    if pose_yolo is None:
        from ultralytics import YOLO

        pose_yolo = YOLO("yolov8n-pose.pt")
        print("YOLOv8-pose model loaded")
    if pose_rf is None:
        pose_rf = joblib.load(POSE_RF_PATH)
        print(f"Pose RF classifier loaded from {POSE_RF_PATH}")
    return pose_yolo, pose_rf


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── CNN prediction ──────────────────────────────────────────────────────────

def predict_cnn(filepath):
    from tensorflow.keras.utils import img_to_array, load_img

    img = load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    m = get_cnn()
    prediction = m.predict(img_array, verbose=0)[0][0]
    is_fall = prediction >= 0.80
    confidence = prediction if is_fall else 1 - prediction

    return {
        "label": "FALL DETECTED" if is_fall else "NO FALL",
        "is_fall": bool(is_fall),
        "confidence": float(confidence * 100),
        "raw_score": float(prediction),
    }


# ── YOLOv8+Pose helpers ────────────────────────────────────────────────────

def _angle(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))


def _joint_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def _mid(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def extract_pose_features(kp_data, bbox, img_shape):
    kp = kp_data[:, :2]
    conf = kp_data[:, 2]

    l_sh, r_sh = kp[5], kp[6]
    l_hip, r_hip = kp[11], kp[12]
    l_knee, r_knee = kp[13], kp[14]
    l_ankle, r_ankle = kp[15], kp[16]
    nose = kp[0]

    sh_mid = _mid(l_sh, r_sh)
    hip_mid = _mid(l_hip, r_hip)
    ankle_mid = _mid(l_ankle, r_ankle)

    bx1, by1, bx2, by2 = bbox
    bw, bh = max(bx2 - bx1, 1), max(by2 - by1, 1)
    diag = np.sqrt(bw**2 + bh**2)

    torso_angle = abs(_angle(hip_mid, sh_mid) + 90) % 180
    body_angle = min(torso_angle, 180 - torso_angle)

    return {
        "body_angle": body_angle,
        "height_ratio": bh / bw,
        "shoulder_hip_dist": np.sqrt((sh_mid[0] - hip_mid[0])**2 + (sh_mid[1] - hip_mid[1])**2) / diag,
        "hip_ankle_dist": np.sqrt((hip_mid[0] - ankle_mid[0])**2 + (hip_mid[1] - ankle_mid[1])**2) / diag,
        "shoulder_angle": abs(_angle(l_sh, r_sh)),
        "hip_angle": abs(_angle(l_hip, r_hip)),
        "knee_angle_left": _joint_angle(l_hip, l_knee, l_ankle) if all(conf[[11, 13, 15]] > 0.3) else 180.0,
        "knee_angle_right": _joint_angle(r_hip, r_knee, r_ankle) if all(conf[[12, 14, 16]] > 0.3) else 180.0,
        "head_hip_ratio": abs(nose[1] - hip_mid[1]) / max(abs(nose[1] - ankle_mid[1]), 1),
        "body_compactness": (np.std(kp[conf > 0.3][:, 0]) * np.std(kp[conf > 0.3][:, 1])) / (bw * bh + 1e-8) if (conf > 0.3).sum() > 2 else 0.0,
    }


def predict_pose(filepath, annotated_filename=None):
    yolo, rf = get_pose_models()
    results = yolo(filepath, verbose=False)
    r = results[0]

    annotated = r.plot()
    ann_filename = annotated_filename if annotated_filename else "pose_" + os.path.basename(filepath)
    ann_path = os.path.join(app.config["UPLOAD_FOLDER"], ann_filename)
    cv2.imwrite(ann_path, annotated)

    if r.keypoints is None or len(r.keypoints.data) == 0:
        return {
            "label": "NO PERSON DETECTED",
            "is_fall": False,
            "confidence": 0.0,
            "raw_score": 0.0,
            "pose_image_url": f"/static/uploads/{ann_filename}",
        }

    best_idx = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes.conf) > 0 else 0
    kp = r.keypoints.data[best_idx].cpu().numpy()
    bbox = r.boxes.xyxy[best_idx].cpu().numpy() if r.boxes is not None else [0, 0, r.orig_shape[1], r.orig_shape[0]]

    feats = extract_pose_features(kp, bbox, r.orig_shape)
    feat_vec = np.array([[feats[c] for c in FEATURE_COLS]])

    proba = rf.predict_proba(feat_vec)[0]
    fall_prob = float(proba[1])
    is_fall = fall_prob >= 0.80
    confidence = fall_prob if is_fall else 1 - fall_prob

    return {
        "label": "FALL DETECTED" if is_fall else "NO FALL",
        "is_fall": is_fall,
        "confidence": float(confidence * 100),
        "raw_score": fall_prob,
        "pose_image_url": f"/static/uploads/{ann_filename}",
    }


def aggregate_multi_frame(per_image, mode):
    """Combine per-frame predictions. `mode`: any | majority | all."""
    n = len(per_image)
    if n == 0:
        return {
            "is_fall": False,
            "label": "NO INPUT",
            "confidence": 0.0,
            "frames_with_fall": 0,
            "total_frames": 0,
            "method": mode,
            "max_raw_score": 0.0,
        }

    falls = sum(1 for r in per_image if r.get("is_fall"))
    scores = [float(r.get("raw_score") or 0.0) for r in per_image]
    max_raw = max(scores) if scores else 0.0

    mode = (mode or "majority").lower()
    if mode == "any":
        is_fall = falls >= 1
    elif mode == "all":
        is_fall = falls == n
    else:
        is_fall = falls > n / 2

    consensus = (falls / n) * 100.0
    return {
        "is_fall": bool(is_fall),
        "label": "FALL DETECTED" if is_fall else "NO FALL",
        "confidence": float(round(consensus, 2)),
        "frames_with_fall": falls,
        "total_frames": n,
        "method": mode,
        "max_raw_score": float(max_raw),
    }


# ── HTTP helpers (camera / mobile clients) ─────────────────────────────────

def _get_single_upload_file():
    """First non-empty upload from common multipart field names."""
    for key in ("file", "image", "photo", "picture", "frame"):
        if key not in request.files:
            continue
        f = request.files[key]
        if f and getattr(f, "filename", None):
            return f
    return None


def _get_batch_upload_files():
    """Collect all parts from repeated fields (camera bursts)."""
    seen_ids = set()
    out = []
    for key in ("files", "images", "frames", "photos", "frame"):
        for f in request.files.getlist(key):
            if not f or not getattr(f, "filename", None):
                continue
            # dedupe same Storage object if client duplicated keys
            fid = id(f)
            if fid in seen_ids:
                continue
            seen_ids.add(fid)
            out.append(f)
    return out


def _run_single_prediction():
    file = _get_single_upload_file()
    if file is None:
        return None, (jsonify({"error": "No image in POST body. Use multipart/form-data with field: file, image, photo, or frame."}), 400)
    if not allowed_file(file.filename):
        return None, (jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, WEBP, or BMP."}), 400)

    model_type = request.form.get("model", "cnn")
    ext = file.filename.rsplit(".", 1)[1].lower() if "." in file.filename else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    if model_type == "yolo":
        result = predict_pose(filepath)
    else:
        result = predict_cnn(filepath)

    result["image_url"] = f"/static/uploads/{filename}"
    result["model"] = model_type
    return result, None


def _run_batch_prediction():
    files = _get_batch_upload_files()
    if not files:
        return None, (
            jsonify(
                {
                    "error": "No images in POST body. Use multipart/form-data with repeated field: files, images, frames, or photos."
                }
            ),
            400,
        )

    model_type = request.form.get("model", "cnn")
    aggregate = request.form.get("aggregate", "majority")
    if aggregate.lower() not in ("any", "majority", "all"):
        aggregate = "majority"

    per_image = []
    for idx, file in enumerate(files):
        if not allowed_file(file.filename):
            return None, (jsonify({"error": f"Invalid file type: {file.filename}"}), 400)

        ext = file.filename.rsplit(".", 1)[1].lower()
        stored_name = f"{uuid.uuid4().hex}_{idx}.{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        file.save(filepath)

        if model_type == "yolo":
            ann_name = f"pose_{uuid.uuid4().hex}_{idx}.jpg"
            result = predict_pose(filepath, annotated_filename=ann_name)
        else:
            result = predict_cnn(filepath)

        result["image_url"] = f"/static/uploads/{stored_name}"
        result["index"] = idx
        result["original_filename"] = file.filename
        result["model"] = model_type
        per_image.append(result)

    agg = aggregate_multi_frame(per_image, aggregate)
    payload = {
        "aggregate": agg,
        "per_image": per_image,
        "model": model_type,
        "aggregate_mode": aggregate.lower(),
    }
    return payload, None


@app.after_request
def _cors_for_api(response):
    if request.path.startswith("/api/"):
        response.headers["Access-Control-Allow-Origin"] = os.environ.get("CORS_ORIGIN", "*")
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS, GET"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "fall-detection"})


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def _predict_single():
    result, err = _run_single_prediction()
    if err:
        return err
    return jsonify(result)


def _predict_batch():
    payload, err = _run_batch_prediction()
    if err:
        return err
    return jsonify(payload)


@app.route("/predict", methods=["POST"])
def predict():
    return _predict_single()


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Accept multiple images (e.g. camera frames). Form: files[] or images[], optional model, aggregate.

    aggregate:
      - majority (default): fall if more than half of frames predict fall
      - any: fall if any frame predicts fall
      - all: fall only if every frame predicts fall
    """
    return _predict_batch()


@app.route("/api/detect", methods=["POST", "OPTIONS"])
def api_detect():
    """JSON API: same as POST /predict — multipart image field: file | image | photo | frame."""
    if request.method == "OPTIONS":
        return "", 204
    return _predict_single()


@app.route("/api/detect/batch", methods=["POST", "OPTIONS"])
def api_detect_batch():
    """JSON API: same as POST /predict/batch — repeated files | images | frames | photos."""
    if request.method == "OPTIONS":
        return "", 204
    return _predict_batch()


if __name__ == "__main__":
    get_cnn()
    get_pose_models()
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("BIND_HOST", "127.0.0.1")
    print(f"\n  Fall Detection App running at: http://{host}:{port}\n")
    app.run(debug=False, host=host, port=port)
