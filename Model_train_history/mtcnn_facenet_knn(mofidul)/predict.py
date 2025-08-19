# predict.py
# Usage: python predict.py path/to/image.jpg
# (Place this file in the same folder as your saved artifacts.)

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps
import joblib
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN

# --------- Load artifacts (saved separately) ----------
MODEL_PATH = "facenet_knn_model.pkl"
ENC_PATH   = "label_encoder.pkl"
TAU_PATH   = "unknown_threshold.pkl"

clf = joblib.load(MODEL_PATH)   # KNeighborsClassifier
le  = joblib.load(ENC_PATH)     # LabelEncoder
TAU = joblib.load(TAU_PATH)     # float threshold

# --------- Networks / preprocessing (same as notebook) ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)
detector = MTCNN()

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),   # facenet-pytorch model handles normalization internally
])

# --------- Helpers (same as notebook) ----------
def load_rgb_with_exif(path: Path):
    """Read as RGB and fix orientation via EXIF."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)  # RGB uint8

def crop_face(image_rgb, min_side=40, out_size=160):
    """
    Detect largest face, add proportional padding (25% of face size),
    clamp to bounds, resize to out_size, return RGB crop (or None).
    """
    faces = detector.detect_faces(image_rgb)
    if not faces:
        return None

    # largest face by area
    faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
    x, y, w, h = faces[0]["box"]
    if w < min_side or h < min_side:
        return None

    pad = int(0.25 * max(w, h))        # 25% margin like training
    H, W = image_rgb.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    face = image_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None

    # keep RGB; resize to 160x160
    face = cv2.resize(face, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return face

@torch.no_grad()
def embed_face_rgb(face_rgb_160: np.ndarray) -> np.ndarray:
    """Input: numpy RGB [160,160,3] -> Output: numpy [512]."""
    pil = Image.fromarray(face_rgb_160)
    x = preprocess(pil).unsqueeze(0).to(device)  # [1,3,160,160]
    emb = resnet(x)                              # [1,512]
    return emb.squeeze(0).cpu().numpy()

def predict_one(image_path: Path, tau: float = TAU):
    """
    End-to-end prediction for one image path.
    Returns dict with: ok, is_unknown, pred_label (or None), nearest_dist, tau_used, reason (if any)
    """
    # 1) load + EXIF fix
    rgb = load_rgb_with_exif(image_path)

    # 2) detect & crop (25% margin)
    face = crop_face(rgb, out_size=160)
    if face is None:
        return {"ok": False, "reason": "no_face_detected"}

    # 3) embed
    emb = embed_face_rgb(face).reshape(1, -1)  # [1,512]

    # 4) KNN predict + distance
    pred_id = clf.predict(emb)[0]
    label   = le.inverse_transform([pred_id])[0]
    dists, _ = clf.kneighbors(emb, n_neighbors=1, return_distance=True)
    dist = float(dists[0][0])

    is_unknown = (tau is not None and dist > float(tau))
    return {
        "ok": True,
        "is_unknown": is_unknown,
        "pred_label": None if is_unknown else label,
        "nearest_dist": dist,
        "tau_used": float(tau) if tau is not None else None,
    }

def main():

    img_path = Path("test4.jpg")
    if not img_path.exists():
        print(f"File not found: {img_path}")
        sys.exit(1)

    result = predict_one(img_path, tau=TAU)
    if not result["ok"]:
        print("Prediction failed:", result.get("reason"))
        sys.exit(2)

    if result["is_unknown"]:
        print(f"Unknown face (dist={result['nearest_dist']:.3f} > tau={result['tau_used']:.3f})")
    else:
        print(f"Predicted: {result['pred_label']} (dist={result['nearest_dist']:.3f} ≤ tau={result['tau_used']:.3f})")

if __name__ == "__main__":
    main()
