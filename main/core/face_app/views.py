from django.shortcuts import render
from django.conf import settings

import sys, base64
from io import BytesIO
from pathlib import Path

import joblib, cv2
import numpy as np
from PIL import Image, ImageOps

# your custom PCA class used during pickling
from ml.pca_from_scratch import PCAFromScratch
sys.modules['__main__'].PCAFromScratch = PCAFromScratch

from mtcnn import MTCNN
detector = MTCNN()

# ----------------- Load artifacts ONCE -----------------
pca       = joblib.load(settings.MODEL_DIR / "pca_final.pkl")
svm       = joblib.load(settings.MODEL_DIR / "svm_final.pkl")
knn_refs  = joblib.load(settings.MODEL_DIR / "knn_final_refs.pkl")
tau       = joblib.load(settings.MODEL_DIR / "threshold.pkl")    # numeric threshold for KNN score
tau = tau + 0.8

ALLOWED_EXTS = (".jpg", ".jpeg")
ALLOWED_CT   = {"image/jpeg"}

# ----------------- Helpers -----------------
def load_rgb_with_exif(file_obj) -> np.ndarray:
    """Read uploaded file -> RGB numpy array (HxWx3, uint8) with EXIF orientation fixed."""
    img = Image.open(file_obj)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)

def crop_face(image_rgb, min_side=40, out_size=160):
    """MTCNN → largest face → pad=0.25 → resize to out_size (match training)."""
    faces = detector.detect_faces(image_rgb)
    if not faces:
        return None
    faces.sort(key=lambda f: f['box'][2] * f['box'][3], reverse=True)
    x, y, w, h = faces[0]['box']
    if w < min_side or h < min_side:
        return None
    pad = int(0.25 * max(w, h))
    H, W = image_rgb.shape[:2]
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
    face = image_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (out_size, out_size), interpolation=cv2.INTER_AREA)

# ----------------- For KNN and SVM -----------------

def to_gray64_float01(face_rgb: np.ndarray, out_size=64) -> np.ndarray:
    """RGB face -> grayscale 64x64 -> float32 [0,1]."""
    pil = Image.fromarray(face_rgb).convert("L").resize((out_size, out_size), Image.BILINEAR)
    return (np.array(pil, dtype=np.float32) / 255.0)

def to_feature_vector(img_array: np.ndarray) -> np.ndarray:
    """Flatten to (4096,)."""
    return img_array.flatten()

def pil_to_data_url_jpeg(img_rgb: np.ndarray) -> str:
    """Return base64 data URL so we can preview the uploaded image again."""
    pil = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def knn_pred_and_score(xq, X_ref, y_ref, k=3):
    """
    Returns:
      pred_label: majority label among k-NN
      score: mean distance of neighbors that match pred_label (lower is better)
    """
    d = np.linalg.norm(X_ref - xq, axis=1)
    idx = np.argpartition(d, k-1)[:k]          # <- k-1, not k
    d_k = d[idx]
    lbl_k = y_ref[idx]
    vals, counts = np.unique(lbl_k, return_counts=True)
    pred = vals[np.argmax(counts)]
    same = (lbl_k == pred)
    score = float(d_k[same].mean())
    return pred, score

# ----------------- For FaceNet -----------------

import torch
from torchvision import transforms
# from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import InceptionResnetV1

device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()   # converts to [0,1] float tensor, channels-first
    # facenet-pytorch model handles its internal normalization
])

def embed_face_rgb(face_rgb_160: np.ndarray) -> np.ndarray:
    """Input: numpy RGB [160,160,3] -> Output: numpy [512]."""
    pil = Image.fromarray(face_rgb_160)
    x = preprocess(pil).unsqueeze(0).to(device)  # [1,3,160,160]
    emb = resnet(x)                              # [1,512]
    return emb.squeeze(0).detach().cpu().numpy()

clf = joblib.load(settings.MODEL_DIR / "facenet_knn_model.pkl")   # KNeighborsClassifier
le  = joblib.load(settings.MODEL_DIR / "label_encoder.pkl")     # LabelEncoder
TAU_facenet = joblib.load(settings.MODEL_DIR / "unknown_threshold.pkl")     # float threshold
# ----------------- Main view -----------------
def home(request):
    context = {}

    if request.method == "POST" and request.FILES.get("image"):
        f = request.FILES["image"]

        # Enforce JPG/JPEG
        if not (f.name.lower().endswith(ALLOWED_EXTS) and f.content_type in ALLOWED_CT):
            context["error"] = "Only JPG/JPEG images are allowed."
            return render(request, "front.html", context)

        # 1) Load + EXIF fix (RGB numpy)
        rgb = load_rgb_with_exif(f)
        context["preview_data_url"] = pil_to_data_url_jpeg(rgb)  # show preview again

        # 2) MTCNN crop -> 160
        face = crop_face(rgb, min_side=40, out_size=160)
        if face is None:
            context.update({"knn": "No face", "svm": "No face", "facenet": "—"})
            return render(request, "front.html", context)

        # 3) grayscale 64x64 -> flatten (4096,)
        arr64 = to_gray64_float01(face, out_size=64)           # (64,64)
        x_raw = to_feature_vector(arr64).astype(np.float64)     # (4096,)
        X_raw = x_raw.reshape(1, -1)                            # (1,4096)

        # 4) PCA for SVM (match training: transform only, no fit)
        Z = pca.transform(X_raw)                                # (1,k)

        # 5) SVM prediction
        try:
            svm_name = svm.predict(Z)[0]
        except Exception:
            svm_name = "Error"

        # 3) embed
        emb = embed_face_rgb(face).reshape(1, -1)  # [1,512]


        #facenet_prediction
        pred_id = clf.predict(emb)[0]
        label = le.inverse_transform([pred_id])[0]
        dists, _ = clf.kneighbors(emb, n_neighbors=1, return_distance=True)
        dist = float(dists[0][0])
        if dist < TAU_facenet:
            facenet_name = label
        else:
            facenet_name = "unknown_label"

        # 6) KNN prediction
        # Your refs may be in PCA space or raw space. Detect and use the right one:
        if isinstance(knn_refs, dict):
            y_ref = knn_refs.get("y_ref")
            if "X_ref" in knn_refs:
                X_ref = knn_refs["X_ref"]
                xq = Z[0]
            else:
                X_ref, xq, y_ref = None, None, None
        else:
            X_ref = xq = y_ref = None

        k = knn_refs.get("k")
        

        if X_ref is not None and y_ref is not None:
            knn_name, score = knn_pred_and_score(xq, X_ref, y_ref, k=k)
            # optional unknown gating with tau (your threshold)
            if isinstance(tau, (int, float, np.floating)) and not np.isnan(tau):
                if score > float(tau):
                    knn_name = "unknown_label"
        else:
            knn_name = "Error"

        # 8) send to template
        context.update({
            "knn": str(knn_name),
            "svm": str(svm_name),
            "facenet": str(facenet_name),
        })

    return render(request, "front.html", context)
