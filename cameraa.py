# ============================================================
# cameraa.py — Webcam + Détection visage + Inférence CNN
# ============================================================
#
#  Fixes v3 :
#    ✔ Suppression du threading (model.predict non thread-safe TF)
#    ✔ CLAHE correctement défini (fix NameError: equalized)
#    ✔ INFERENCE_EVERY_N = 12 (CNN ~1x/seconde à 12fps)
#    ✔ Redimensionnement frame avant Haar → détection 3x plus rapide
# ============================================================

import os
import random
import zipfile
import json
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict

import cv2
import numpy as np

from utils import EMOTION_LABELS, PLAYABLE_EMOTIONS

# ── MODE SIMULATION ──────────────────────────────────────────
SIMULATION_MODE = False

# ── Chemin modèle ────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "model", "emotion_model.keras")

# ── Haar Cascade ─────────────────────────────────────────────
_CASCADE  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_detector = cv2.CascadeClassifier(_CASCADE)

IMG_SIZE          = (48, 48)
SMOOTHING_WINDOW  = 3    # réduit → s'adapte plus vite aux changements d'émotion
INFERENCE_EVERY_N = 8    # CNN toutes les 8 frames
MAX_FACES_ALLOWED = 1


# ============================================================
#  DATACLASS RÉSULTAT
# ============================================================

@dataclass
class CameraResult:
    emotion:    Optional[str]
    confidence: float
    probas:     Dict[str, float]
    face_found: bool
    multi_face: bool = False


# ============================================================
#  ÉTAT INTERNE
# ============================================================

class _InferenceState:
    def __init__(self):
        self.smoother_window: deque      = deque(maxlen=SMOOTHING_WINDOW)
        self.frame_counter:   int        = 0
        self.last_probs:      np.ndarray = (
            np.ones(len(EMOTION_LABELS), dtype="float32") / len(EMOTION_LABELS)
        )
        self.last_emotion:    str        = "Neutral"
        self.last_confidence: float      = 0.0

    def reset(self):
        self.smoother_window.clear()
        self.frame_counter   = 0
        self.last_emotion    = "Neutral"
        self.last_confidence = 0.0
        self.last_probs      = (
            np.ones(len(EMOTION_LABELS), dtype="float32") / len(EMOTION_LABELS)
        )


_state = _InferenceState()


# ============================================================
#  1. CHARGEMENT DU MODÈLE
# ============================================================

def _rebuild_architecture():
    import tensorflow as tf
    kl = tf.keras.layers
    return tf.keras.Sequential([
        tf.keras.Input(shape=(48, 48, 1), name="conv2d_21_input"),
        kl.Conv2D(64,  (5, 5), activation="relu",             name="conv2d_21"),
        kl.MaxPooling2D((5, 5), strides=(2, 2),               name="max_pooling2d_6"),
        kl.Conv2D(64,  (3, 3), activation="relu",             name="conv2d_22"),
        kl.Conv2D(64,  (3, 3), activation="relu",             name="conv2d_23"),
        kl.AveragePooling2D((3, 3), strides=(2, 2),           name="average_pooling2d_2"),
        kl.Conv2D(128, (3, 3), activation="relu",             name="conv2d_24"),
        kl.Conv2D(128, (3, 3), activation="relu",             name="conv2d_25"),
        kl.AveragePooling2D((3, 3), strides=(2, 2),           name="average_pooling2d_3"),
        kl.Flatten(                                            name="flatten_3"),
        kl.Dense(1024, activation="relu",                     name="dense_3"),
        kl.Dropout(0.2,                                       name="dropout_4"),
        kl.Dense(1024, activation="relu",                     name="dense_4"),
        kl.Dropout(0.2,                                       name="dropout_5"),
        kl.Dense(7,    activation="softmax",                  name="dense_5"),
    ], name="sequential_2")


def _fix_keras_config_and_reload(path: str):
    import tensorflow as tf
    fixed_path = path.replace(".keras", "_fixed.keras")
    if not os.path.exists(fixed_path):
        def _fix(obj):
            if isinstance(obj, dict):
                if (obj.get("class_name") == "InputLayer"
                        and "batch_input_shape" in obj.get("config", {})):
                    obj["config"]["batch_shape"] = obj["config"].pop("batch_input_shape")
                else:
                    obj.get("config", {}).pop("batch_input_shape", None)
                return {k: _fix(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_fix(i) for i in obj]
            return obj
        with zipfile.ZipFile(path, "r") as zin:
            with zipfile.ZipFile(fixed_path, "w", zipfile.ZIP_STORED) as zout:
                for name in zin.namelist():
                    data = zin.read(name)
                    if name == "config.json":
                        data = json.dumps(
                            _fix(json.loads(data.decode("utf-8")))
                        ).encode("utf-8")
                    zout.writestr(name, data)
    return tf.keras.models.load_model(fixed_path, compile=False)


def _warmup(model) -> None:
    """
    Lance une prédiction factice au chargement.
    TF compile le graphe une seule fois ici → les vraies inférences
    seront immédiatement rapides (pas de délai à la première frame).
    """
    dummy = np.zeros((1, 48, 48, 1), dtype="float32")
    model.predict(dummy, verbose=0)
    print("[camera] ✅ Warmup OK — inférences rapides dès la première frame.")


def load_model():
    if SIMULATION_MODE:
        return None
    if not os.path.exists(MODEL_PATH):
        print(f"[camera] ⚠️  Introuvable : {MODEL_PATH} → simulation.")
        return None

    print(f"[camera] Chargement : {MODEL_PATH}")
    import tensorflow as tf

    # Tentative 1 — direct
    try:
        m = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"[camera] ✅ Direct. Input : {m.input_shape}")
        _warmup(m)
        return m
    except Exception as e1:
        print(f"[camera] Direct échoué ({type(e1).__name__}). Correction config...")

    # Tentative 2 — correction batch_input_shape
    try:
        m = _fix_keras_config_and_reload(MODEL_PATH)
        print(f"[camera] ✅ Config corrigé. Input : {m.input_shape}")
        _warmup(m)
        return m
    except Exception as e2:
        print(f"[camera] Config échoué ({type(e2).__name__}). Rebuild...")

    # Tentative 3 — rebuild + poids dynamiques
    try:
        import tempfile
        with zipfile.ZipFile(MODEL_PATH, "r") as z:
            h5_files = [f for f in z.namelist() if f.endswith(".h5")]
            print(f"[camera] Archive : {z.namelist()}")
            if not h5_files:
                raise FileNotFoundError(f"Aucun .h5 dans l'archive")
            with z.open(h5_files[0]) as src:
                with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
                    tmp.write(src.read())
                    tmp_path = tmp.name
        m = _rebuild_architecture()
        m.load_weights(tmp_path)
        os.remove(tmp_path)
        print(f"[camera] ✅ Reconstruit. Input : {m.input_shape}")
        _warmup(m)
        return m
    except Exception as e3:
        print(f"[camera] ❌ Échec total : {e3}")
        return None


# ============================================================
#  2. PREPROCESSING  (CLAHE — bug corrigé)
# ============================================================

def _preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    """Resize → CLAHE → normalise → reshape (1,48,48,1)."""
    resized   = cv2.resize(face_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(resized)                   # ← correctement défini
    normalized = equalized.astype("float32") / 255.0
    return normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)


# ============================================================
#  3. DÉTECTION VISAGE
# ============================================================

def _detect_faces(image_rgb: np.ndarray):
    """
    Réduit l'image à 320px de large avant la détection Haar
    → 3-4× plus rapide sur CPU, sans perte de précision.
    """
    h, w = image_rgb.shape[:2]
    scale = min(1.0, 320 / w)                         # réduit si > 320px
    small = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))

    gray  = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    faces = _detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    n_faces = len(faces) if len(faces) > 0 else 0
    if n_faces == 0:
        return 0, None

    # Remettre à l'échelle originale pour le crop
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    x  = int(x  / scale)
    y  = int(y  / scale)
    fw = int(fw / scale)
    fh = int(fh / scale)

    gray_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return n_faces, gray_orig[y:y + fh, x:x + fw]


# ============================================================
#  4. LISSAGE TEMPOREL
# ============================================================

def _smooth_probs(raw: np.ndarray) -> np.ndarray:
    _state.smoother_window.append(raw.copy())
    return np.mean(_state.smoother_window, axis=0)


# ============================================================
#  5. SIMULATION
# ============================================================

def _simulate_prediction(target_hint=None) -> CameraResult:
    predicted = (
        target_hint if (target_hint and random.random() < 0.55)
        else random.choice(PLAYABLE_EMOTIONS)
    )
    raw      = np.random.dirichlet(np.ones(len(EMOTION_LABELS)) * 0.5)
    idx      = EMOTION_LABELS.index(predicted)
    raw[idx] += random.uniform(0.3, 0.6)
    raw       = raw / raw.sum()
    return CameraResult(
        emotion    = predicted,
        confidence = float(raw[idx]),
        probas     = {EMOTION_LABELS[i]: float(raw[i]) for i in range(len(EMOTION_LABELS))},
        face_found = True,
        multi_face = False,
    )


# ============================================================
#  6. PRÉDICTION PRINCIPALE
# ============================================================

def predict_from_image(model, image_rgb: np.ndarray,
                       target_hint: str = None) -> CameraResult:
    """
    Appelée à chaque frame.

    Stratégie performance :
      • Haar sur image réduite à 320px → rapide
      • CNN toutes les INFERENCE_EVERY_N frames (pas à chaque frame)
      • Entre les frames CNN → résultat en cache, retour immédiat
      • Pas de thread → predict() toujours dans le thread principal TF
    """
    _state.frame_counter += 1

    n_faces, face_gray = _detect_faces(image_rgb)

    if n_faces == 0:
        _state.reset()
        return CameraResult(emotion=None, confidence=0.0, probas={},
                            face_found=False)

    if n_faces > MAX_FACES_ALLOWED:
        _state.reset()
        return CameraResult(emotion=None, confidence=0.0, probas={},
                            face_found=False, multi_face=True)

    if SIMULATION_MODE or model is None:
        return _simulate_prediction(target_hint)

    # ── CNN toutes les N frames ou au premier appel ───────────
    run_cnn = (
        _state.frame_counter % INFERENCE_EVERY_N == 0
        or len(_state.smoother_window) == 0
    )

    if run_cnn and face_gray is not None:
        tensor = _preprocess_face(face_gray)
        raw    = model.predict(tensor, verbose=0)[0]
        smooth = _smooth_probs(raw)
        top      = int(np.argmax(smooth))
        _state.last_emotion    = EMOTION_LABELS[top]
        _state.last_confidence = float(smooth[top])
        _state.last_probs      = smooth

    probas = {
        EMOTION_LABELS[i]: float(_state.last_probs[i])
        for i in range(len(EMOTION_LABELS))
    }

    return CameraResult(
        emotion    = _state.last_emotion,
        confidence = _state.last_confidence,
        probas     = probas,
        face_found = True,
        multi_face = False,
    )
