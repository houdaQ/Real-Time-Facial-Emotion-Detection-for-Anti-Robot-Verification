# =============================================================
# camera.py — Interface Webcam + Détection Visage
# =============================================================
# ⚠️  CE FICHIER EST LA RESPONSABILITÉ DE LA PERSONNE TÂCHE 2.
#
# Il doit exposer la classe CameraModule avec exactement cette
# interface pour que app.py puisse l'utiliser sans modification.
#
# En attendant le vrai module, un STUB de simulation est fourni.
# =============================================================

import cv2
import numpy as np


class CameraModule:
    """
    Capture vidéo + détection de visage.
    
    Interface publique attendue par app.py :
    ─────────────────────────────────────────
    cam = CameraModule()
    cam.start()
    
    frame, faces, face_rect, face_roi = cam.get_frame()
    #   frame     : image BGR complète (avec rectangle dessiné)
    #   faces     : liste de tuples (x, y, w, h) — tous les visages détectés
    #   face_rect : tuple (x, y, w, h) du visage principal, ou None
    #   face_roi  : région recadrée + préprocessée (48x48 grayscale) pour le CNN
    
    cam.release()
    """

    # ── Cascade Haar pour la détection (built-in OpenCV) ─────
    _CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(self, camera_index=0):
        self._cap     = None
        self._index   = camera_index
        self._cascade = cv2.CascadeClassifier(self._CASCADE_PATH)

    def start(self):
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._index}")

    def get_frame(self):
        """
        Retourne (frame, faces, face_rect, face_roi).
        Si aucun visage → face_rect = None, face_roi = None.
        """
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")

        frame = cv2.flip(frame, 1)   # miroir horizontal (plus naturel)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        face_rect = None
        face_roi  = None

        if len(faces) > 0:
            # Visage principal = le plus grand
            face_rect = tuple(max(faces, key=lambda f: f[2] * f[3]))
            x, y, w, h = face_rect

            # Dessin du rectangle sur le frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 100), 2)

            # Préprocessing pour le CNN (48×48 grayscale normalisé)
            roi       = gray[y:y + h, x:x + w]
            roi_48    = cv2.resize(roi, (48, 48))
            face_roi  = roi_48.astype("float32") / 255.0
            face_roi  = np.expand_dims(face_roi, axis=(0, -1))  # (1, 48, 48, 1)

        return frame, list(faces), face_rect, face_roi

    def release(self):
        if self._cap is not None:
            self._cap.release()
