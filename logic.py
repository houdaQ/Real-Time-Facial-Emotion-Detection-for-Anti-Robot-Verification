# =============================================================
# logic.py — Logique de décision PASS/FAIL + Système Anti-triche
# =============================================================

import cv2
import numpy as np
from collections import deque


# ─────────────────────────────────────────────
# ANTI-CHEAT ENGINE
# ─────────────────────────────────────────────

class AntiCheatEngine:
    """
    Détecte les tentatives de triche :
    - Photo / image statique
    - Plusieurs visages
    - Visage trop petit / trop loin
    - Absence de mouvement naturel (optical flow)
    - Visage trop proche du bord
    """

    def __init__(self):
        self.face_centers     = deque(maxlen=45)   # ~1.5s à 30fps
        self.prev_gray        = None
        self._static_counter  = 0
        self._motion_history  = deque(maxlen=20)

    # ── Vérifications individuelles ──────────────────────────

    def _check_face_count(self, faces):
        n = len(faces)
        if n == 0:
            return "NO_FACE_DETECTED"
        if n > 1:
            return "MULTIPLE_FACES"
        return None

    def _check_face_size(self, face_rect, frame_shape):
        if face_rect is None:
            return None
        x, y, w, h = face_rect
        fh, fw = frame_shape[:2]
        ratio = (w * h) / (fw * fh)
        if ratio < 0.04:
            return "FACE_TOO_SMALL"       # trop loin de la caméra
        if ratio > 0.75:
            return "FACE_TOO_CLOSE"       # trop près (peu probable mais possible)
        return None

    def _check_face_in_frame(self, face_rect, frame_shape):
        """Le visage doit être bien centré, pas coupé par les bords"""
        if face_rect is None:
            return None
        x, y, w, h = face_rect
        fh, fw = frame_shape[:2]
        margin = 10
        if x < margin or y < margin or (x + w) > (fw - margin) or (y + h) > (fh - margin):
            return "FACE_OUT_OF_FRAME"
        return None

    def _check_natural_movement(self, frame):
        """
        Optical flow minimal pour distinguer vidéo réelle d'une photo.
        Une vraie personne bouge légèrement (respiration, micro-mouvements).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_motion = float(np.mean(magnitude))

        self.prev_gray = gray
        self._motion_history.append(mean_motion)

        if len(self._motion_history) >= 20:
            avg = np.mean(self._motion_history)
            if avg < 0.08:   # quasi-aucun mouvement
                self._static_counter += 1
            else:
                self._static_counter = max(0, self._static_counter - 2)

        if self._static_counter > 45:   # ~1.5s sans mouvement
            return "STATIC_IMAGE_DETECTED"
        return None

    # ── Interface publique ────────────────────────────────────

    def check(self, frame, faces, face_rect):
        """
        Retourne une liste de codes d'alerte (vide = tout est OK).
        """
        warnings = []

        w = self._check_face_count(faces)
        if w:
            warnings.append(w)
            return warnings   # inutile de continuer sans visage valide

        if face_rect is not None:
            w = self._check_face_size(face_rect, frame.shape)
            if w: warnings.append(w)

            w = self._check_face_in_frame(face_rect, frame.shape)
            if w: warnings.append(w)

        w = self._check_natural_movement(frame)
        if w: warnings.append(w)

        return warnings

    def reset(self):
        self.face_centers.clear()
        self._motion_history.clear()
        self.prev_gray    = None
        self._static_counter = 0


# ─────────────────────────────────────────────
# GAME LOGIC
# ─────────────────────────────────────────────

class GameLogic:
    """
    Gère les rounds, les vies, le score et l'intégration anti-triche.
    """

    FRAMES_REQUIRED = 18   # frames correctes consécutives pour valider
    MAX_LIVES       = 3

    def __init__(self):
        self.lives          = self.MAX_LIVES
        self.rounds_passed  = 0
        self.correct_frames = 0
        self.anti_cheat     = AntiCheatEngine()

    # ── Propriétés ───────────────────────────────────────────

    @property
    def progress(self):
        """0.0 → 1.0 : progression vers le PASS du round courant"""
        return min(1.0, self.correct_frames / self.FRAMES_REQUIRED)

    def is_game_over(self):
        return self.lives <= 0

    # ── Mise à jour par frame ─────────────────────────────────

    def update(self, predicted_emotion, target_emotion,
               frame, faces, face_rect):
        """
        Appelée à chaque frame.

        Retourne : (status, warnings)
          status  : "OK" | "PASS" | "CHEAT"
          warnings: liste de codes d'alerte
        """

        # 1. Anti-triche
        warnings = self.anti_cheat.check(frame, faces, face_rect)
        if warnings:
            self.correct_frames = max(0, self.correct_frames - 2)
            return "CHEAT", warnings

        # 2. Comparaison émotion
        if predicted_emotion == target_emotion:
            self.correct_frames += 1
        else:
            self.correct_frames = max(0, self.correct_frames - 1)

        # 3. PASS ?
        if self.correct_frames >= self.FRAMES_REQUIRED:
            self.correct_frames = 0
            self.rounds_passed += 1
            return "PASS", []

        return "OK", []

    # ── Gestion des rounds ────────────────────────────────────

    def on_timeout(self):
        """Appelée quand le timer expire. Retourne True si des vies restent."""
        self.lives -= 1
        self.correct_frames = 0
        self.anti_cheat.reset()
        return self.lives > 0

    def reset_for_new_round(self):
        """Remet à zéro pour un nouveau round (garde les vies et le score)."""
        self.correct_frames = 0
        self.anti_cheat.reset()

    def full_reset(self):
        """Remet complètement à zéro le jeu."""
        self.lives          = self.MAX_LIVES
        self.rounds_passed  = 0
        self.correct_frames = 0
        self.anti_cheat.reset()


# ─────────────────────────────────────────────
# MESSAGES D'ALERTE (pour l'UI)
# ─────────────────────────────────────────────

WARNING_MESSAGES = {
    "NO_FACE_DETECTED":      "No face detected — look at the camera",
    "MULTIPLE_FACES":        "Multiple faces detected!",
    "FACE_TOO_SMALL":        "Move closer to the camera",
    "FACE_TOO_CLOSE":        "Move back a little",
    "FACE_OUT_OF_FRAME":     "Center your face in the frame",
    "STATIC_IMAGE_DETECTED": "Live camera required — no photos!",
}

def get_warning_text(code):
    return WARNING_MESSAGES.get(code, code)
