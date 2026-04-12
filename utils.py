# =============================================================
# utils.py — Outils partagés : émotions, emojis, timer
# =============================================================

import random
import time

# ─────────────────────────────────────────────
# CONFIGURATION DES ÉMOTIONS
# ─────────────────────────────────────────────

EMOTIONS = {
    "happy":     {"emoji": ":)",  "color": (0, 220, 100),  "label": "HAPPY"},
    "sad":       {"emoji": ":(",  "color": (200, 100, 0),   "label": "SAD"},
    "angry":     {"emoji": ">:(", "color": (0, 50, 220),    "label": "ANGRY"},
    "surprised": {"emoji": ":O",  "color": (0, 200, 255),   "label": "SURPRISED"},
    "neutral":   {"emoji": ":|",  "color": (180, 180, 180), "label": "NEUTRAL"},
    "fear":      {"emoji": "D:",  "color": (200, 50, 200),  "label": "FEAR"},
    "disgust":   {"emoji": ":x",  "color": (50, 180, 50),   "label": "DISGUST"},
}

EMOTION_LIST = list(EMOTIONS.keys())

def get_random_emotion(exclude=None):
    """Retourne une émotion aléatoire (différente de exclude)"""
    choices = [e for e in EMOTION_LIST if e != exclude]
    return random.choice(choices)

def get_emotion_color(emotion):
    return EMOTIONS.get(emotion, {}).get("color", (255, 255, 255))

def get_emotion_emoji(emotion):
    return EMOTIONS.get(emotion, {}).get("emoji", "?")

def get_emotion_label(emotion):
    return EMOTIONS.get(emotion, {}).get("label", emotion.upper())


# ─────────────────────────────────────────────
# TIMER
# ─────────────────────────────────────────────

class GameTimer:
    """Timer avec gestion de progression et d'expiration"""

    def __init__(self, limit=10):
        self.limit = limit
        self._start = None

    def start(self):
        self._start = time.time()

    def reset(self, new_limit=None):
        if new_limit is not None:
            self.limit = new_limit
        self._start = time.time()

    def elapsed(self):
        if self._start is None:
            return 0.0
        return time.time() - self._start

    def remaining(self):
        return max(0.0, self.limit - self.elapsed())

    def is_expired(self):
        return self.elapsed() >= self.limit

    def progress(self):
        """0.0 → début, 1.0 → temps écoulé"""
        return min(1.0, self.elapsed() / self.limit)

    def urgency_color(self):
        """Couleur BGR qui vire au rouge en fin de temps"""
        r = self.remaining()
        if r > self.limit * 0.5:
            return (0, 200, 255)   # cyan
        elif r > self.limit * 0.25:
            return (0, 140, 255)   # orange
        else:
            return (0, 60, 255)    # rouge
