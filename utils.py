# ============================================================
# utils.py — Random emotion + Timer
# ============================================================

import random
import time

# ── Ordre exact des labels du modèle emotion_model__.keras ──
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ── Émotions jouables (les plus visibles à la caméra) ────────
PLAYABLE_EMOTIONS = ['Happy', 'Sad']

# ── Emojis ───────────────────────────────────────────────────
EMOTION_EMOJIS = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😄',
    'Sad':      '😢',
    'Surprise': '😮',
    'Neutral':  '😐',
}

# ── Couleurs CSS ─────────────────────────────────────────────
EMOTION_COLORS = {
    'Happy':    '#22c55e',
    'Sad':      '#60a5fa',
    'Angry':    '#f87171',
    'Surprise': '#f59e0b',
    'Neutral':  '#94a3b8',
    'Fear':     '#a78bfa',
    'Disgust':  '#34d399',
}

# ── Conseils utilisateur ─────────────────────────────────────
EMOTION_TIPS = {
    'Happy':    'Souris largement, montre tes dents !',
    'Sad':      'Fronce les sourcils, baisse les coins de la bouche.',
    'Angry':    'Serre la mâchoire, plisse le front.',
    'Surprise': 'Ouvre grands les yeux et la bouche.',
    'Neutral':  'Visage détendu, aucune expression.',
    'Fear':     'Grands yeux, sourcils levés, bouche entrouverte.',
    'Disgust':  'Plisse le nez, lèvre supérieure relevée.',
}


def get_random_emotion(exclude=None):
    """Retourne une émotion aléatoire différente de exclude."""
    pool = [e for e in PLAYABLE_EMOTIONS if e != exclude]
    return random.choice(pool)


# ── Timer ────────────────────────────────────────────────────

class GameTimer:
    """Timer simple basé sur time.time()."""

    def __init__(self, limit: int = 15):
        self.limit  = limit
        self._start = None

    def start(self):
        self._start = time.time()

    def reset(self, new_limit: int = None):
        if new_limit:
            self.limit = new_limit
        self._start = time.time()

    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.time() - self._start

    def remaining(self) -> float:
        return max(0.0, self.limit - self.elapsed())

    def is_expired(self) -> bool:
        return self.elapsed() >= self.limit

    def progress(self) -> float:
        """0.0 = début, 1.0 = temps écoulé."""
        return min(1.0, self.elapsed() / self.limit)
