# ============================================================
# logic.py — PASS / FAIL + Anti-triche + Gestion de session
# ============================================================

import time
import streamlit as st

from utils import get_random_emotion

# ── Configuration ────────────────────────────────────────────
TIME_LIMIT      = 20      # secondes — augmenté pour plus de confort
MAX_LIVES       = 3       # 3 tentatives
STREAK_REQUIRED = 1       # 1 bonne détection CNN suffit
CONFIDENCE_MIN  = 0.35    # seuil confiance minimum
ROUNDS_TO_WIN   = 1       # 1 seul round

# ── Codes résultat ────────────────────────────────────────────
OK       = "ok"
VERIFIED = "verified"
TIMEOUT  = "timeout"
FAIL     = "fail"
NO_FACE  = "no_face"
LOW_CONF = "low_conf"
WRONG    = "wrong"
PASS     = "pass"

CHEAT_MESSAGES = {
    NO_FACE:           "⚠️  Aucun visage détecté — centrez votre visage",
    LOW_CONF:          "⚠️  Expression peu claire — soyez plus expressif !",
    "COVERING_CAMERA": "🚫  Caméra obstruée — veuillez dégager la vue",
}


# ════════════════════════════════════════════════════════════
# SESSION
# ════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "screen":          "welcome",
        "lives":           MAX_LIVES,
        "rounds_passed":   0,
        "target_emotion":  None,
        "correct_streak":  0,
        "result_kind":     None,
        "last_emotion":    None,
        "last_confidence": 0.0,
        "last_probas":     {},
        "face_found":      False,
        "cheat_flags":     [],
        "capture_count":   0,
        "no_face_streak":  0,
        # ── Nouveau : la caméra est-elle ouverte ? ───────────
        "camera_active":   False,        # True seulement après clic bouton
        "_round_start":    None,         # None = timer pas encore lancé
        "last_result_id":  0,            # ID du dernier résultat CNN traité
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ════════════════════════════════════════════════════════════
# NAVIGATION
# ════════════════════════════════════════════════════════════

def start_game():
    """
    Passe à l'écran challenge et choisit l'émotion cible.
    ⚠️  Le timer N'EST PAS démarré ici — il démarre seulement
        quand l'utilisateur clique sur "Ouvrir la caméra".
    """
    st.session_state.screen         = "challenge"
    st.session_state.lives          = MAX_LIVES
    st.session_state.rounds_passed  = 0
    st.session_state.target_emotion = get_random_emotion()
    st.session_state.correct_streak = 0
    st.session_state.capture_count  = 0
    st.session_state.no_face_streak = 0
    st.session_state.cheat_flags    = []
    st.session_state.last_emotion   = None
    st.session_state.result_kind    = None
    st.session_state.camera_active  = False   # caméra fermée au départ
    st.session_state._round_start   = None    # timer pas encore lancé
    st.session_state.last_result_id = 0


def start_round_timer():
    """
    Démarre le timer du round.
    Appelé quand l'utilisateur clique sur "Ouvrir la caméra".
    """
    st.session_state._round_start  = time.time()
    st.session_state.camera_active = True


def next_round():
    """Rejouer (après timeout, une vie en moins)."""
    prev = st.session_state.target_emotion
    st.session_state.screen         = "challenge"
    st.session_state.target_emotion = get_random_emotion(exclude=prev)
    st.session_state.correct_streak = 0
    st.session_state.capture_count  = 0
    st.session_state.no_face_streak = 0
    st.session_state.cheat_flags    = []
    st.session_state.last_emotion   = None
    st.session_state.result_kind    = None
    st.session_state.camera_active  = False   # doit recliquer sur le bouton
    st.session_state._round_start   = None
    st.session_state.last_result_id = 0


def restart():
    """Remet tout à zéro → welcome."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session()


# ════════════════════════════════════════════════════════════
# TIMER
# ════════════════════════════════════════════════════════════

def time_remaining() -> float:
    start = st.session_state.get("_round_start", None)
    if start is None:
        return float(TIME_LIMIT)          # timer pas encore lancé
    return max(0.0, TIME_LIMIT - (time.time() - start))


def time_progress() -> float:
    """0.0 = début, 1.0 = temps écoulé."""
    start = st.session_state.get("_round_start", None)
    if start is None:
        return 0.0
    return min(1.0, (time.time() - start) / TIME_LIMIT)


def is_expired() -> bool:
    """False si le timer n'est pas encore démarré."""
    start = st.session_state.get("_round_start", None)
    if start is None:
        return False
    return (time.time() - start) >= TIME_LIMIT


# ════════════════════════════════════════════════════════════
# ANTI-TRICHE
# ════════════════════════════════════════════════════════════

def _check_anti_cheat(emotion, confidence, face_found) -> list:
    flags = []
    if not face_found:
        st.session_state.no_face_streak += 1
        flags.append(NO_FACE)
        if st.session_state.no_face_streak >= 5:
            flags.append("COVERING_CAMERA")
    else:
        st.session_state.no_face_streak = 0
    if face_found and confidence < CONFIDENCE_MIN:
        flags.append(LOW_CONF)
    return flags


# ════════════════════════════════════════════════════════════
# LOGIQUE PRINCIPALE — appelée uniquement sur nouveau résultat CNN
# ════════════════════════════════════════════════════════════

def process_capture(emotion: str, confidence: float, face_found: bool) -> str:
    """
    Appelée UNE SEULE FOIS par nouveau résultat CNN (pas à chaque rerun).
    Le contrôle de fréquence est fait dans app.py via result_id.
    """
    st.session_state.capture_count += 1

    # 1. Timeout
    if is_expired():
        return _handle_timeout()

    # 2. Anti-triche
    flags = _check_anti_cheat(emotion, confidence, face_found)
    st.session_state.cheat_flags = flags
    if flags:
        # Ne décrémente PAS le streak — on pénalise seulement si assez longtemps
        return flags[0]

    # 3. Comparaison émotion
    target = st.session_state.target_emotion
    if emotion == target:
        st.session_state.correct_streak += 1
    else:
        # Mauvaise émotion → ne décrémente pas (juste ne progresse pas)
        return WRONG

    # 4. VERIFIED
    if st.session_state.correct_streak >= STREAK_REQUIRED:
        st.session_state.rounds_passed += 1
        st.session_state.screen      = "result"
        st.session_state.result_kind = VERIFIED
        return VERIFIED

    return OK


def _handle_timeout() -> str:
    st.session_state.lives          -= 1
    st.session_state.correct_streak  = 0
    if st.session_state.lives <= 0:
        st.session_state.screen      = "result"
        st.session_state.result_kind = FAIL
        return FAIL
    else:
        st.session_state.screen      = "result"
        st.session_state.result_kind = TIMEOUT
        return TIMEOUT
