# =============================================================
# app.py — Anti-Robot Emotion CAPTCHA
# =============================================================
#
# FLOW :
#   [1] WELCOME  → écran "Prove you're human"
#   [2] READY    → compte à rebours
#   [3] CHALLENGE → détection émotion en temps réel
#   [4] PASS / FAIL / TIMEOUT → résultat
#   [5] VERIFIED  → accès accordé
#
# =============================================================

import cv2
import numpy as np
import time

from camera import CameraModule
from logic  import GameLogic, get_warning_text
from utils  import (get_random_emotion, get_emotion_color,
                    get_emotion_label, get_emotion_emoji, GameTimer)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

MODE_SIMULATION = True
MODEL_PATH      = "model/cnn_model.h5"
TIME_LIMIT      = 12
ROUNDS_TO_WIN   = 3
WIN_W, WIN_H    = 1280, 800
WINDOW          = "Anti-Robot Verification"

EMOTION_LIST = ["happy", "sad", "angry", "surprised", "neutral", "fear", "disgust"]

# ── Couleurs BGR ──────────────────────────────
BG        = (14, 13, 20)
BG2       = (22, 21, 30)
WHITE     = (235, 235, 240)
MUTED     = (100, 100, 120)
ACCENT    = (0, 210, 255)       # cyan principal
GREEN     = (0, 210, 100)
RED       = (70, 70, 220)
ORANGE    = (40, 140, 255)
BORDER    = (40, 40, 58)

FONT   = cv2.FONT_HERSHEY_SIMPLEX
FONT2  = cv2.FONT_HERSHEY_DUPLEX


# ──────────────────────────────────────────────
# MODÈLE CNN
# ──────────────────────────────────────────────

def load_model():
    if MODE_SIMULATION:
        return None
    try:
        from tensorflow.keras.models import load_model as km
        return km(MODEL_PATH)
    except Exception as e:
        print(f"[WARN] CNN non chargé ({e}) → simulation")
        return None


def predict(model, face_roi):
    import random
    if model is None or face_roi is None:
        return random.choice(EMOTION_LIST)
    preds = model.predict(face_roi, verbose=0)[0]
    return EMOTION_LIST[int(np.argmax(preds))]


# ──────────────────────────────────────────────
# PRIMITIVES DE DESSIN
# ──────────────────────────────────────────────

def fill(img, x1, y1, x2, y2, color, alpha=0.92):
    sub = img[y1:y2, x1:x2]
    bg  = np.full(sub.shape, color, dtype=np.uint8)
    cv2.addWeighted(bg, alpha, sub, 1 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub


def txt(img, text, x, y, scale=0.65, color=WHITE, thick=1, font=FONT):
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def txt_c(img, text, cx, y, scale=0.65, color=WHITE, thick=1, font=FONT):
    w = cv2.getTextSize(text, font, scale, thick)[0][0]
    cv2.putText(img, text, (cx - w // 2, y), font, scale, color, thick, cv2.LINE_AA)


def hline(img, y, x1=0, x2=None, color=BORDER):
    if x2 is None: x2 = img.shape[1]
    cv2.line(img, (x1, y), (x2, y), color, 1)


def vline(img, x, y1, y2, color=BORDER):
    cv2.line(img, (x, y1), (x, y2), color, 1)


def progress_bar(img, x, y, w, h, pct, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 42), -1)
    fw = int(w * min(1.0, max(0.0, pct)))
    if fw > 2:
        cv2.rectangle(img, (x, y), (x + fw, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDER, 1)


def arc_timer(img, cx, cy, r, pct, color):
    cv2.circle(img, (cx, cy), r, (35, 35, 48), 5)
    deg = int(360 * (1.0 - pct))
    if deg < 360:
        cv2.ellipse(img, (cx, cy), (r, r), -90, 0, 360 - deg, color, 5)


def face_box(img, rect, color):
    if rect is None:
        return
    x, y, w, h = rect
    L = 18
    for pts in [
        [(x, y), (x + L, y + 3)], [(x, y), (x + 3, y + L)],
        [(x + w - L, y), (x + w, y + 3)], [(x + w - 3, y), (x + w, y + L)],
        [(x, y + h - 3), (x + L, y + h)], [(x, y + h - L), (x + 3, y + h)],
        [(x + w - L, y + h - 3), (x + w, y + h)],
        [(x + w - 3, y + h - L), (x + w, y + h)],
    ]:
        cv2.rectangle(img, pts[0], pts[1], color, -1)


def shield_icon(img, cx, cy, size=28, color=ACCENT):
    """Icône bouclier minimaliste"""
    pts = np.array([
        [cx - size, cy - size + 4],
        [cx + size, cy - size + 4],
        [cx + size, cy + size // 3],
        [cx,        cy + size],
        [cx - size, cy + size // 3],
    ], np.int32)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    # checkmark ou lock intérieur
    cv2.line(img, (cx - 8, cy), (cx - 2, cy + 7), color, 2, cv2.LINE_AA)
    cv2.line(img, (cx - 2, cy + 7), (cx + 9, cy - 6), color, 2, cv2.LINE_AA)


def dot_steps(img, current, total, cx, y, color_done, color_curr):
    """Indicateur de progression par points"""
    spacing = 28
    start   = cx - (total - 1) * spacing // 2
    for i in range(total):
        c = cx - (total // 2 - i) * spacing
        if i < current:
            cv2.circle(img, (c, y), 6, color_done, -1)
        elif i == current:
            cv2.circle(img, (c, y), 7, color_curr, 2)
            cv2.circle(img, (c, y), 3, color_curr, -1)
        else:
            cv2.circle(img, (c, y), 5, BORDER, -1)
            cv2.circle(img, (c, y), 5, MUTED, 1)


# ──────────────────────────────────────────────
# ÉCRAN 1 — WELCOME (pas de caméra)
# ──────────────────────────────────────────────

def draw_welcome(img):
    h, w = img.shape[:2]
    fill(img, 0, 0, w, h, BG, alpha=1.0)

    # ── Panneau central ──────────────────────
    pw, ph = 440, 340
    px, py = (w - pw) // 2, (h - ph) // 2 - 10
    fill(img, px, py, px + pw, py + ph, BG2, alpha=1.0)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), BORDER, 1)

    # Bande accent haut
    cv2.rectangle(img, (px, py), (px + pw, py + 3), ACCENT, -1)

    # Icône bouclier
    shield_icon(img, px + pw // 2, py + 60, size=28, color=ACCENT)

    # Titre principal
    txt_c(img, "HUMAN VERIFICATION", px + pw // 2, py + 112,
          scale=0.75, color=WHITE, thick=2)

    # Sous-titre
    txt_c(img, "Anti-Robot Emotion CAPTCHA", px + pw // 2, py + 138,
          scale=0.5, color=MUTED)

    # Séparateur
    hline(img, py + 158, px + 24, px + pw - 24, BORDER)

    # Description
    lines = [
        "You will be asked to show  3  different",
        "facial emotions in front of your camera.",
        "",
        "Each round has a  12 second  time limit.",
        "You have  3 lives  in total.",
    ]
    for i, line in enumerate(lines):
        txt_c(img, line, px + pw // 2, py + 188 + i * 24,
              scale=0.5, color=(160, 160, 180) if line else MUTED)

    # Bouton START
    bx, by = px + 60, py + ph - 72
    bw, bh = pw - 120, 44
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), ACCENT, -1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), ACCENT, 1)
    txt_c(img, "PRESS  ENTER  TO START", bx + bw // 2, by + 28,
          scale=0.55, color=BG, thick=2)

    # Footer
    txt_c(img, "Press Q to quit", w // 2, h - 24, scale=0.4, color=(55, 55, 72))


# ──────────────────────────────────────────────
# ÉCRAN 2 — COMPTE À REBOURS
# ──────────────────────────────────────────────

def draw_ready(frame, n, target, round_num):
    h, w = frame.shape[:2]
    fill(frame, 0, 0, w, h, BG, alpha=0.75)

    txt_c(frame, f"ROUND  {round_num}", w // 2, h // 2 - 100,
          scale=0.6, color=MUTED)

    emo_c = get_emotion_color(target)
    txt_c(frame, "Your emotion:", w // 2, h // 2 - 60,
          scale=0.5, color=MUTED)
    txt_c(frame, get_emotion_label(target), w // 2, h // 2 - 20,
          scale=1.4, color=emo_c, thick=3, font=FONT2)
    txt_c(frame, get_emotion_emoji(target), w // 2, h // 2 + 24,
          scale=0.75, color=MUTED)

    # Chiffre du countdown
    txt_c(frame, str(n), w // 2, h // 2 + 105,
          scale=4.0, color=WHITE, thick=5)

    txt_c(frame, "GET READY", w // 2, h // 2 + 155,
          scale=0.55, color=MUTED)


# ──────────────────────────────────────────────
# ÉCRAN 3 — CHALLENGE (boucle principale)
# ──────────────────────────────────────────────

def draw_challenge(frame, target, timer, logic, pred, warnings, status, face_rect):
    h, w = frame.shape[:2]
    emo_c = get_emotion_color(target)
    tc    = timer.urgency_color()
    rem   = max(0, int(timer.remaining()))

    # ── TOP BAR (56px) ───────────────────────
    fill(frame, 0, 0, w, 56, BG, alpha=0.94)
    hline(frame, 56)

    # Logo / titre
    txt(frame, "ANTI-ROBOT CHECK", 20, 38, scale=0.65, color=WHITE, thick=2)

    # Barre de rounds (dots)
    dot_steps(frame, logic.rounds_passed, ROUNDS_TO_WIN,
              w // 2, 28, GREEN, ACCENT)

    # Vies (droite)
    for i in range(logic.MAX_LIVES):
        col = RED if i < logic.lives else (40, 40, 55)
        cx  = w - 30 - (logic.MAX_LIVES - 1 - i) * 28
        cv2.circle(frame, (cx, 28), 8, col, -1)

    # ── FACE BOX ────────────────────────────
    box_color = {
        "PASS":  GREEN,
        "CHEAT": RED,
        "OK":    BORDER,
    }.get(status, BORDER)
    face_box(frame, face_rect, box_color)

    # ── SIDE PANEL GAUCHE (info émotion) ────
    PW = 210
    fill(frame, 0, 56, PW, h - 80, BG, alpha=0.90)
    vline(frame, PW, 56, h - 80)

    # Label
    txt(frame, "SHOW THIS", 18, 94, scale=0.42, color=MUTED)

    # Barre accent
    cv2.rectangle(frame, (18, 102), (22, 154), emo_c, -1)

    # Émotion (grand)
    txt(frame, get_emotion_label(target), 30, 140,
        scale=1.3, color=emo_c, thick=3, font=FONT2)

    # Emoji
    txt(frame, get_emotion_emoji(target), 30, 168,
        scale=0.7, color=(160, 160, 180))

    # Séparateur
    hline(frame, 185, 18, PW - 18)

    # Détection actuelle
    txt(frame, "DETECTED", 18, 210, scale=0.4, color=MUTED)
    if pred:
        match = pred == target
        pc = GREEN if match else MUTED
        txt(frame, get_emotion_label(pred), 18, 240,
            scale=0.85, color=pc, thick=2)
    else:
        txt(frame, "--", 18, 240, scale=0.7, color=MUTED)

    # Séparateur
    hline(frame, 260, 18, PW - 18)

    # Progress
    txt(frame, "PROGRESS", 18, 285, scale=0.4, color=MUTED)
    progress_bar(frame, 18, 295, PW - 36, 10, logic.progress, emo_c)
    pct = int(logic.progress * 100)
    txt(frame, f"{pct}%", 18, 320, scale=0.45, color=emo_c if pct > 50 else MUTED)

    # ── SIDE PANEL DROIT (timer) ─────────────
    fill(frame, w - PW, 56, w, h - 80, BG, alpha=0.90)
    vline(frame, w - PW, 56, h - 80)

    txt_c(frame, "TIME", w - PW + PW // 2, 90, scale=0.42, color=MUTED)

    cx, cy = w - PW + PW // 2, 155
    arc_timer(frame, cx, cy, 44, 1.0 - timer.progress(), tc)
    txt_c(frame, str(rem), cx, cy + 10, scale=1.4, color=tc, thick=2)
    txt_c(frame, "sec",     cx, cy + 34, scale=0.42, color=MUTED)

    # Conseil de l'émotion
    hline(frame, 220, w - PW + 18, w - 18)
    txt_c(frame, "TIP", cx, 242, scale=0.4, color=MUTED)
    tips = {
        "happy":     ["Smile wide,", "teeth showing"],
        "sad":       ["Frown, look", "down slightly"],
        "angry":     ["Furrowed brow,", "tight jaw"],
        "surprised": ["Wide eyes,", "open mouth"],
        "neutral":   ["Relaxed face,", "no expression"],
        "fear":      ["Wide eyes,", "tense"],
        "disgust":   ["Wrinkled nose,", "slight squint"],
    }
    tip_lines = tips.get(target, ["Show the", "emotion clearly"])
    txt_c(frame, tip_lines[0], cx, 264, scale=0.45, color=(160, 160, 180))
    txt_c(frame, tip_lines[1], cx, 284, scale=0.45, color=(160, 160, 180))

    # ── ALERTE ANTI-TRICHE ───────────────────
    if warnings:
        msg = get_warning_text(warnings[0])
        mw  = cv2.getTextSize(msg, FONT, 0.6, 2)[0][0]
        bx  = w // 2 - mw // 2 - 24
        by  = h // 2 - 24
        fill(frame, bx, by, bx + mw + 48, by + 46, (10, 16, 48), alpha=0.95)
        cv2.rectangle(frame, (bx, by), (bx + mw + 48, by + 46), ORANGE, 1)
        txt_c(frame, msg, w // 2, by + 30, scale=0.6, color=(160, 190, 255), thick=2)

    # ── BOTTOM BAR (80px) ────────────────────
    fill(frame, 0, h - 80, w, h, BG, alpha=0.94)
    hline(frame, h - 80)

    txt_c(frame, "[Q] quit   [R] restart", w // 2, h - 22,
          scale=0.42, color=(55, 55, 72))

    # Mode simulation badge
    if MODE_SIMULATION:
        txt(frame, "[ SIMULATION ]", w - 160, h - 24,
            scale=0.38, color=(55, 70, 120))


# ──────────────────────────────────────────────
# ÉCRANS DE RÉSULTAT
# ──────────────────────────────────────────────

def draw_result(frame, kind, rounds_done):
    """kind : 'verified' | 'timeout' | 'fail' | 'pass' """
    h, w = frame.shape[:2]
    fill(frame, 0, 0, w, h, BG, alpha=0.85)

    pw, ph = 460, 280
    px, py = (w - pw) // 2, (h - ph) // 2

    configs = {
        "verified": {
            "border": GREEN,
            "title":  "ACCESS GRANTED",
            "sub1":   f"All {rounds_done} challenges completed successfully.",
            "sub2":   "Identity verified — you are human.",
            "footer": "[R] restart   [Q] quit",
        },
        "pass": {
            "border": ACCENT,
            "title":  f"ROUND {rounds_done} PASSED",
            "sub1":   f"{ROUNDS_TO_WIN - rounds_done} round(s) remaining.",
            "sub2":   "Press SPACE to continue.",
            "footer": "[SPACE] next round   [Q] quit",
        },
        "timeout": {
            "border": ORANGE,
            "title":  "TIME OUT",
            "sub1":   "You ran out of time.",
            "sub2":   "Press SPACE to try again.",
            "footer": "[SPACE] retry   [R] restart   [Q] quit",
        },
        "fail": {
            "border": RED,
            "title":  "ACCESS DENIED",
            "sub1":   "Too many failed attempts.",
            "sub2":   "Robot detected.",
            "footer": "[R] restart   [Q] quit",
        },
    }

    cfg = configs.get(kind, configs["fail"])
    col = cfg["border"]

    fill(frame, px, py, px + pw, py + ph, BG2, alpha=0.97)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), col, 1)
    cv2.rectangle(frame, (px, py), (px + pw, py + 4), col, -1)

    # Icône selon résultat
    icon_cx = px + pw // 2
    if kind == "verified":
        shield_icon(frame, icon_cx, py + 55, size=24, color=col)
    else:
        cv2.circle(frame, (icon_cx, py + 55), 22, col, 2)
        sign = "!" if kind in ("fail", "timeout") else ">"
        txt_c(frame, sign, icon_cx, py + 64, scale=0.9, color=col, thick=2)

    txt_c(frame, cfg["title"], icon_cx, py + 110,
          scale=1.0, color=col, thick=2, font=FONT2)

    hline(frame, py + 128, px + 30, px + pw - 30, col)

    txt_c(frame, cfg["sub1"], icon_cx, py + 160, scale=0.55, color=WHITE)
    txt_c(frame, cfg["sub2"], icon_cx, py + 188, scale=0.55, color=MUTED)

    hline(frame, py + ph - 44, px + 30, px + pw - 30)
    txt_c(frame, cfg["footer"], icon_cx, py + ph - 20,
          scale=0.42, color=(65, 65, 85))


# ──────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ──────────────────────────────────────────────

def run():
    model = load_model()
    cam   = CameraModule()
    logic = GameLogic()
    timer = GameTimer(limit=TIME_LIMIT)

    # Démarre la caméra immédiatement (besoin du feed même en welcome)
    cam.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, WIN_W, WIN_H)

    # ── Machine d'états ───────────────────────
    # WELCOME → READY → CHALLENGE → (PASS | TIMEOUT | FAIL) → VERIFIED
    STATE      = "WELCOME"
    target     = get_random_emotion()
    pred       = None
    warnings   = []
    last_status = "OK"
    face_rect  = None
    cd_start   = 0.0
    CD_DUR     = 3

    print(f"\n▶  Anti-Robot CAPTCHA démarré")
    print(f"   Mode : {'simulation' if MODE_SIMULATION else 'CNN'}")

    while True:
        # Capture toujours (pour avoir la caméra en fond même sur welcome)
        try:
            frame_raw, faces, face_rect, face_roi = cam.get_frame()
        except RuntimeError:
            frame_raw = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
            faces, face_rect, face_roi = [], None, None

        frame = cv2.resize(frame_raw, (WIN_W, WIN_H))

        # ══════════════════════════════════════
        if STATE == "WELCOME":
            # Fond = caméra assombrie
            fill(frame, 0, 0, WIN_W, WIN_H, BG, alpha=0.7)
            draw_welcome(frame)

        elif STATE == "READY":
            n = max(1, CD_DUR - int(time.time() - cd_start))
            draw_ready(frame, n, target, logic.rounds_passed + 1)
            if time.time() - cd_start >= CD_DUR:
                timer.reset()
                timer.start()
                STATE = "CHALLENGE"

        elif STATE == "CHALLENGE":
            pred = predict(model, face_roi)
            last_status, warnings = logic.update(
                pred, target, frame, faces, face_rect)

            draw_challenge(frame, target, timer, logic,
                           pred, warnings, last_status, face_rect)

            if last_status == "PASS":
                if logic.rounds_passed >= ROUNDS_TO_WIN:
                    STATE = "VERIFIED"
                else:
                    STATE = "PASS"

            elif timer.is_expired():
                alive = logic.on_timeout()
                STATE = "TIMEOUT" if alive else "FAIL"

        elif STATE == "PASS":
            draw_challenge(frame, target, timer, logic, pred, [], "PASS", face_rect)
            draw_result(frame, "pass", logic.rounds_passed)

        elif STATE == "VERIFIED":
            draw_result(frame, "verified", logic.rounds_passed)

        elif STATE == "TIMEOUT":
            draw_result(frame, "timeout", logic.rounds_passed)

        elif STATE == "FAIL":
            draw_result(frame, "fail", logic.rounds_passed)

        # ── Affichage ─────────────────────────
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        # ── Clavier ───────────────────────────
        if key == ord('q') or key == 27:           # Q ou Echap
            break

        elif key == 13 and STATE == "WELCOME":     # Enter → démarre
            cd_start = time.time()
            STATE    = "READY"
            target   = get_random_emotion()
            print(f"   Démarrage — émotion : {target}")

        elif key == ord('r'):                       # Restart complet
            logic.full_reset()
            target   = get_random_emotion()
            pred     = None
            warnings = []
            STATE    = "WELCOME"
            print(f"   Restart")

        elif key == ord(' ') and STATE in ("PASS", "TIMEOUT"):   # Round suivant
            target   = get_random_emotion(exclude=target)
            logic.reset_for_new_round()
            cd_start = time.time()
            STATE    = "READY"
            print(f"   Round suivant — émotion : {target}")

    cam.release()
    cv2.destroyAllWindows()
    print(f"\n■  Session terminée | Rounds : {logic.rounds_passed} | Vies : {logic.lives}")


if __name__ == "__main__":
    run()
