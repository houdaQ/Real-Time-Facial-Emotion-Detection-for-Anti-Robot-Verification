# ============================================================
# app.py — Anti-Robot Verification | Vidéo temps réel
# Fix v3 : base64 inline image → plus de MediaFileHandler errors
# ============================================================

import cv2
import time
import base64
import numpy as np
import streamlit as st

from cameraa import load_model, predict_from_image, _state, INFERENCE_EVERY_N
from utils   import EMOTION_COLORS, EMOTION_TIPS
from logic   import (
    init_session, start_game, start_round_timer, next_round, restart,
    process_capture, time_remaining, time_progress, is_expired,
    MAX_LIVES, TIME_LIMIT, STREAK_REQUIRED,
    VERIFIED, TIMEOUT, FAIL, CHEAT_MESSAGES,
)

# ── Haar pour overlay rectangle ──────────────────────────────
_HAAR     = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_det = cv2.CascadeClassifier(_HAAR)

st.set_page_config(
    page_title="Human Verification",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
[data-testid="stAppViewContainer"] { background: #f8f9fb; }
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stSidebar"] { display: none !important; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; color: #1a1d23; }
.stButton > button {
    all: unset; display: block; width: 100%; text-align: center;
    cursor: pointer; font-family: 'Inter', sans-serif;
    font-size: 13px; font-weight: 500; letter-spacing: 0.5px;
    padding: 12px 24px; border-radius: 6px; transition: all 0.15s ease;
    background: #1a1d23; color: #ffffff; border: 1px solid #1a1d23;
}
.stButton > button:hover { background: #2d3139; border-color: #2d3139; }
.stProgress > div > div { background: #e5e7eb; border-radius: 4px; height: 4px !important; }
.stProgress > div > div > div { border-radius: 4px; background: #1a1d23 !important; }
</style>
""", unsafe_allow_html=True)

def r(html):    st.markdown(html, unsafe_allow_html=True)
def sep():      r('<div style="height:1px;background:#e5e7eb;margin:20px 0"></div>')
def gap(px=16): r(f'<div style="height:{px}px"></div>')


# ════════════════════════════════════════════════════════════
# AFFICHAGE IMAGE EN BASE64  (fix MediaFileHandler)
# ════════════════════════════════════════════════════════════

def show_frame_b64(frame_bgr: np.ndarray, width_pct: int = 100) -> None:
    """
    Encode la frame en JPEG base64 et l'injecte en HTML inline.
    Zéro fichier temporaire → plus de MediaFileHandler: Missing file.
    """
    # Réduire la résolution pour alléger le transfert (~640px)
    h, w = frame_bgr.shape[:2]
    if w > 640:
        scale      = 640 / w
        frame_bgr  = cv2.resize(frame_bgr,
                                (640, int(h * scale)),
                                interpolation=cv2.INTER_AREA)

    _, buffer = cv2.imencode(
        ".jpg", frame_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 80]   # qualité 80 → bon compromis taille/qualité
    )
    b64 = base64.b64encode(buffer).decode("utf-8")
    r(f'<img src="data:image/jpeg;base64,{b64}" '
      f'style="width:{width_pct}%;border-radius:8px;display:block">')


# ════════════════════════════════════════════════════════════
# OVERLAY SUR LA FRAME
# ════════════════════════════════════════════════════════════

def draw_overlay(frame_bgr: np.ndarray, emotion: str,
                 confidence: float, face_found: bool,
                 target: str) -> np.ndarray:
    out    = frame_bgr.copy()
    h_im, w_im = out.shape[:2]

    # Rectangle visage (Haar sur image réduite)
    h, w   = frame_bgr.shape[:2]
    scale  = min(1.0, 320 / w)
    small  = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces  = _face_det.detectMultiScale(
        gray_s, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        x  = int(x / scale); y  = int(y / scale)
        fw = int(fw / scale); fh = int(fh / scale)

        rc = (46, 200, 100) if (face_found and emotion == target) else (80, 80, 220)
        cv2.rectangle(out, (x, y), (x + fw, y + fh), rc, 2)

        if face_found and emotion:
            lbl = f"{emotion}  {int(confidence * 100)}%"
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            pad = 4
            ly  = y - lh - pad * 2 if y - lh - pad * 2 > 0 else y + fh + lh + pad
            cv2.rectangle(out, (x, ly - pad), (x + lw + pad * 2, ly + lh + pad), rc, -1)
            cv2.putText(out, lbl, (x + pad, ly + lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Bande noire en bas
    cv2.rectangle(out, (0, h_im - 40), (w_im, h_im), (18, 18, 22), -1)
    if not face_found:
        msg, col = "Aucun visage détecté", (80, 100, 230)
    elif emotion:
        match    = (emotion == target)
        msg      = f"{'✓ ' if match else ''}{emotion}  {int(confidence * 100)}%"
        col      = (46, 200, 100) if match else (80, 80, 220)
    else:
        msg, col = "Analyse...", (160, 160, 160)

    cv2.putText(out, msg, (12, h_im - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
    return out


# ════════════════════════════════════════════════════════════
# SINGLETONS
# ════════════════════════════════════════════════════════════

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          15)   # limite webcam à 15fps
    return cap


# ════════════════════════════════════════════════════════════
# ÉCRAN 1 — WELCOME
# ════════════════════════════════════════════════════════════

def page_welcome():
    _, col, _ = st.columns([1, 3, 1])
    with col:
        gap(40)
        r("""
<div style="margin-bottom:48px">
  <div style="font-size:11px;font-weight:600;color:#9ca3af;
              letter-spacing:3px;text-transform:uppercase;margin-bottom:4px">Secure Access</div>
  <div style="width:32px;height:2px;background:#1a1d23"></div>
</div>""")
        r(f"""
<div style="margin-bottom:40px">
  <h1 style="font-size:28px;font-weight:600;color:#1a1d23;line-height:1.3;margin-bottom:12px">
    Vérification d'identité
  </h1>
  <p style="font-size:14px;color:#6b7280;line-height:1.8;font-weight:400">
    Imitez l'expression demandée devant votre caméra<br>
    pour prouver que vous êtes humain.<br><br>
    <span style="color:#1a1d23;font-weight:500">
      1 défi · 3 tentatives · {TIME_LIMIT}s · Détection en temps réel
    </span>
  </p>
</div>""")
        if st.button("Êtes-vous humain ?", use_container_width=True):
            start_game()
            st.rerun()
        gap(24); sep(); gap(8)
        r("""
<p style="font-size:11px;color:#9ca3af;text-align:center;line-height:1.7">
  Ce système analyse votre expression en temps réel.<br>
  Aucune donnée n'est enregistrée.
</p>""")


# ════════════════════════════════════════════════════════════
# ÉCRAN 2 — CHALLENGE
# ════════════════════════════════════════════════════════════

def page_challenge():
    target        = st.session_state.target_emotion
    camera_active = st.session_state.get("camera_active", False)
    rem           = time_remaining()
    prog          = time_progress()
    streak        = st.session_state.correct_streak
    lives         = st.session_state.lives
    color         = EMOTION_COLORS.get(target, "#1a1d23")
    tip           = EMOTION_TIPS.get(target, "")
    model         = get_model()

    # ── Navigation ───────────────────────────────────────────
    r("""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:14px 0;margin-bottom:4px">
  <div style="font-size:11px;font-weight:600;color:#9ca3af;letter-spacing:3px">SECURE ACCESS</div>
  <div style="font-size:11px;color:#9ca3af">Vérification en cours</div>
</div>""")
    sep()

    # ── Indicateurs ──────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        r(f"""
<div style="text-align:center;padding:8px 0">
  <div style="font-size:10px;font-weight:500;color:#9ca3af;letter-spacing:2px;margin-bottom:6px">TENTATIVES</div>
  <div style="font-size:22px;font-weight:600;color:#1a1d23">
    {lives}<span style="font-size:14px;color:#9ca3af"> / {MAX_LIVES}</span></div>
</div>""")
    with c2:
        if not camera_active:
            tc, label = "#9ca3af", str(TIME_LIMIT)
        else:
            tc    = "#dc2626" if rem < 5 else "#1a1d23"
            label = f"{int(rem):02d}"
        r(f"""
<div style="text-align:center;padding:8px 0">
  <div style="font-size:10px;font-weight:500;color:#9ca3af;letter-spacing:2px;margin-bottom:6px">TEMPS</div>
  <div style="font-size:22px;font-weight:600;color:{tc}">
    {label}<span style="font-size:14px;color:#9ca3af">s</span></div>
</div>""")
        st.progress(prog)
    with c3:
        r(f"""
<div style="text-align:center;padding:8px 0">
  <div style="font-size:10px;font-weight:500;color:#9ca3af;letter-spacing:2px;margin-bottom:6px">DÉTECTIONS</div>
  <div style="font-size:22px;font-weight:600;color:#1a1d23">
    {streak}<span style="font-size:14px;color:#9ca3af"> / {STREAK_REQUIRED}</span></div>
</div>""")
    sep()

    col_vid, col_info = st.columns([3, 2], gap="large")

    with col_vid:
        # Carte émotion cible
        r(f"""
<div style="background:#ffffff;border:1px solid #e5e7eb;
            border-left:3px solid {color};border-radius:8px;
            padding:14px 18px;margin-bottom:14px">
  <div style="font-size:10px;font-weight:600;color:#9ca3af;
              letter-spacing:2px;margin-bottom:6px">EXPRESSION DEMANDÉE</div>
  <div style="font-size:22px;font-weight:700;color:#1a1d23;margin-bottom:4px">{target}</div>
  <div style="font-size:12px;color:#6b7280">{tip}</div>
</div>""")

        # ── ÉTAT A : caméra fermée ────────────────────────────
        if not camera_active:
            r(f"""
<div style="background:#f9fafb;border:2px dashed #d1d5db;border-radius:8px;
            padding:48px 20px;text-align:center;margin-bottom:14px">
  <div style="font-size:36px;margin-bottom:12px">📷</div>
  <div style="font-size:13px;color:#6b7280;margin-bottom:4px">Préparez-vous à imiter</div>
  <div style="font-size:18px;font-weight:700;color:#1a1d23;margin-bottom:8px">{target}</div>
  <div style="font-size:12px;color:#9ca3af">
    Le chronomètre de {TIME_LIMIT}s démarre dès l'ouverture de la caméra
  </div>
</div>""")
            if st.button("📷  Ouvrir la caméra et démarrer", use_container_width=True):
                _state.reset()
                start_round_timer()
                st.session_state.last_cnn_frame = 0
                st.rerun()

        # ── ÉTAT B : caméra active ────────────────────────────
        else:
            try:
                cap = get_webcam()
                ret, frame_bgr = cap.read()

                if not ret or frame_bgr is None:
                    st.error("❌ Impossible de lire la caméra.")
                else:
                    # Timeout avant toute analyse
                    if is_expired():
                        _state.reset()
                        process_capture(None, 0.0, False)
                        st.rerun()

                    # Prédiction
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    result    = predict_from_image(model, frame_rgb, target_hint=target)

                    # Overlay + affichage BASE64 (zéro fichier temp)
                    annotated = draw_overlay(
                        frame_bgr,
                        emotion    = result.emotion,
                        confidence = result.confidence,
                        face_found = result.face_found,
                        target     = target,
                    )
                    show_frame_b64(annotated)          # ← base64 inline

                    # Mise à jour session
                    st.session_state.last_emotion    = result.emotion
                    st.session_state.last_confidence = result.confidence
                    st.session_state.last_probas     = result.probas
                    st.session_state.face_found      = result.face_found

                    # process_capture uniquement sur nouveau résultat CNN
                    cur = _state.frame_counter
                    old = st.session_state.get("last_cnn_frame", 0)
                    if cur != old and cur % INFERENCE_EVERY_N == 0:
                        st.session_state.last_cnn_frame = cur
                        gs = process_capture(
                            result.emotion,
                            result.confidence,
                            result.face_found,
                        )
                        if gs in (VERIFIED, TIMEOUT, FAIL):
                            _state.reset()
                            st.rerun()

            except RuntimeError as e:
                st.error(f"❌ Caméra : {e}")

            gap(8)
            if st.button("Annuler la vérification", use_container_width=True):
                _state.reset()
                restart()
                st.rerun()

    # ── Panneau info ─────────────────────────────────────────
    with col_info:
        last_emo  = st.session_state.last_emotion
        last_conf = st.session_state.last_confidence
        last_prob = st.session_state.last_probas

        if last_emo is None:
            msg = "Préparez votre expression…" if not camera_active \
                  else "Placez votre visage face à la caméra…"
            r(f"""
<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;
            padding:18px 20px;margin-bottom:12px">
  <div style="font-size:10px;font-weight:600;color:#9ca3af;letter-spacing:2px;margin-bottom:10px">ANALYSE</div>
  <div style="font-size:12px;color:#d1d5db">{msg}</div>
</div>""")
        else:
            match  = (last_emo == target)
            sc     = "#16a34a" if match else "#dc2626"
            status = "✓ Correct" if match else "✗ Incorrect"
            cp     = int(last_conf * 100)
            bc     = "#f0fdf4" if match else "#fef2f2"
            bbc    = "#bbf7d0" if match else "#fecaca"
            r(f"""
<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;
            padding:18px 20px;margin-bottom:12px">
  <div style="font-size:10px;font-weight:600;color:#9ca3af;letter-spacing:2px;margin-bottom:12px">ANALYSE</div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <div>
      <div style="font-size:16px;font-weight:600;color:#1a1d23">{last_emo}</div>
      <div style="font-size:11px;color:#9ca3af;margin-top:2px">Confiance : {cp}%</div>
    </div>
    <div style="background:{bc};border:1px solid {bbc};border-radius:4px;
                padding:4px 10px;font-size:11px;font-weight:500;color:{sc}">{status}</div>
  </div>
</div>""")
            if last_prob:
                r('<div style="font-size:10px;font-weight:600;color:#9ca3af;'
                  'letter-spacing:2px;margin-bottom:8px">PROBABILITÉS</div>')
                for emo, prob in sorted(last_prob.items(), key=lambda x: x[1], reverse=True)[:4]:
                    pct  = int(prob * 100)
                    bold = "font-weight:600;" if emo == target else "font-weight:400;"
                    tc2  = "#1a1d23" if emo == target else "#6b7280"
                    bc2  = "#1a1d23" if emo == target else "#d1d5db"
                    r(f"""
<div style="margin-bottom:7px">
  <div style="display:flex;justify-content:space-between;font-size:11px;{bold}margin-bottom:3px">
    <span style="color:{tc2}">{emo}</span><span style="color:{tc2}">{pct}%</span>
  </div>
  <div style="background:#f3f4f6;border-radius:3px;height:4px">
    <div style="width:{max(2,pct)}%;background:{bc2};height:4px;border-radius:3px"></div>
  </div>
</div>""")

        # Progression
        r(f"""
<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;
            padding:18px 20px;margin-bottom:12px">
  <div style="font-size:10px;font-weight:600;color:#9ca3af;letter-spacing:2px;margin-bottom:12px">PROGRESSION</div>
  <div style="display:flex;gap:6px">""")
        for i in range(STREAK_REQUIRED):
            bg  = "#1a1d23" if i < streak else "#f3f4f6"
            bc3 = "#1a1d23" if i < streak else "#e5e7eb"
            r(f'<div style="flex:1;height:6px;background:{bg};'
              f'border-radius:3px;border:1px solid {bc3}"></div>')
        r(f"""</div>
  <div style="font-size:11px;color:#9ca3af;margin-top:8px">
    {streak} / {STREAK_REQUIRED} détection(s) correcte(s)</div>
</div>""")

        for flag in st.session_state.get("cheat_flags", []):
            msg = CHEAT_MESSAGES.get(flag, flag)
            r(f'<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:6px;'
              f'padding:10px 14px;font-size:12px;color:#92400e;margin-bottom:6px">{msg}</div>')
        if camera_active and 0 < rem < 6:
            r(f'<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;'
              f'padding:10px 14px;font-size:12px;color:#991b1b;margin-bottom:6px">'
              f'⚠️ Plus que {int(rem)}s !</div>')

    # ── Rerun automatique si caméra active ───────────────────
    if camera_active:
        st.rerun()


# ════════════════════════════════════════════════════════════
# ÉCRAN 3 — RÉSULTAT
# ════════════════════════════════════════════════════════════

def page_result():
    kind  = st.session_state.result_kind
    lives = st.session_state.lives

    _, col, _ = st.columns([1, 3, 1])
    with col:
        gap(40)
        r("""<div style="font-size:11px;font-weight:600;color:#9ca3af;
                         letter-spacing:3px;margin-bottom:40px">SECURE ACCESS</div>""")

        configs = {
            VERIFIED: ("#16a34a", "#f0fdf4", "#bbf7d0", "ACCÈS ACCORDÉ",
                       "Vérification réussie ✓",
                       "Votre identité humaine a été confirmée.", None),
            TIMEOUT:  ("#d97706", "#fffbeb", "#fde68a", "INFORMATION",
                       "Temps écoulé",
                       f"Il vous reste {lives} tentative(s). Réessayez.", "Réessayer"),
            FAIL:     ("#dc2626", "#fef2f2", "#fecaca", "ACCÈS REFUSÉ",
                       "Vérification échouée",
                       "Toutes vos tentatives ont été utilisées.", None),
        }
        tc, bg, brd, label_top, title, msg, btn_next = configs.get(kind, configs[FAIL])

        r(f"""
<div style="background:{bg};border:1px solid {brd};border-radius:10px;
            padding:32px 28px;margin-bottom:24px">
  <div style="font-size:10px;font-weight:600;color:{tc};letter-spacing:2px;margin-bottom:12px">
    {label_top}</div>
  <div style="font-size:20px;font-weight:600;color:#1a1d23;margin-bottom:10px">{title}</div>
  <div style="font-size:13px;color:#4b5563;line-height:1.6">{msg}</div>
</div>""")

        r(f"""
<div style="display:flex;gap:24px;margin-bottom:24px">
  <div>
    <div style="font-size:10px;color:#9ca3af;margin-bottom:2px">Tentatives restantes</div>
    <div style="font-size:15px;font-weight:600;color:#1a1d23">{lives} / {MAX_LIVES}</div>
  </div>
</div>""")

        if btn_next:
            if st.button(btn_next, use_container_width=True):
                _state.reset()
                next_round()
                st.rerun()
            gap(8)

        if st.button("Recommencer la vérification", use_container_width=True):
            _state.reset()
            restart()
            st.rerun()


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    init_session()
    screen = st.session_state.screen
    if   screen == "welcome":   page_welcome()
    elif screen == "challenge": page_challenge()
    elif screen == "result":    page_result()

if __name__ == "__main__":
    main()
