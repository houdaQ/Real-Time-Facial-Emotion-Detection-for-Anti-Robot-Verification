import streamlit as st
import cv2
import numpy as np
import time
import random

from utils import get_random_emotion, get_emotion_label
@st.cache_resource
def get_camera():
    return cv2.VideoCapture(0)

cap = get_camera()
# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Anti-Robot CAPTCHA", layout="centered")

TIME_LIMIT = 12
ROUNDS_TO_WIN = 3

EMOTIONS = ["happy", "sad", "angry", "surprised", "neutral"]

# -------------------------------
# SESSION STATE
# -------------------------------
if "state" not in st.session_state:
    st.session_state.state = "WELCOME"
    st.session_state.rounds = 0
    st.session_state.target = random.choice(EMOTIONS)
    st.session_state.start_time = 0

# -------------------------------
# STYLE CSS 🔥
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #0e0d14;
    color: white;
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
.box {
    background-color: #1a1925;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
}
.stButton>button {
    background-color: #00d2ff;
    color: black;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='title'>🤖 Human Verification</div>", unsafe_allow_html=True)

# ===============================
# STATE : WELCOME
# ===============================
if st.session_state.state == "WELCOME":

    st.markdown("""
    <div class='box'>
        <h3>Anti-Robot Emotion CAPTCHA</h3>
        <p>Show different facial emotions to pass the test.</p>
        <p>⏱ 12 seconds per round</p>
        <p>🎯 3 rounds to win</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Start"):
        st.session_state.state = "READY"
        st.rerun()

# ===============================
# STATE : READY (countdown)
# ===============================
elif st.session_state.state == "READY":

    st.markdown("## ⏳ Get Ready...")

    for i in range(3, 0, -1):
        st.markdown(f"# {i}")
        time.sleep(1)

    st.session_state.state = "CHALLENGE"
    st.session_state.start_time = time.time()
    st.rerun()

# ===============================
# STATE : CHALLENGE
# ===============================
elif st.session_state.state == "CHALLENGE":

    st.markdown(f"## 😃 Show: **{get_emotion_label(st.session_state.target)}**")

    FRAME_WINDOW = st.image([])



    ret, frame = cap.read()

    if not ret:
        st.error("❌ Camera not detected")
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 🔁 simulation (remplacer par ton modèle plus tard)
        pred = random.choice(EMOTIONS)

        elapsed = time.time() - st.session_state.start_time
        remaining = int(TIME_LIMIT - elapsed)

        st.write(f"⏱ Time: {remaining} sec")
        st.write(f"🤖 Detected: {pred}")

        # ✅ SUCCESS
        if pred == st.session_state.target:
            st.session_state.rounds += 1

            if st.session_state.rounds >= ROUNDS_TO_WIN:
                st.session_state.state = "SUCCESS"
            else:
                st.session_state.target = random.choice(EMOTIONS)
                st.session_state.state = "READY"

            
            st.rerun()

        # ❌ TIMEOUT
        elif elapsed > TIME_LIMIT:
            st.session_state.state = "FAIL"
            st.rerun()
            

    # 🔁 refresh automatique (IMPORTANT)
    time.sleep(0.05)
    st.rerun()

# ===============================
# SUCCESS
# ===============================
elif st.session_state.state == "SUCCESS":

    st.success("✅ Access Granted — You are Human!")

    if st.button("🔄 Restart"):
        st.session_state.state = "WELCOME"
        st.session_state.rounds = 0
        st.rerun()

# ===============================
# FAIL
# ===============================
elif st.session_state.state == "FAIL":

    st.error("❌ Failed — Try Again")

    if st.button("🔁 Retry"):
        st.session_state.state = "WELCOME"
        st.session_state.rounds = 0
        st.rerun()