import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import tempfile
import os
import pandas as pd
import io
import plotly.express as px
from streamlit_image_coordinates import streamlit_image_coordinates

# Versionsnummer
VERSION = "v0.009"

st.set_page_config(page_title="Physik Strobe Pro", layout="wide")

# --- Session State Initialisierung ---
if "clicks_ref" not in st.session_state:
    st.session_state.clicks_ref = []
if "ref_confirmed" not in st.session_state:
    st.session_state.ref_confirmed = False
if "clicks_track" not in st.session_state:
    st.session_state.clicks_track = []
if "current_frame_idx" not in st.session_state:
    st.session_state.current_frame_idx = 0
if "extracted_frames" not in st.session_state:
    st.session_state.extracted_frames = []
if "base_strobe" not in st.session_state:
    st.session_state.base_strobe = None
if "video_info" not in st.session_state:
    st.session_state.video_info = {}
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

def process_video_and_strobe(video_path, num_images, alpha):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, total_frames // num_images)
    actual_freq = fps / step
    frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or len(frames) >= num_images:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames: return None, None, 0
    acc = frames[0].astype(np.float32)
    for i in range(1, len(frames)):
        acc = cv2.addWeighted(acc, 1.0 - alpha, frames[i].astype(np.float32), alpha, 0)
    strobe_img = np.clip(acc, 0, 255).astype(np.uint8)
    return frames, strobe_img, actual_freq

def enhance_image(pil_img, brightness, contrast):
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(pil_img)
    return enhancer.enhance(contrast)

# --- UI Header ---
st.title("📸 Physik Stroboskop-Analyse Pro")

# --- Sidebar ---
st.sidebar.header("1. Video & Stroboskop")
uploaded_file = st.sidebar.file_uploader("Video laden", type=["mp4", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
    cap_info = cv2.VideoCapture(temp_path)
    max_f = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap_info.get(cv2.CAP_PROP_FPS)
    cap_info.release()
    num_imgs = st.sidebar.slider("Anzahl der Bilder", 2, min(max_f, 100), 10)
    calc_freq = orig_fps / max(1, max_f // num_imgs)
    st.sidebar.info(f"Frequenz: ({calc_freq:.2f} Hz)")
    alpha_val = st.sidebar.slider("Transparenz (Strobe)", 0.05, 1.0, 0.3)
    if st.sidebar.button("Video verarbeiten", use_container_width=True):
        frames, strobe, freq = process_video_and_strobe(temp_path, num_imgs, alpha_val)
        st.session_state.extracted_frames = frames
        st.session_state.base_strobe = strobe
        st.session_state.video_info = {"freq": freq}
        st.session_state.clicks_ref = []
        st.session_state.ref_confirmed = False
        st.session_state.clicks_track = []
        st.session_state.current_frame_idx = 0
    os.unlink(temp_path)

# --- Hauptbereich ---
if st.session_state.base_strobe is not None:
    st.sidebar.header("2. Anzeige & Optimierung")
    brightness = st.sidebar.slider("Helligkeit", 0.5, 3.0, 1.0)
    contrast = st.sidebar.slider("Kontrast", 0.5, 3.0, 1.0)
    canvas_width = st.sidebar.slider("Anzeigebreite (Pixel)", 400, 1200, 800)
    real_dist = st.sidebar.number_input("Referenzstrecke (m)", value=1.0)

    # --- BENUTZERFÜHRUNG & LOGIK ---
    st.subheader("💡 Aktueller Status")
    
    # Logik für die Anzeige des richtigen Frames
    if len(st.session_state.clicks_ref) < 2:
        st.info(f"📏 **Referenz markieren**: Setze Punkt {len(st.session_state.clicks_ref)+1}/2 für den Maßstab.")
        idx_to_show = 0
    elif not st.session_state.ref_confirmed:
        st.warning("⚠️ **Referenz prüfen**: Passt die gelbe Linie? Bestätige unten, um fortzufahren.")
        idx_to_show = 0
    elif st.session_state.edit_mode:
        st.warning(f"🛠️ **Korrektur-Modus**: Setze Punkt {st.session_state.edit_idx + 1} neu.")
        idx_to_show = st.session_state.edit_idx
    else:
        curr_pts = len(st.session_state.clicks_track)
        if curr_pts < len(st.session_state.extracted_frames):
            st.success(f"🎯 **Tracking**: Bildpunkt {curr_pts + 1} von {len(st.session_state.extracted_frames)}")
            idx_to_show = st.session_state.current_frame_idx
        else:
            st.success("✅ **Fertig**: Alle Punkte erfasst.")
            idx_to_show = len(st.session_state.extracted_frames) - 1

    # --- TRACKING INTERFACE ---
    raw_frame = st.session_state.extracted_frames[idx_to_show].copy()
    pil_img = enhance_image(Image.fromarray(raw_frame), brightness, contrast)
    draw = ImageDraw.Draw(pil_img)
    
    # Referenz zeichnen (Gelb & Dick)
    if len(st.session_state.clicks_ref) >= 1:
        for p in st.session_state.clicks_ref:
            draw.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6], fill="yellow", outline="black")
        if len(st.session_state.clicks_ref) == 2:
            draw.line(st.session_state.clicks_ref, fill="yellow", width=8) # Dicke gelbe Linie

    # Tracking Punkte mit Nummern zeichnen
    for i, p in enumerate(st.session_state.clicks_track):
        color = "cyan" if (st.session_state.edit_mode and i == st.session_state.edit_idx) else "red"
        draw.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=color)
        draw.text((p[0]+8, p[1]+8), str(i+1), fill=color)

    w_orig, _ = pil_img.size
    value = streamlit_image_coordinates(pil_img, width=canvas_width, key="main_canvas")

    if value:
        scale = w_orig / canvas_width
        rx, ry = value["x"] * scale, value["y"] * scale
        
        # 1. Referenz-Punkte setzen
        if len(st.session_state.clicks_ref) < 2:
            st.session_state.clicks_ref.append((rx, ry))
            st.rerun()
        # 2. Im Tracking-Modus (nur wenn Referenz bestätigt)
        elif st.session_state.ref_confirmed:
            if st.session_state.edit_mode:
                st.session_state.clicks_track[st.session_state.edit_idx] = (rx, ry)
                st.session_state.edit_mode = False
                st.rerun()
            else:
                if not st.session_state.clicks_track or (abs(st.session_state.clicks_track[-1][0] - rx) > 0.5):
                    st.session_state.clicks_track.append((rx, ry))
                    if st.session_state.current_frame_idx < len(st.session_state.extracted_frames) - 1:
                        st.session_state.current_frame_idx += 1
                    st.rerun()

    # --- SPEZIELLE BESTÄTIGUNGS-BUTTONS ---
    if len(st.session_state.clicks_ref) == 2 and not st.session_state.ref_confirmed:
        if st.button("✅ Referenzstrecke passt so - Tracking starten", type="primary", use_container_width=True):
            st.session_state.ref_confirmed = True
            st.rerun()
        if st.button("❌ Punkte falsch - Referenz neu setzen", use_container_width=True):
            st.session_state.clicks_ref = []
            st.rerun()

    # --- NORMALE CONTROLS ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⏪ Letzter Punkt weg / Frame zurück", use_container_width=True):
            st.session_state.current_frame_idx = max(0, st.session_state.current_frame_idx - 1)
            if st.session_state.clicks_track: st.session_state.clicks_track.pop()
            st.rerun()
    with c2:
        if len(st.session_state.clicks_track) > 0:
            edit_q = st.number_input("Punkt korrigieren (#)", 1, len(st.session_state.clicks_track), step=1)
            if st.button("🎯 Korrektur starten"):
                st.session_state.edit_mode = True
                st.session_state.edit_idx = edit_q - 1
                st.rerun()
    with c3:
        if st.button("📏 Referenz komplett neu", use_container_width=True):
            st.session_state.clicks_ref = []
            st.session_state.ref_confirmed = False
            st.rerun()

    # --- STROBE ANZEIGE ---
    with st.expander("🖼️ Stroboskop-Übersicht"):
        st_clean = enhance_image(Image.fromarray(st.session_state.base_strobe), brightness, contrast)
        st_eval = st_clean.copy()
        draw_st = ImageDraw.Draw(st_eval)
        if len(st.session_state.clicks_ref) == 2:
            draw_st.line(st.session_state.clicks_ref, fill="yellow", width=8)
        for i, p in enumerate(st.session_state.clicks_track):
            draw_st.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill="red", outline="white")
            draw_st.text((p[0]+8, p[1]+8), str(i+1), fill="red")
        st.image(st_eval, use_container_width=True)
        buf = io.BytesIO()
        st_eval.save(buf, format="PNG")
        st.sidebar.download_button("📥 Strobe-Bild speichern", buf.getvalue(), "strobe_analyse.png", "image/png", use_container_width=True)

    # --- DIAGRAMME ---
    if len(st.session_state.clicks_track) >= 2 and st.session_state.ref_confirmed:
        st.divider()
        st.subheader("📊 Diagramme")
        f = st.session_state.video_info["freq"]
        dt = 1 / f
        px_dist = np.sqrt((st.session_state.clicks_ref[0][0]-st.session_state.clicks_ref[1][0])**2 + (st.session_state.clicks_ref[0][1]-st.session_state.clicks_ref[1][1])**2)
        m_per_px = real_dist / px_dist
        times = [i * dt for i in range(len(st.session_state.clicks_track))]
        dist_cum = [0.0]
        v_list = []
        track = st.session_state.clicks_track
        for i in range(len(track)-1):
            d_m = np.sqrt((track[i][0]-track[i+1][0])**2 + (track[i][1]-track[i+1][1])**2) * m_per_px
            dist_cum.append(dist_cum[-1] + d_m)
            v_list.append(d_m / dt)
        
        t1, t2 = st.tabs(["t-s Diagramm", "t-v Diagramm"])
        with t1:
            st.plotly_chart(px.line(x=times, y=dist_cum, labels={'x':'t (s)', 'y':'s (m)'}, markers=True), use_container_width=True)
        with t2:
            st.plotly_chart(px.line(x=times[1:], y=v_list, labels={'x':'t (s)', 'y':'v (m/s)'}, markers=True), use_container_width=True)

st.divider()
st.caption(f"ByLKI Physik-Analyse | {VERSION} | Kilian Betz")
