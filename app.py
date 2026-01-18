import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import tempfile
import os
import pandas as pd
import io
import plotly.express as px
import plotly.io as pio
from streamlit_image_coordinates import streamlit_image_coordinates

# Versionsnummer
VERSION = "v0.002"

st.set_page_config(page_title="Physik Strobe Pro", layout="wide")

# --- Session State Initialisierung ---
if "clicks_ref" not in st.session_state:
    st.session_state.clicks_ref = []
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

    # --- STROBOSKOPBILD MIT AUSWERTUNG ---
    st.subheader("🖼️ Stroboskopbild mit Auswertung")
    
    # Basis-Strobe nehmen und optimieren
    strobe_pil = Image.fromarray(st.session_state.base_strobe)
    strobe_enhanced = enhance_image(strobe_pil, brightness, contrast)
    draw_strobe = ImageDraw.Draw(strobe_enhanced)
    
    # Referenz einzeichnen
    if len(st.session_state.clicks_ref) == 2:
        draw_strobe.line(st.session_state.clicks_ref, fill="yellow", width=5)
    
    # Tracking Pfad einzeichnen
    if len(st.session_state.clicks_track) >= 2:
        draw_strobe.line(st.session_state.clicks_track, fill="red", width=3)
        for p in st.session_state.clicks_track:
            draw_strobe.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill="red", outline="white")

    st.image(strobe_enhanced, use_container_width=True)
    
    # Download Button für das Strobe Bild
    buf_strobe = io.BytesIO()
    strobe_enhanced.save(buf_strobe, format="PNG")
    st.sidebar.download_button("📥 Download Strobe + Pfad", buf_strobe.getvalue(), "strobe_analyse.png", "image/png", use_container_width=True)

    st.divider()

    # --- INTERAKTIVES TRACKING ---
    st.subheader("🎯 Messpunkt-Erfassung")
    if len(st.session_state.clicks_ref) < 2:
        st.info("📏 Schritt 1: Referenz markieren")
        idx = 0
    else:
        idx = st.session_state.current_frame_idx
        st.success(f"Tracking: Bild {idx + 1}/{len(st.session_state.extracted_frames)}")

    raw_frame = st.session_state.extracted_frames[idx].copy()
    pil_img = enhance_image(Image.fromarray(raw_frame), brightness, contrast)
    draw = ImageDraw.Draw(pil_img)
    
    if len(st.session_state.clicks_ref) == 2:
        draw.line(st.session_state.clicks_ref, fill="yellow", width=5)
    for p in st.session_state.clicks_track:
        draw.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill="red")

    w_orig, _ = pil_img.size
    value = streamlit_image_coordinates(pil_img, width=canvas_width, key="main_canvas")

    if value:
        scale = w_orig / canvas_width
        rx, ry = value["x"] * scale, value["y"] * scale
        if len(st.session_state.clicks_ref) < 2:
            st.session_state.clicks_ref.append((rx, ry))
            st.rerun()
        else:
            if not st.session_state.clicks_track or (abs(st.session_state.clicks_track[-1][0] - rx) > 0.5):
                st.session_state.clicks_track.append((rx, ry))
                if st.session_state.current_frame_idx < len(st.session_state.extracted_frames) - 1:
                    st.session_state.current_frame_idx += 1
                st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Tracking zurücksetzen", use_container_width=True):
            st.session_state.clicks_track = []
            st.session_state.current_frame_idx = 0
            st.rerun()
    with c2:
        if st.button("🗑️ Referenz neu setzen", use_container_width=True):
            st.session_state.clicks_ref = []
            st.rerun()

    # --- PHYSIKALISCHE DIAGRAMME ---
    if len(st.session_state.clicks_track) >= 2:
        st.divider()
        st.subheader("📊 Diagramme & Analyse")
        
        f = st.session_state.video_info["freq"]
        dt = 1 / f
        px_dist = np.sqrt((st.session_state.clicks_ref[0][0]-st.session_state.clicks_ref[1][0])**2 + (st.session_state.clicks_ref[0][1]-st.session_state.clicks_ref[1][1])**2)
        m_per_px = real_dist / px_dist
        
        times = [i * dt for i in range(len(st.session_state.clicks_track))]
        dist_cum = [0.0]
        v_list = []
        a_list = []
        
        track = st.session_state.clicks_track
        for i in range(len(track)-1):
            d_m = np.sqrt((track[i][0]-track[i+1][0])**2 + (track[i][1]-track[i+1][1])**2) * m_per_px
            dist_cum.append(dist_cum[-1] + d_m)
            v_list.append(d_m / dt)
            
        for i in range(len(v_list)-1):
            a_list.append((v_list[i+1] - v_list[i]) / dt)

        tab1, tab2, tab3 = st.tabs(["Zeit-Weg (s-t)", "Zeit-Geschw. (v-t)", "Zeit-Beschl. (a-t)"])
        
        with tab1:
            fig_s = px.line(x=times, y=dist_cum, labels={'x':'Zeit (s)', 'y':'Weg (m)'}, title="s-t Diagramm", markers=True)
            st.plotly_chart(fig_s, use_container_width=True)
            st.download_button("📥 s-t Diagramm (HTML)", fig_s.to_html(), "st_diagramm.html", "text/html")

        with tab2:
            fig_v = px.line(x=times[1:], y=v_list, labels={'x':'Zeit (s)', 'y':'v (m/s)'}, title="v-t Diagramm", markers=True)
            st.plotly_chart(fig_v, use_container_width=True)
            st.download_button("📥 v-t Diagramm (HTML)", fig_v.to_html(), "vt_diagramm.html", "text/html")

        with tab3:
            if len(a_list) > 0:
                fig_a = px.line(x=times[2:], y=a_list, labels={'x':'Zeit (s)', 'y':'a (m/s²)'}, title="a-t Diagramm", markers=True)
                st.plotly_chart(fig_a, use_container_width=True)
                st.download_button("📥 a-t Diagramm (HTML)", fig_a.to_html(), "at_diagramm.html", "text/html")
            else:
                st.info("Beschleunigung benötigt mindestens 4 Messpunkte.")

        # CSV Download
        df_res = pd.DataFrame({"Zeit_s": times[1:], "v_ms": v_list})
        st.sidebar.download_button("📥 Download Messdaten (CSV)", df_res.to_csv(index=False).encode('utf-8'), "messung.csv", "text/csv", use_container_width=True)

# Footer
st.divider()
st.caption(f"ByLKI Physik-Analyse | Version: {VERSION} | Kilian Betz")
