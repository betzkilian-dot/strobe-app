import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Physik Strobe Pro", layout="wide")

# --- Session State Initialisierung ---
if "clicks" not in st.session_state:
    st.session_state.clicks = []
if "current_frame_idx" not in st.session_state:
    st.session_state.current_frame_idx = 0
if "extracted_frames" not in st.session_state:
    st.session_state.extracted_frames = []
if "strobe_preview" not in st.session_state:
    st.session_state.strobe_preview = None
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

    if not frames:
        return None, 0, 0

    # Stroboskop-Bild generieren (Ebenen übereinanderlegen)
    acc = frames[0].astype(np.float32)
    for i in range(1, len(frames)):
        acc = cv2.addWeighted(acc, 1.0 - alpha, frames[i].astype(np.float32), alpha, 0)
    
    strobe_img = np.clip(acc, 0, 255).astype(np.uint8)
    return frames, strobe_img, actual_freq

# --- UI ---
st.title("📸 Physik Stroboskop-Analyse Pro")

st.sidebar.header("1. Video & Stroboskop")
uploaded_file = st.sidebar.file_uploader("Video hochladen", type=["mp4", "mov"])

if uploaded_file:
    # Video Parameter
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    cap_info = cv2.VideoCapture(tfile.name)
    max_f = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap_info.get(cv2.CAP_PROP_FPS)
    cap_info.release()

    # Regler
    num_imgs = st.sidebar.slider("Anzahl der Bilder", 2, min(max_f, 100), 10)
    
    # Frequenzanzeige berechnen
    calc_step = max(1, max_f // num_imgs)
    calc_freq = orig_fps / calc_step
    st.sidebar.write(f"**Frequenz: ({calc_freq:.2f} Hz)**")
    
    alpha_val = st.sidebar.slider("Transparenz (Strobe-Effekt)", 0.05, 1.0, 0.3)

    if st.sidebar.button("Stroboskop generieren & Analyse starten"):
        frames, strobe, freq = process_video_and_strobe(tfile.name, num_imgs, alpha_val)
        st.session_state.extracted_frames = frames
        st.session_state.strobe_preview = strobe
        st.session_state.video_info = {"freq": freq}
        st.session_state.clicks = []
        st.session_state.current_frame_idx = 0
    os.unlink(tfile.name)

# --- Hauptbereich ---
if st.session_state.strobe_preview is not None:
    
    # 1. Stroboskopbild anzeigen
    st.subheader("Vorschau: Generiertes Stroboskopbild")
    st.image(st.session_state.strobe_preview, use_column_width=True)
    
    st.divider()
    
    # 2. Interaktive Analyse
    st.subheader("Interaktive Messung")
    st.sidebar.header("2. Messung & Zoom")
    zoom = st.sidebar.slider("Zoom Faktor", 1.0, 5.0, 1.5)
    real_dist = st.sidebar.number_input("Referenzstrecke in Meter", value=1.0)
    
    # Anleitung
    if len(st.session_state.clicks) < 2:
        st.info(f"Setze 2 Punkte für die Referenz (Klick {len(st.session_state.clicks)+1}/2)")
    else:
        curr_idx = st.session_state.current_frame_idx
        st.success(f"Objekt-Tracking: Klicke auf den Ball in Bild {curr_idx + 1}/{len(st.session_state.extracted_frames)}")

    # Aktuelles Bild für Tracking
    idx = st.session_state.current_frame_idx
    frame_to_show = Image.fromarray(st.session_state.extracted_frames[idx])
    w, h = frame_to_show.size
    
    # Zoom anwenden
    display_img = frame_to_show.resize((int(w * zoom), int(h * zoom)))
    
    # Klick-Interface
    value = streamlit_image_coordinates(display_img, key="tracking_canvas")

    if value:
        rx, ry = value["x"] / zoom, value["y"] / zoom
        new_pt = (rx, ry)
        
        # Punkt hinzufügen
        if not st.session_state.clicks or (abs(st.session_state.clicks[-1][0] - rx) > 0.5):
            st.session_state.clicks.append(new_pt)
            # Springe zum nächsten Bild, wenn Referenz fertig ist
            if len(st.session_state.clicks) > 2:
                if st.session_state.current_frame_idx < len(st.session_state.extracted_frames) - 1:
                    st.session_state.current_frame_idx += 1
            st.rerun()

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Letztes Bild"):
            st.session_state.current_frame_idx = max(0, idx - 1)
            st.rerun()
    with col2:
        if st.button("➡️ Nächstes Bild"):
            st.session_state.current_frame_idx = min(len(st.session_state.extracted_frames)-1, idx + 1)
            st.rerun()
    with col3:
        if st.button("🗑️ Punkt löschen"):
            if st.session_state.clicks:
                st.session_state.clicks.pop()
                if len(st.session_state.clicks) >= 2:
                    st.session_state.current_frame_idx = max(0, st.session_state.current_frame_idx - 1)
                st.rerun()

    # --- Auswertung ---
    if st.button("📊 Geschwindigkeiten berechnen", type="primary"):
        pts = st.session_state.clicks
        if len(pts) < 4:
            st.error("Nicht genug Punkte. Du brauchst 2 für die Referenz und mindestens 2 für die Bewegung.")
        else:
            # Maßstab
            p1, p2 = pts[0], pts[1]
            px_dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            m_per_px = real_dist / px_dist
            f = st.session_state.video_info["freq"]
            
            # Tabelle
            results = []
            tracking_pts = pts[2:]
            for i in range(len(tracking_pts)-1):
                d_px = np.sqrt((tracking_pts[i][0]-tracking_pts[i+1][0])**2 + (tracking_pts[i][1]-tracking_pts[i+1][1])**2)
                d_m = d_px * m_per_px
                v = d_m * f
                results.append({
                    "Intervall": f"Bild {i+1} ➔ {i+2}",
                    "Distanz (m)": round(d_m, 3),
                    "v (m/s)": round(v, 2),
                    "v (km/h)": round(v * 3.6, 2)
                })
            st.table(pd.DataFrame(results))
else:
    st.info("Bitte Video hochladen und links auf den Button klicken.")
