import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os
import pandas as pd
import io
from streamlit_image_coordinates import streamlit_image_coordinates

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

    if not frames: return None, None, 0
    
    # Stroboskop-Berechnung
    acc = frames[0].astype(np.float32)
    for i in range(1, len(frames)):
        acc = cv2.addWeighted(acc, 1.0 - alpha, frames[i].astype(np.float32), alpha, 0)
    
    strobe_img = np.clip(acc, 0, 255).astype(np.uint8)
    return frames, strobe_img, actual_freq

# --- UI Header ---
st.title("📸 Physik Stroboskop-Analyse Pro")

# --- Sidebar ---
st.sidebar.header("1. Video & Stroboskop")
uploaded_file = st.sidebar.file_uploader("Video hochladen", type=["mp4", "mov"])

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

    if st.sidebar.button("Video verarbeiten & Strobe erstellen", use_container_width=True):
        frames, strobe, freq = process_video_and_strobe(temp_path, num_imgs, alpha_val)
        st.session_state.extracted_frames = frames
        st.session_state.strobe_preview = strobe
        st.session_state.video_info = {"freq": freq}
        st.session_state.clicks_ref = []
        st.session_state.clicks_track = []
        st.session_state.current_frame_idx = 0
    os.unlink(temp_path)

# --- Hauptbereich ---
if st.session_state.strobe_preview is not None:
    
    # DOWNLOADS IN SIDEBAR (Immer verfügbar wenn Strobe da ist)
    st.sidebar.header("2. Downloads")
    
    # Strobe Bild Download
    strobe_pil = Image.fromarray(st.session_state.strobe_preview)
    buf = io.BytesIO()
    strobe_pil.save(buf, format="PNG")
    st.sidebar.download_button("📥 Download Stroboskop Bild", buf.getvalue(), "stroboskop.png", "image/png", use_container_width=True)

    # CSV Download Logik
    if len(st.session_state.clicks_track) >= 2 and len(st.session_state.clicks_ref) == 2:
        ref = st.session_state.clicks_ref
        track = st.session_state.clicks_track
        real_d = st.session_state.get("last_real_dist", 1.0)
        px_d = np.sqrt((ref[0][0]-ref[1][0])**2 + (ref[0][1]-ref[1][1])**2)
        m_px = real_d / px_d
        f = st.session_state.video_info["freq"]
        csv_list = [{"Intervall": i+1, "v_ms": (np.sqrt((track[i][0]-track[i+1][0])**2 + (track[i][1]-track[i+1][1])**2) * m_px * f)} for i in range(len(track)-1)]
        st.sidebar.download_button("📥 Download Geschwindigkeiten (CSV)", pd.DataFrame(csv_list).to_csv(index=False).encode('utf-8'), "messdaten.csv", "text/csv", use_container_width=True)

    # --- ANZEIGE STROBOSKOPBILD (Immer oben) ---
    with st.expander("🖼️ Stroboskop-Referenzbild (Dauerhaft verfügbar)", expanded=True):
        st.image(st.session_state.strobe_preview, use_column_width=True)

    st.divider()

    # --- INTERAKTIVE ANALYSE ---
    st.subheader("🎯 Interaktive Punkt-Analyse")
    
    st.sidebar.header("3. Analyse-Tools")
    zoom = st.sidebar.slider("🔍 Zoom zum Klicken", 1.0, 5.0, 1.5)
    real_dist = st.sidebar.number_input("Referenzstrecke (m)", value=1.0)
    st.session_state.last_real_dist = real_dist
    
    if st.sidebar.button("Referenz neu setzen"):
        st.session_state.clicks_ref = []
        st.rerun()
    
    if st.sidebar.button("Tracking zurücksetzen"):
        st.session_state.clicks_track = []
        st.session_state.current_frame_idx = 0
        st.rerun()

    # Status & Bild-Logik
    if len(st.session_state.clicks_ref) < 2:
        st.info(f"📏 Schritt 1: Referenz markieren ({len(st.session_state.clicks_ref)+1}/2)")
        idx_to_show = 0
    else:
        st.success(f"🎯 Schritt 2: Tracking (Bild {st.session_state.current_frame_idx + 1}/{len(st.session_state.extracted_frames)})")
        idx_to_show = st.session_state.current_frame_idx

    # Bild mit Einzeichnungen vorbereiten
    curr_frame = st.session_state.extracted_frames[idx_to_show].copy()
    pil_img = Image.fromarray(curr_frame)
    draw = ImageDraw.Draw(pil_img)

    # Gelbe Referenzlinie dauerhaft zeichnen
    if len(st.session_state.clicks_ref) == 2:
        draw.line(st.session_state.clicks_ref, fill="yellow", width=5)
    
    # Rote Tracking-Punkte zeichnen
    for p in st.session_state.clicks_track:
        draw.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill="red")

    # Zoom-Ansicht für das Klicken
    w, h = pil_img.size
    display_img = pil_img.resize((int(w * zoom), int(h * zoom)))
    
    st.write("Tippe auf das Bild, um Punkte zu setzen. Das Bild springt automatisch weiter.")
    value = streamlit_image_coordinates(display_img, key="main_canvas")

    if value:
        rx, ry = value["x"] / zoom, value["y"] / zoom
        new_pt = (rx, ry)
        
        # Referenz setzen
        if len(st.session_state.clicks_ref) < 2:
            if not st.session_state.clicks_ref or (abs(st.session_state.clicks_ref[-1][0] - rx) > 1):
                st.session_state.clicks_ref.append(new_pt)
                st.rerun()
        # Tracking setzen & Auto-Jump
        else:
            if not st.session_state.clicks_track or (abs(st.session_state.clicks_track[-1][0] - rx) > 0.5):
                st.session_state.clicks_track.append(new_pt)
                if st.session_state.current_frame_idx < len(st.session_state.extracted_frames) - 1:
                    st.session_state.current_frame_idx += 1
                st.rerun()

    # Manuelle Steuerung
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⬅️ Vorheriges Bild"):
            st.session_state.current_frame_idx = max(0, st.session_state.current_frame_idx - 1)
            st.rerun()
    with c2:
        if st.button("Nächstes Bild ➡️"):
            st.session_state.current_frame_idx = min(len(st.session_state.extracted_frames)-1, st.session_state.current_frame_idx + 1)
            st.rerun()
    with c3:
        if st.button("🗑️ Letzten Tracking-Punkt löschen"):
            if st.session_state.clicks_track:
                st.session_state.clicks_track.pop()
                st.session_state.current_frame_idx = max(0, st.session_state.current_frame_idx - 1)
                st.rerun()

    # Tabelle anzeigen
    if len(st.session_state.clicks_track) >= 2:
        st.divider()
        st.subheader("Ergebnisse")
        # Berechnung (identisch zu oben für die Anzeige)
        px_dist = np.sqrt((st.session_state.clicks_ref[0][0]-st.session_state.clicks_ref[1][0])**2 + (st.session_state.clicks_ref[0][1]-st.session_state.clicks_ref[1][1])**2)
        m_per_px = real_dist / px_dist
        f = st.session_state.video_info["freq"]
        res = []
        for i in range(len(st.session_state.clicks_track)-1):
            p_a, p_b = st.session_state.clicks_track[i], st.session_state.clicks_track[i+1]
            d_m = np.sqrt((p_a[0]-p_b[0])**2 + (p_a[1]-p_b[1])**2) * m_per_px
            v = d_m * f
            res.append({"Intervall": i+1, "v (m/s)": round(v, 2), "v (km/h)": round(v*3.6, 2)})
        st.table(pd.DataFrame(res))

else:
    st.info("Willkommen! Lade ein Video hoch und klicke in der Sidebar auf 'Video verarbeiten'.")
