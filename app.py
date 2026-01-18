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
    acc = frames[0].astype(np.float32)
    for i in range(1, len(frames)):
        acc = cv2.addWeighted(acc, 1.0 - alpha, frames[i].astype(np.float32), alpha, 0)
    
    return frames, np.clip(acc, 0, 255).astype(np.uint8), actual_freq

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

# --- Download Sektion in Sidebar ---
if st.session_state.strobe_preview is not None:
    st.sidebar.header("2. Downloads")
    
    # Stroboskop Bild Download
    strobe_pil = Image.fromarray(st.session_state.strobe_preview)
    buf = io.BytesIO()
    strobe_pil.save(buf, format="PNG")
    st.sidebar.download_button(
        label="📥 Download Stroboskop Bild",
        data=buf.getvalue(),
        file_name="stroboskop_aufnahme.png",
        mime="image/png",
        use_container_width=True
    )

    # CSV Download (nur wenn Daten vorhanden)
    if len(st.session_state.clicks_track) >= 2 and len(st.session_state.clicks_ref) == 2:
        ref = st.session_state.clicks_ref
        track = st.session_state.clicks_track
        real_dist_val = st.session_state.get("last_real_dist", 1.0)
        
        px_dist = np.sqrt((ref[0][0]-ref[1][0])**2 + (ref[0][1]-ref[1][1])**2)
        m_per_px = real_dist_val / px_dist
        f = st.session_state.video_info["freq"]
        
        csv_data = []
        for i in range(len(track)-1):
            d_px = np.sqrt((track[i][0]-track[i+1][0])**2 + (track[i][1]-track[i+1][1])**2)
            d_m = d_px * m_per_px
            v = d_m * f
            csv_data.append({"Intervall": i+1, "Weg_m": d_m, "v_ms": v, "v_kmh": v*3.6})
        
        df = pd.DataFrame(csv_data)
        st.sidebar.download_button(
            label="📥 Download Geschwindigkeiten (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="messdaten.csv",
            mime="text/csv",
            use_container_width=True
        )

# --- Hauptbereich ---
if st.session_state.strobe_preview is not None:
    st.sidebar.header("3. Analyse-Tools")
    zoom = st.sidebar.slider("🔍 Zoom (für Klicks)", 1.0, 5.0, 1.5)
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
        st.warning(f"📏 Schritt 1: Referenz markieren ({len(st.session_state.clicks_ref)+1}/2)")
        frame_to_show = 0
    else:
        st.success(f"🎯 Schritt 2: Tracking (Bild {st.session_state.current_frame_idx + 1}/{len(st.session_state.extracted_frames)})")
        frame_to_show = st.session_state.current_frame_idx

    # Bild zeichnen
    pil_img = Image.fromarray(st.session_state.extracted_frames[frame_idx_to_show := frame_to_show]).copy()
    draw = ImageDraw.Draw(pil_img)

    # Referenzlinie (Gelb)
    if len(st.session_state.clicks_ref) == 2:
        draw.line(st.session_state.clicks_ref, fill="yellow", width=5)
    
    # Tracking-Punkte (Rot)
    for p in st.session_state.clicks_track:
        draw.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill="red")

    # Interaktives Zoomen
    w, h = pil_img.size
    display_img = pil_img.resize((int(w * zoom), int(h * zoom)))
    value = streamlit_image_coordinates(display_img, key="main_canvas")

    if value:
        rx, ry = value["x"] / zoom, value["y"] / zoom
        new_pt = (rx, ry)
        
        # Punkt setzen Logik
        if len(st.session_state.clicks_ref) < 2:
            if not st.session_state.clicks_ref or (abs(st.session_state.clicks_ref[-1][0] - rx) > 1):
                st.session_state.clicks_ref.append(new_pt)
                st.rerun()
        else:
            if not st.session_state.clicks_track or (abs(st.session_state.clicks_track[-1][0] - rx) > 0.5):
                st.session_state.clicks_track.append(new_pt)
                if st.session_state.current_frame_idx < len(st.session_state.extracted_frames) - 1:
                    st.session_state.current_frame_idx += 1
                st.rerun()

    # Controls unterm Bild
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⬅️ Zurück"):
            st.session_state.current_frame_idx = max(0, st.session_state.current_frame_idx - 1)
            st.rerun()
    with c2:
        if st.button("Vorwärts ➡️"):
            st.session_state.current_frame_idx = min(len(st.session_state.extracted_frames)-1, st.session_state.current_frame_idx + 1)
            st.rerun()
    with c3:
        if st.button("🗑️ Letzten Punkt löschen"):
            if st.session_state.clicks_track:
                st.session_state.clicks_track.pop()
                st.session_state.current_frame_idx = max(0, st.session_state.current_frame_idx - 1)
                st.rerun()

    # Auswertungstabelle anzeigen
    if len(st.session_state.clicks_track) >= 2:
        st.divider()
        st.subheader("Berechnete Geschwindigkeiten")
        # (Wiederholung der Berechnung für die Anzeige)
        px_dist = np.sqrt((st.session_state.clicks_ref[0][0]-st.session_state.clicks_ref[1][0])**2 + (st.session_state.clicks_ref[0][1]-st.session_state.clicks_ref[1][1])**2)
        m_per_px = real_dist / px_dist
        f = st.session_state.video_info["freq"]
        res = []
        for i in range(len(st.session_state.clicks_track)-1):
            p_a, p_b = st.session_state.clicks_track[i], st.session_state.clicks_track[i+1]
            d_m = np.sqrt((p_a[0]-p_b[0])**2 + (p_a[1]-p_b[1])**2) * m_per_px
            v = d_m * f
            res.append({"Intervall": f"P{i+1}➔P{i+2}", "v (m/s)": round(v, 2), "v (km/h)": round(v*3.6, 2)})
        st.table(pd.DataFrame(res))
else:
    st.info("Bitte Video hochladen und links 'Video verarbeiten' klicken.")
