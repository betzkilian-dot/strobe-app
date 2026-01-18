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
    st.session_state.clicks = []  # Speichert (x, y)
if "current_frame_idx" not in st.session_state:
    st.session_state.current_frame_idx = 0
if "extracted_frames" not in st.session_state:
    st.session_state.extracted_frames = []
if "video_info" not in st.session_state:
    st.session_state.video_info = {}

def get_video_data(video_path, num_images):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    # Berechne Schrittweite, um auf die gewünschte Anzahl Bilder zu kommen
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
    return frames, actual_freq, duration

# --- UI Layout ---
st.title("📸 Physik Stroboskop-Analyse")

# Sidebar
st.sidebar.header("1. Video-Konfiguration")
uploaded_file = st.sidebar.file_uploader("Video laden", type=["mp4", "mov"])

if uploaded_file:
    # Temporäres Speichern für OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    # Video-Eigenschaften auslesen (einmalig)
    cap_temp = cv2.VideoCapture(tfile.name)
    max_f = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_orig = cap_temp.get(cv2.CAP_PROP_FPS)
    cap_temp.release()

    # Anzahl der Bilder & Frequenzanzeige
    num_images = st.sidebar.slider("Anzahl der Bilder", 2, min(max_f, 50), 10)
    
    # Frequenz berechnen für die Anzeige
    step_calc = max(1, max_f // num_images)
    freq_calc = fps_orig / step_calc
    st.sidebar.info(f"Effektive Frequenz: ({freq_calc:.2f} Hz)")

    if st.sidebar.button("Video verarbeiten / Reset"):
        frames, freq, dur = get_video_data(tfile.name, num_images)
        st.session_state.extracted_frames = frames
        st.session_state.video_info = {"freq": freq, "dur": dur}
        st.session_state.clicks = []
        st.session_state.current_frame_idx = 0
    
    os.unlink(tfile.name)

# --- Analyse-Bereich ---
if st.session_state.extracted_frames:
    frames = st.session_state.extracted_frames
    idx = st.session_state.current_frame_idx
    
    st.sidebar.header("2. Mess-Optionen")
    real_dist = st.sidebar.number_input("Referenzstrecke (m)", value=1.0)
    zoom = st.sidebar.slider("Zoom Faktor", 1.0, 4.0, 1.5)
    
    # Statusanzeige
    if len(st.session_state.clicks) < 2:
        st.info(f"👉 Schritt 1: Markiere die Referenzstrecke auf dem Bild (Klick {len(st.session_state.clicks)+1}/2)")
    else:
        st.success(f"👉 Schritt 2: Verfolge das Objekt. Aktuelles Bild: {idx+1} von {len(frames)}")

    # Aktuelles Bild zum Klicken vorbereiten
    current_img = Image.fromarray(frames[idx])
    w, h = current_img.size
    
    # Zeichne bisherige Klicks zur Orientierung ein
    draw = ImageDraw.Draw(current_img)
    for p in st.session_state.clicks:
        draw.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill="red", outline="white")

    # Zoom für iPad-Anzeige
    display_img = current_img.resize((int(w * zoom), int(h * zoom)))
    
    # Interaktion
    value = streamlit_image_coordinates(display_img, key="tracker")

    if value:
        rx, ry = value["x"] / zoom, value["y"] / zoom
        new_point = (rx, ry)
        
        # Nur speichern, wenn es ein neuer Klick ist
        if not st.session_state.clicks or (abs(st.session_state.clicks[-1][0] - rx) > 1):
            st.session_state.clicks.append(new_point)
            
            # Logik für das Springen zum nächsten Bild
            if len(st.session_state.clicks) > 2: # Erst nach der Kalibrierung springen
                if st.session_state.current_frame_idx < len(frames) - 1:
                    st.session_state.current_frame_idx += 1
            st.rerun()

    # Steuerung
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        if st.button("⬅️ Vorheriges Bild"):
            st.session_state.current_frame_idx = max(0, idx - 1)
            st.rerun()
    with col2:
        if st.button("➡️ Nächstes Bild"):
            st.session_state.current_frame_idx = min(len(frames) - 1, idx + 1)
            st.rerun()
    with col3:
        if st.button("🗑️ Letzten Punkt löschen"):
            if st.session_state.clicks:
                st.session_state.clicks.pop()
                if len(st.session_state.clicks) >= 2:
                    st.session_state.current_frame_idx = max(0, idx - 1)
                st.rerun()

    # --- Auswertung ---
    st.divider()
    if st.button("📊 Geschwindigkeiten berechnen", type="primary"):
        points = st.session_state.clicks
        if len(points) < 4:
            st.error("Bitte markiere die Referenz (2 Klicks) und mindestens 2 Positionen des Objekts.")
        else:
            # Maßstab
            p1, p2 = points[0], points[1]
            px_dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            m_per_px = real_dist / px_dist
            
            f = st.session_state.video_info["freq"]
            data = []
            path_pts = points[2:]
            
            for i in range(len(path_pts)-1):
                d_px = np.sqrt((path_pts[i][0]-path_pts[i+1][0])**2 + (path_pts[i][1]-path_pts[i+1][1])**2)
                d_m = d_px * m_per_px
                v = d_m * f
                data.append({
                    "Bild-Intervall": f"{i+1} ➔ {i+2}",
                    "Weg (m)": round(d_m, 3),
                    "v (m/s)": round(v, 2),
                    "v (km/h)": round(v * 3.6, 2)
                })
            
            df = pd.DataFrame(data)
            st.subheader("Analyse-Ergebnisse")
            st.table(df)
            
            # Finales Stroboskop-Bild als Vorschau
            st.subheader("Stroboskop-Vorschau (Alle Ebenen)")
            alpha = 0.3
            acc = frames[0].astype(np.float32)
            for f_img in frames[1:]:
                acc = cv2.addWeighted(acc, 1-alpha, f_img.astype(np.float32), alpha, 0)
            
            res_img = Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))
            draw_res = ImageDraw.Draw(res_img)
            # Pfad einzeichnen
            if len(points) > 3:
                draw_res.line(points[2:], fill="red", width=3)
            st.image(res_img, use_column_width=True)

else:
    st.info("Bitte lade ein Video hoch und klicke auf 'Video verarbeiten'.")
