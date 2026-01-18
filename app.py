import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Physik Strobe Pro", layout="wide")

# Session State Initialisierung
if "clicks" not in st.session_state:
    st.session_state.clicks = [] # Speichert Koordinaten im Originalmaßstab
if "strobe_img" not in st.session_state:
    st.session_state.strobe_img = None
if "show_results" not in st.session_state:
    st.session_state.show_results = False

def create_transparent_strobe(video_path, frequency, alpha):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    frame_interval = max(1, int(fps / frequency))
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    cap.release()
    if not frames: return None
    acc = frames[0].astype(np.float32)
    for i in range(1, len(frames)):
        acc = cv2.addWeighted(acc, 1.0 - alpha, frames[i].astype(np.float32), alpha, 0)
    return np.clip(acc, 0, 255).astype(np.uint8)

st.title("📸 Physik Stroboskop Analyse Pro")

# --- Sidebar ---
st.sidebar.header("1. Video & Strobe")
uploaded_file = st.sidebar.file_uploader("Video laden", type=["mp4", "mov"])
freq = st.sidebar.slider("Frequenz (Hz / Bilder pro Sek.)", 0.5, 30.0, 5.0)
alpha = st.sidebar.slider("Transparenz", 0.05, 1.0, 0.3)

st.sidebar.header("2. Messung")
real_dist = st.sidebar.number_input("Referenzstrecke (m)", value=1.0)
zoom_factor = st.sidebar.slider("Zoom Faktor (für genaues Klicken)", 1.0, 4.0, 1.0, 0.5)
show_path = st.sidebar.checkbox("Rote Linie anzeigen", value=True)

if st.sidebar.button("Alle Punkte löschen"):
    st.session_state.clicks = []
    st.session_state.show_results = False
    st.rerun()

# --- Hauptlogik ---
if uploaded_file:
    if st.session_state.strobe_img is None or st.sidebar.button("Strobe neu berechnen"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            st.session_state.strobe_img = create_transparent_strobe(tfile.name, freq, alpha)
        os.unlink(tfile.name)

    if st.session_state.strobe_img is not None:
        # 1. Originalbild vorbereiten
        base_img = Image.fromarray(st.session_state.strobe_img)
        width, height = base_img.size
        
        # 2. Zeichnen auf einer Kopie (im Originalmaßstab)
        draw_img = base_img.copy()
        draw = ImageDraw.Draw(draw_img)
        points = st.session_state.clicks
        
        # Referenzlinie (Gelb)
        if len(points) >= 2:
            draw.line([points[0], points[1]], fill="yellow", width=5)
        
        # Pfad (Rot)
        if show_path and len(points) >= 3:
            draw.line(points[2:], fill="red", width=3)
            for p in points[2:]:
                draw.ellipse([p[0]-8, p[1]-8, p[0]+8, p[1]+8], fill="red")

        # 3. Zoom anwenden für die Anzeige
        new_size = (int(width * zoom_factor), int(height * zoom_factor))
        display_img = draw_img.resize(new_size)

        # 4. Interaktives Bild
        st.write(f"Anzeige-Zoom: {zoom_factor}x. Scrolle zum gewünschten Bereich.")
        value = streamlit_image_coordinates(display_img, key="strobe_pro")

        if value:
            # Klick zurückrechnen auf Originalgröße
            real_x = value["x"] / zoom_factor
            real_y = value["y"] / zoom_factor
            new_click = (real_x, real_y)
            
            if not st.session_state.clicks or (abs(st.session_state.clicks[-1][0] - real_x) > 1):
                st.session_state.clicks.append(new_click)
                st.rerun()

        # --- Auswertung ---
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Geschwindigkeiten berechnen", type="primary"):
                st.session_state.show_results = True

        if st.session_state.show_results:
            if len(points) < 4:
                st.warning("Bitte markiere die Referenz (2 Klicks) und mindestens 2 Ballpositionen.")
            else:
                # Pixel-Maßstab berechnen
                p1, p2 = points[0], points[1]
                px_dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                scale = px_dist / real_dist # Pixel pro Meter
                
                # Daten für Tabelle
                data = []
                path_pts = points[2:]
                for i in range(len(path_pts) - 1):
                    pa, pb = path_pts[i], path_pts[i+1]
                    d_px = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
                    d_m = d_px / scale
                    v = d_m * freq
                    data.append({
                        "Intervall": f"Punkt {i+1} -> {i+2}",
                        "Distanz (m)": round(d_m, 3),
                        "v (m/s)": round(v, 2),
                        "v (km/h)": round(v * 3.6, 2)
                    })
                
                df = pd.DataFrame(data)
                st.subheader("Analyse-Daten")
                st.table(df)
                
                # Download Button für CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Daten als CSV exportieren", csv, "messung.csv", "text/csv")

        # Finales Bild Download
        st.subheader("Vorschau Export")
        st.image(draw_img, caption="Originalauflösung mit Einzeichnungen", use_column_width=True)

