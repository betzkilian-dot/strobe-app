import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Physik Strobe Analyser", layout="wide")

# Session State Initialisierung
if "clicks" not in st.session_state:
    st.session_state.clicks = []
if "strobe_img" not in st.session_state:
    st.session_state.strobe_img = None
if "scale" not in st.session_state:
    st.session_state.scale = None # Pixel pro Meter

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

st.title("🚀 Physik-Stroboskop-Analyse")
st.write("Anleitung: 1. Video laden & Strobe erzeugen. 2. Die ersten zwei Klicks definieren die Messstrecke. 3. Weitere Klicks markieren die Ballpositionen.")

# --- Sidebar ---
st.sidebar.header("1. Einstellungen")
uploaded_file = st.sidebar.file_uploader("Video hochladen", type=["mp4", "mov"])
freq = st.sidebar.slider("Frequenz (Bilder/Sekunde)", 0.5, 20.0, 5.0)
alpha = st.sidebar.slider("Transparenz (Alpha)", 0.05, 1.0, 0.3)
real_dist = st.sidebar.number_input("Reale Distanz der Messstrecke (m)", value=1.0)
show_path = st.sidebar.checkbox("Pfad und Geschwindigkeit anzeigen", value=True)

if st.sidebar.button("Klicks zurücksetzen"):
    st.session_state.clicks = []
    st.session_state.scale = None
    st.rerun()

# --- Hauptbereich ---
if uploaded_file:
    if st.session_state.strobe_img is None or st.sidebar.button("Stroboskop neu berechnen"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            st.session_state.strobe_img = create_transparent_strobe(tfile.name, freq, alpha)
        os.unlink(tfile.name)

    if st.session_state.strobe_img is not None:
        # Arbeitskopie des Bildes zum Zeichnen
        img_pil = Image.fromarray(st.session_state.strobe_img)
        draw = ImageDraw.Draw(img_pil)
        
        # Logik für Zeichnungen
        points = st.session_state.clicks
        
        # 1. Kalibrierung (erste zwei Punkte)
        if len(points) >= 2:
            p1, p2 = points[0], points[1]
            draw.line([p1, p2], fill="yellow", width=3)
            px_dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            st.session_state.scale = px_dist / real_dist
            draw.text(p1, f"Referenz: {real_dist}m", fill="yellow")

        # 2. Pfad und Geschwindigkeit (ab Punkt 3)
        if show_path and len(points) >= 3:
            path_points = points[2:]
            for i in range(len(path_points)):
                p = path_points[i]
                # Punkt zeichnen
                draw.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5], fill="red")
                
                # Linie und Geschwindigkeit berechnen
                if i > 0:
                    prev_p = path_points[i-1]
                    draw.line([prev_p, p], fill="red", width=2)
                    
                    if st.session_state.scale:
                        px_move = np.sqrt((p[0]-prev_p[0])**2 + (p[1]-prev_p[1])**2)
                        meter_move = px_move / st.session_state.scale
                        velocity = meter_move * freq # v = s / (1/f) = s * f
                        draw.text(p, f"{velocity:.2f} m/s", fill="white")

        # Bild mit Klick-Funktion anzeigen
        value = streamlit_image_coordinates(img_pil, key="strobe_clicks")

        if value:
            new_click = (value["x"], value["y"])
            if not st.session_state.clicks or st.session_state.clicks[-1] != new_click:
                st.session_state.clicks.append(new_click)
                st.rerun()

        # Download des fertigen Analysebildes
        st.subheader("Analyse-Ergebnis")
        st.image(img_pil, use_column_width=True)
        
        # Download Button
        img_pil.save("analyse.png")
        with open("analyse.png", "rb") as f:
            st.download_button("Analyse-Bild speichern", f, "analyse.png", "image/png")
