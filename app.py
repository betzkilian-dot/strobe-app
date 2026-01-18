import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Seite konfigurieren für mobile Ansicht
st.set_page_config(page_title="Video Strobe", layout="centered")

def create_transparent_strobe(video_path, frequency, alpha):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30 # Fallback
    
    frame_interval = max(1, int(fps / frequency))
    frames = []
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        count += 1
    cap.release()
    
    if not frames:
        return None

    accumulator = frames[0].astype(np.float32)
    for i in range(1, len(frames)):
        current_frame = frames[i].astype(np.float32)
        accumulator = cv2.addWeighted(accumulator, 1.0 - alpha, current_frame, alpha, 0)
        
    return np.clip(accumulator, 0, 255).astype(np.uint8)

st.title("📸 Stroboskop-Generator")
st.write("Erstelle Mehrfachbelichtungen direkt im Browser.")

# Upload-Bereich
uploaded_file = st.file_uploader("Video hochladen (MP4, MOV)", type=["mp4", "mov", "avi"])

# Regler für die Einstellungen
freq = st.slider("Bilder pro Sekunde (Frequenz)", 0.5, 10.0, 2.0, 0.5)
alpha_val = st.slider("Transparenz (Alpha)", 0.05, 1.0, 0.3, 0.05)

if uploaded_file is not None:
    # Temporäre Datei erstellen
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name

    if st.button("Bild jetzt generieren", type="primary"):
        with st.spinner('Verarbeite Video...'):
            result_img = create_transparent_strobe(temp_path, freq, alpha_val)
            
            if result_img is not None:
                st.image(result_img, caption="Dein Stroboskop-Bild", use_column_width=True)
                
                # Download-Button für das iPad
                res_pil = Image.fromarray(result_img)
                res_pil.save("strobe_result.png")
                with open("strobe_result.png", "rb") as file:
                    st.download_button(
                        label="Bild auf iPad speichern",
                        data=file,
                        file_name="stroboskop_aufnahme.png",
                        mime="image/png"
                    )
            else:
                st.error("Fehler bei der Verarbeitung.")
    
    # Datei löschen
    os.unlink(temp_path)