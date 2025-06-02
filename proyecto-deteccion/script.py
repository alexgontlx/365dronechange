import streamlit as st
import torch
import cv2
import os
from PIL import Image
import glob

# CONFIG
MODEL_PATH = "yolov5s.pt"  # o "runs/train/exp/weights/best.pt"
OUTPUT_FOLDER = "frames_detectados"
CONFIDENCE_THRESHOLD = 0.4

# Interfaz
st.set_page_config(page_title="Detector de Personas con Dron", layout="centered")
st.title("🕵️ Detección de Personas en Vídeos de Drones")
st.markdown("Sube un vídeo y detectaremos personas usando YOLOv5.")

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False)
    model.conf = CONFIDENCE_THRESHOLD
    return model

model = cargar_modelo()

# Subida del vídeo
video_file = st.file_uploader("🎥 Sube un vídeo (.mp4)", type=["mp4"])

if video_file is not None:
    # Guardar temporalmente el vídeo
    VIDEO_PATH = "video_subido.mp4"
    with open(VIDEO_PATH, "wb") as f:
        f.write(video_file.read())

    # Procesamiento
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0
    frames_guardados = 0

    st.info("⏳ Procesando vídeo...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0]  # x1, y1, x2, y2, conf, class

        for *box, conf, cls in detections:
            if int(cls) == 0:  # clase 'person'
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Persona {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if any(int(cls) == 0 for *_, cls in detections):
            output_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            frames_guardados += 1

        frame_count += 1
        if frame_count > 90:
            break

    cap.release()
    st.success(f"✅ Proceso finalizado. Se detectaron personas en {frames_guardados} frames.")

    # Mostrar los primeros 5 frames detectados
    st.markdown("### 🖼️ Ejemplos de detecciones")

    imagenes = sorted(glob.glob(f"{OUTPUT_FOLDER}/*.jpg"))[:5]
    if imagenes:
        for img_path in imagenes:
            st.image(Image.open(img_path), caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.warning("⚠️ No se detectaron personas visibles en los frames.")
