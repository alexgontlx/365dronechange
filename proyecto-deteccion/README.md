```markdown
# 🕵️ Detección de Personas con Drones usando YOLOv5 + Streamlit

Este proyecto permite **detectar personas en vídeos capturados por drones** mediante el modelo **YOLOv5**. La detección se realiza a través de una **interfaz web desarrollada con Streamlit**, que permite subir un vídeo `.mp4`, analizarlo y visualizar los fotogramas donde se detectan personas.

---

## 🚀 Requisitos

Asegúrate de tener **Python 3.8+** instalado y ejecuta:

```bash
pip install torch torchvision opencv-python-headless streamlit Pillow
```

---

## ▶️ Cómo ejecutar el script

```bash
streamlit run app.py
```

---

## 🧠 ¿Cómo funciona?

1. Se carga el modelo **YOLOv5** con un umbral mínimo de confianza.
2. Se abre el vídeo subido y se procesan los **primeros 90 fotogramas**.
3. Se detectan personas (clase **0** del dataset COCO) en cada fotograma.
4. Si se detecta al menos una persona, se guarda la imagen procesada.
5. Se muestran los **primeros 5 fotogramas con detecciones** en la interfaz web.

---

## 📁 Archivos generados

- 📂 `frames_detectados/`: contiene los fotogramas donde se han detectado personas.
- 📹 `video_subido.mp4`: es el vídeo temporal generado a partir de la subida del usuario.

---

## 📷 Capturas esperadas

En la interfaz se mostrarán hasta **5 imágenes representativas**, con personas enmarcadas y acompañadas por su porcentaje de confianza.

---

## ⚙️ Personalización

- Puedes usar tu propio modelo entrenado, sustituyendo `yolov5s.pt` por otro en `MODEL_PATH`.
- Cambia el umbral mínimo de confianza modificando la variable:

```python
CONFIDENCE_THRESHOLD = 0.4
```

- Incrementa la cantidad de fotogramas a analizar cambiando:

```python
if frame_count > 90:
    break
```

---

## 🛠 Tecnologías utilizadas

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

---

## 👤 Autor

📛 Proyecto desarrollado por **365dronechange**

📷 Instagram: [@365dronechange](https://instagram.com/365dronechange)  
🔗 GitHub: [github.com/tu_usuario](https://github.com/tu_usuario)
```
