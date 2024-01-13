from flask import Flask, render_template, Response
import cv2
import torch
from pathlib import Path

app = Flask(__name__)
video_capture = cv2.VideoCapture(0)  # Cambia a 1 si estás usando la cámara externa

# Cargar el modelo YOLOv5
model_path = Path("models/hemletYoloV8_100epochs.pt")
model = torch.hub.load('ultralytics/yolov5:v5.0', 'custom', path_or_model=model_path)

def process_frame(frame):
    # Tu lógica de procesamiento de fotogramas aquí
    # Puedes usar el modelo YOLOv5 para detectar objetos en el frame
    results = model(frame)
    output_frame = results.render()[0]

    return output_frame

def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            modified_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', modified_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
