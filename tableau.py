import time
import cv2
from ultralytics import YOLO
import pandas as pd

# Liste des modèles YOLO à tester
models_to_test = {
    'YOLOv8n': 'yolov8n.pt',
    'YOLOv8s': 'yolov8s.pt',
    'YOLOv8m': 'yolov8m.pt',
    'YOLOv8l': 'yolov8l.pt',
}

# Chemin de la vidéo de test
video_path = 'data/videos/person1.mp4'

# Nombre de frames à tester
NUM_FRAMES = 100

results = []

for model_name, model_path in models_to_test.items():
    print(f"\nTesting {model_name}...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()

    while frame_count < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        _ = model(frame)  # Détection
        frame_count += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time

    results.append({
        "Modèle": model_name,
        "FPS moyen": round(fps, 2),
        "Frames testées": frame_count,
        "Temps total (s)": round(elapsed_time, 2)
    })

    cap.release()

# Création d'un DataFrame et affichage du tableau
results_df = pd.DataFrame(results)
print("\nRésultats comparatifs de FPS :")
print(results_df.to_markdown(index=False))
