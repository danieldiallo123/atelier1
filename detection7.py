import streamlit as st
import cv2
import time
import pandas as pd
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine

# Fonction de comparaison de vecteurs
def is_new_person(feature, known_descriptors, threshold=0.5):
    for known in known_descriptors:
        if cosine(feature, known) < threshold:
            return False
    return True

# Fonction d'export (optionnelle ici si besoin)
def export_stats_to_excel(count, frame_person_counts, total_frames, video_duration, output_dir='exports'):
    os.makedirs(output_dir, exist_ok=True)
    mean_persons_per_frame = sum(frame_person_counts) / len(frame_person_counts) if frame_person_counts else 0
    max_persons_per_frame = max(frame_person_counts) if frame_person_counts else 0
    duration_avg_per_person = video_duration / count if count > 0 else 0
    stats = {
        "Statistique": [
            "Nombre total de personnes uniques",
            "Nombre moyen de personnes par frame",
            "Nombre maximum de personnes simultanées",
            "Nombre total de frames analysées",
            "Durée totale (s)",
            "Durée moyenne de présence par personne (s)"
        ],
        "Valeur": [
            count,
            round(mean_persons_per_frame, 2),
            max_persons_per_frame,
            total_frames,
            round(video_duration, 2),
            round(duration_avg_per_person, 2)
        ]
    }
    df_stats = pd.DataFrame(stats)
    output_path = os.path.join(output_dir, "statistiques_webcam.xlsx")
    df_stats.to_excel(output_path, index=False)
    return output_path

# Interface Streamlit
st.set_page_config(layout="wide")
st.markdown("### Système intelligent de détection et de comptage d’objets en temps réel")

# Chargement modèle
model = YOLO("yolo11n.pt")
tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)

# Paramètres ajustables
similarity_threshold = st.slider("Seuil de similarité (cosine distance)", 0.3, 0.9, 0.65, step=0.01)
confidence_threshold = st.slider("Seuil de confiance YOLO", 0.1, 1.0, 0.3, step=0.05)
use_webcam = st.checkbox("Activer la webcam", value=False)

if use_webcam:
    cap = cv2.VideoCapture(0)

    known_descriptors = []
    frame_person_counts = []
    stframe = st.empty()
    fps_text = st.empty()
    count_text = st.empty()

    prev_time = time.time()
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Impossible de lire depuis la webcam.")
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        persons_in_frame = 0

        for track in tracks:
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            feature = track.features[0] if hasattr(track, "features") and track.features else None
            if feature is not None and is_new_person(feature, known_descriptors, threshold=similarity_threshold):
                known_descriptors.append(feature)

            persons_in_frame += 1

        frame_person_counts.append(persons_in_frame)
        total_frames += 1
        curr_time = time.time()
        fps = total_frames / (curr_time - prev_time)

        resized = cv2.resize(frame, (960, 540))
        stframe.image(resized, channels="BGR")
        fps_text.markdown(f"**FPS :** {fps:.2f}")
        count_text.markdown(
            f"**Personnes uniques détectées :** {len(known_descriptors)}<br>"
            f"**Personnes visibles dans la frame :** {persons_in_frame}",
            unsafe_allow_html=True
        )

    cap.release()
else:
    st.info("Activer la Webcam ici !")
