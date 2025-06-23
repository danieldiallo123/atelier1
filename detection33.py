import streamlit as st
import cv2
import time
import pandas as pd
import os
import numpy as np
from types import SimpleNamespace
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker

# Fonction d'export (inchangée)
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

# Initialisation Streamlit
st.set_page_config(layout="wide")
st.markdown("### Système intelligent de détection et de comptage de personnes en temps réel")

model = YOLO("yolo11n.pt")
confidence_threshold = st.slider("Seuil de confiance YOLO", 0.1, 1.0, 0.3, step=0.05)
mode = st.radio("Choisir une source", ("Charger une vidéo", "Activer la webcam"))

counted_ids = set()
frame_person_counts = []
stframe = st.empty()
fps_text = st.empty()
count_text = st.empty()

if mode == "Activer la webcam":
    cap = cv2.VideoCapture(0)
else:
    uploaded_video = st.file_uploader("Uploader une vidéo", type=["mp4", "avi"])
    if uploaded_video:
        temp_file = os.path.join("temp_video.mp4")
        with open(temp_file, "wb") as f:
            f.write(uploaded_video.read())
        cap = cv2.VideoCapture(temp_file)
    else:
        cap = None

if cap is not None and cap.isOpened():
    tracker_args = SimpleNamespace(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
        mot20=False
    )
    tracker = BYTETracker(tracker_args, frame_rate=30)

    total_frames = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > confidence_threshold:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf])

        if detections:
            dets_np = np.array(detections, dtype=np.float32)
            height, width = frame.shape[:2]
            img_info = (height, width)
            img_size = (height, width)
            online_targets = tracker.update(dets_np, img_info, img_size)
        else:
            online_targets = []

        persons_in_frame = 0

        for t in online_targets:
            if not t.is_activated:
                continue

            track_id = t.track_id
            tlbr = t.tlbr
            l, t_coord, r, b = map(int, tlbr)
            cv2.rectangle(frame, (l, t_coord), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t_coord - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            counted_ids.add(track_id)
            persons_in_frame += 1

        frame_person_counts.append(persons_in_frame)
        total_frames += 1
        elapsed = time.time() - start_time
        fps = total_frames / elapsed if elapsed > 0 else 0

        resized = cv2.resize(frame, (960, 540))
        stframe.image(resized, channels="BGR")
        fps_text.markdown(f"**FPS :** {fps:.2f}")
        count_text.markdown(
            f"**Personnes uniques détectées :** {len(counted_ids)}<br>"
            f"**Personnes visibles dans la frame :** {persons_in_frame}",
            unsafe_allow_html=True
        )

    cap.release()
else:
    st.info("Veuillez activer la Webcam ou charger une vidéo.")
