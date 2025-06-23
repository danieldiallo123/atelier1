import streamlit as st
import cv2
import tempfile
import time
import os
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine

# Fonction pour comparer les vecteurs
def is_new_person(feature, known_descriptors, threshold=0.4):
    for known in known_descriptors:
        if cosine(feature, known) < threshold:
            return False
    return True

# Fonction pour exporter les statistiques
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
            "Durée totale de la vidéo (s)",
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
    output_path = os.path.join(output_dir, "statistiques_detection.xlsx")
    df_stats.to_excel(output_path, index=False)
    return output_path

# Streamlit App
st.set_page_config(layout="wide")
st.title("Système intelligent de détection et de comptage d’objets en temps réel")

model = YOLO("yolo11n.pt")
tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)

uploaded_video = st.file_uploader("Uploader une vidéo", type=["mp4", "avi"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

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
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        persons_in_frame = 0

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track.track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            feature = track.features[0] if hasattr(track, "features") and track.features else None

            if feature is not None and is_new_person(feature, known_descriptors):
                known_descriptors.append(feature)

            persons_in_frame += 1

        frame_person_counts.append(persons_in_frame)
        total_frames += 1
        curr_time = time.time()
        fps = total_frames / (curr_time - prev_time)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Personnes uniques: {len(known_descriptors)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized = cv2.resize(frame, (960, 540))
        stframe.image(resized, channels="BGR")
        fps_text.markdown(f"**FPS actuel :** {fps:.2f}")
        count_text.markdown(f"**Total personnes uniques détectées :** {len(known_descriptors)}")

    cap.release()
    os.remove(tfile.name)

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video > 0:
        video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps_video
    else:
        video_duration = total_frames / fps if fps > 0 else 0

    path_xlsx = export_stats_to_excel(len(known_descriptors), frame_person_counts, total_frames, video_duration)
    st.success("✅ Statistiques exportées avec succès.")
    with open(path_xlsx, "rb") as f:
        st.download_button("📥 Télécharger le fichier Excel", f, file_name="statistiques_detection.xlsx")
else:
    st.info("Charger une vidéo ici !")
