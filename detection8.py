import streamlit as st
import cv2
import time
import pandas as pd
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
import numpy as np

# Fonction de comparaison de vecteurs
def is_new_person(feature, known_descriptors, threshold=0.5):
    for known in known_descriptors:
        if cosine(feature, known) < threshold:
            return False
    return True

# Suppression des rectangles très similaires (NMS maison)
def suppress_overlapping_boxes(boxes, iou_threshold=0.6):
    if not boxes:
        return []
    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return [boxes[i] for i in keep]

# Interface Streamlit
st.set_page_config(layout="wide")
st.markdown("### Système intelligent de détection et de comptage d’objets en temps réel")

model = YOLO("yolo11n.pt")
tracker = DeepSort(max_age=30, n_init=5, max_iou_distance=0.6)

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
        all_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width, height = x2 - x1, y2 - y1
                if width * height > 4000:
                    detections.append(([x1, y1, width, height], conf, 'person'))
                    all_boxes.append([x1, y1, x2, y2])

        # Suppression des doublons visuels
        filtered_boxes = suppress_overlapping_boxes(all_boxes)

        # Mise à jour du tracker uniquement avec les boxes conservées
        confirmed_detections = []
        for box in filtered_boxes:
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            confirmed_detections.append(([x1, y1, width, height], 0.9, 'person'))

        tracks = tracker.update_tracks(confirmed_detections, frame=frame)
        persons_in_frame = 0

        for track in tracks:
            if not track.is_confirmed():
                continue

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
