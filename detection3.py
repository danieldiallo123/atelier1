from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Initialisation du modèle de détection YOLO
model = YOLO('yolo11n.pt')

# Initialisation du tracker DeepSORT avec réglages plus stricts
tracker = DeepSort(
    max_age=30,
    n_init=3,  # nombre minimum de frames avant qu'une détection soit "confirmée"
    max_iou_distance=0.7  # plus bas = plus strict sur l'association
)

# Set pour stocker les personnes déjà comptées
counted_ids = set()

# Ouverture de la vidéo
cap = cv2.VideoCapture('data/videos/person.mp4')

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf > 0.4:  # filtre confiance
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        # Mise à jour du tracker avec les détections filtrées
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if track_id not in counted_ids:
                counted_ids.add(track_id)

        # Affichage du compteur
        cv2.putText(frame, f"Total personnes: {len(counted_ids)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        resized = cv2.resize(frame, (1200, 600))
        cv2.imshow("YOLOv11n + DeepSORT", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
