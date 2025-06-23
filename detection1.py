from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Initialisation du modèle de détection YOLO
model = YOLO('yolo11n.pt')

# Initialisation du tracker DeepSORT
tracker = DeepSort(max_age=30)  # max_age = frames sans update

# Set pour stocker les personnes déjà comptées
counted_ids = set()

# Ouverture de la vidéo
cap = cv2.VideoCapture('data/videos/person1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0:  # personne uniquement
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Update tracker avec détections actuelles
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        x2, y2 = l + w, t + h

        # Dessiner rectangle + ID
        cv2.rectangle(frame, (l, t), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Compter personne si non déjà comptée
        if track_id not in counted_ids:
            counted_ids.add(track_id)

    # Affichage du compteur en haut à gauche
    cv2.putText(frame, f"Total personnes: {len(counted_ids)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    resized = cv2.resize(frame, (1400, 800))
    cv2.imshow("YOLOv11n + DeepSORT", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
