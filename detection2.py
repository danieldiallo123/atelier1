from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Initialisation du détecteur et du tracker
model = YOLO('yolo11n.pt')
tracker = DeepSort(max_age=60, n_init=2, nms_max_overlap=1.0)

# Zone virtuelle (ligne) de comptage
line_y = 500

# Pour mémoriser les IDs comptés une seule fois
counted_ids = set()

# Vidéo d'entrée
cap = cv2.VideoCapture('data/videos/person1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Détection des personnes
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0:  # Classe "person"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Mise à jour du tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    current_ids = set()
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cx = (l + r) // 2
        cy = (t + b) // 2

        # Affichage
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        # Comptage si traversé la ligne et jamais compté
        if cy > line_y - 10 and cy < line_y + 10 and track_id not in counted_ids:
            counted_ids.add(track_id)

        current_ids.add(track_id)

    # Ligne virtuelle de comptage
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

    # Affichage des compteurs
    cv2.putText(frame, f"Total: {len(counted_ids)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(frame, f"Actives: {len(current_ids)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Affichage
    frame_resized = cv2.resize(frame, (1400, 800))
    cv2.imshow("YOLO + DeepSORT - Compteur", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
