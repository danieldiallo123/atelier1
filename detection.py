from ultralytics import YOLO
import cv2

# Charger le modèle
model = YOLO('yolo11n.pt')  # Assure-toi que ce modèle est compatible Ultralytics

# Ouvrir la vidéo (ou mettre 0 pour webcam)
cap = cv2.VideoCapture('data/videos/person1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Exécuter la détection
    results = model(frame)[0]  # batch[0] si batch_size=1

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0:  # Classe 0 = 'person'
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    resized_frame = cv2.resize(frame, (1400, 800))
    cv2.imshow("YOLOv11n - Person Detection", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
