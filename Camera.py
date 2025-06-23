import cv2
import numpy as np

# === CONFIGURATION ===
model_path = "yolo11n.onnx"  # Ton modèle ONNX
video_path = "data/videos/person.mp4"  # Mets ici le chemin réel de ta vidéo locale
image_size = 640
conf_threshold = 0.3

# === CHARGER LE MODÈLE ===
def load_model(onnx_path):
    net = cv2.dnn.readNetFromONNX(onnx_path)
    return net

# === DÉTECTION DE PERSONNES ===
def detect_persons(net, frame, image_size=640, conf_threshold=0.3):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (image_size, image_size), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    outputs = outputs[0]  # shape: [N, 85]

    h, w, _ = frame.shape
    x_factor = w / image_size
    y_factor = h / image_size

    boxes = []
    confidences = []

    for row in outputs:
        confidence = row[4]
        class_scores = row[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]

        # Classe "person" uniquement (ID 0 dans COCO)
        if confidence > conf_threshold and class_id == 0:
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            left = int((cx - bw / 2) * x_factor)
            top = int((cy - bh / 2) * y_factor)
            width = int(bw * x_factor)
            height = int(bh * y_factor)
            boxes.append([left, top, width, height])
            confidences.append(float(class_conf))

    return boxes, confidences

# === MAIN ===
if __name__ == "__main__":
    net = load_model(model_path)
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confidences = detect_persons(net, frame, image_size, conf_threshold)

        # Dessiner les rectangles directement après détection
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "person", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Afficher le compteur
        cv2.putText(frame, f"Personnes détectées: {len(boxes)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("YOLOv11 ONNX - Détection directe", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
