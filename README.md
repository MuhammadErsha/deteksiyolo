# deteksiyolo
import cv2
import numpy as np

# Path ke file model
model_cfg = "yolov4.cfg"  # Ganti dengan path ke file konfigurasi YOLO
model_weights = "yolov4.weights"  # Ganti dengan path ke file berat model
labels_path = "coco.names"  # Ganti dengan path ke file nama kelas

# Muat label kelas
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Muat model YOLO
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Fungsi untuk mendeteksi objek
def detect_objects(frame):
    height, width, _ = frame.shape

    # Preprocessing gambar
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Jalankan forward pass
    outputs = net.forward(output_layers)

    # Ekstrak data deteksi
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Threshold kepercayaan
                center_x, center_y, w, h = (
                    int(detection[0] * width),
                    int(detection[1] * height),
                    int(detection[2] * width),
                    int(detection[3] * height),
                )
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression untuk menghindari duplikasi
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(labels[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Warna kotak
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return frame

# Buka kamera/webcam
cap = cv2.VideoCapture(0)  # Ganti dengan path video jika ingin memuat video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek
    output_frame = detect_objects(frame)

    # Tampilkan hasil
    cv2.imshow("YOLO Object Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
