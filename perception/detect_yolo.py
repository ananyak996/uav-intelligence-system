from ultralytics import YOLO
import cv2

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")  # nano model = fastest, sufficient

# Video input
video_path = "pedestrians_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# COCO class IDs
PERSON_CLASS_ID = 0
CAR_CLASS_ID = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.45, verbose=False)

    # Extract detections
    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls_id = int(box.cls[0])

            # Filter only person and car
            if cls_id not in [PERSON_CLASS_ID, CAR_CLASS_ID]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = "Person" if cls_id == PERSON_CLASS_ID else "Car"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 150), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255,200,150),
                2
            )

    cv2.imshow("YOLO Detection - Phase A", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
