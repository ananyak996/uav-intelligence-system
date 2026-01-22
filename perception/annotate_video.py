from ultralytics import YOLO
import cv2
import numpy as np


"""
YOLO-Based Situational Awareness System for UAV Perception

This script implements an end-to-end computer vision pipeline for UAV
situational awareness in urban environments. It performs:

1. Object Detection:
   - Uses a pretrained YOLOv8 model to detect persons and vehicles
     in each video frame.

2. Tracking and Memory:
   - Assigns persistent unique IDs to detected objects using
     centroid-based tracking.
   - Maintains short-term object histories to handle occlusions
     and temporary disappearance.

3. Motion State Classification:
   - Analyzes centroid displacement over multiple frames to
     classify objects as MOVING or STATIONARY.
   - Filters out camera jitter using displacement thresholds.

4. Output Generation:
   - Annotates each frame with object ID, class, and motion state.
   - Writes the processed frames to an output video file.

This script is designed as a single-file reference implementation
to clearly demonstrate the full perception pipeline from raw video
input to interpretable situational awareness output.

Inputs:
- Video file containing urban pedestrian and traffic scenes

Outputs:
- Annotated video with ID : CLASS - STATE labels

Note:
- Model weights and video files are excluded from the repository
  due to size constraints.
"""


# ==================================
# CONFIGURATION PARAMETERS
# ==================================


VIDEO_PATH = "pedestrians_video.mp4"

PERSON_CLASS_ID = 0
CAR_CLASS_ID = 2

MAX_DISAPPEARED = 10              #max frames to wait before deregistering object
MATCH_DISTANCE_THRESHOLD = 50     #max distance to consider same object
MOVEMENT_THRESHOLD = 5           #minimum displacement to consider object as moving
HISTORY_LENGTH = 5                #no of frames over which movement is evaluated


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(VIDEO_PATH)


#output video writer - phase C
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "output_video.mp4",
    fourcc,
    fps,
    (width, height)
)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


#tracking memory
objects = {}        # id -> data
next_object_id = 0



# ==================================
# UTILITY FUNCTIONS
# ==================================


def compute_centroid(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


#main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    detections = []



# ==================================
# DETECTION PHASE
# ==================================


    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in [PERSON_CLASS_ID, CAR_CLASS_ID]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            centroid = compute_centroid((x1, y1, x2, y2))
            label = "Person" if cls_id == PERSON_CLASS_ID else "Car"

            detections.append((centroid, label))

    used_ids = set()



# ==================================
# TRACKING & MOTION STATE
# ==================================


    for centroid, label in detections:
        matched_id = None
        min_dist = float("inf")

        for object_id, data in objects.items():
            if data["class"] != label:
                continue

            dist = euclidean(centroid, data["centroid"])
            if dist < min_dist:
                min_dist = dist
                matched_id = object_id

        #existing object
        if matched_id is not None and min_dist < MATCH_DISTANCE_THRESHOLD:
            obj = objects[matched_id]
            obj["centroid"] = centroid
            obj["history"].append(centroid)

            if len(obj["history"]) > HISTORY_LENGTH:
                obj["history"].pop(0)

            obj["disappeared"] = 0
            used_ids.add(matched_id)

            #movement state evaluation
            history = obj["history"]
            if len(history) >= 2:
                dx = history[-1][0] - history[0][0]
                dy = history[-1][1] - history[0][1]
                displacement = np.sqrt(dx**2 + dy**2)

                if displacement > MOVEMENT_THRESHOLD:
                    obj["state"] = "MOVING"
                else:
                    obj["state"] = "STATIONARY"

        #new object
        else:
            objects[next_object_id] = {
                "centroid": centroid,
                "history": [centroid],
                "class": label,
                "disappeared": 0,
                "state": "STATIONARY"
            }
            used_ids.add(next_object_id)
            next_object_id += 1

    #handling disappeared objects
    for object_id in list(objects.keys()):
        if object_id not in used_ids:
            objects[object_id]["disappeared"] += 1
            if objects[object_id]["disappeared"] > MAX_DISAPPEARED:
                del objects[object_id]



# ==================================
# VISUALIZATION & OUTPUT
# ==================================


    for object_id, data in objects.items():
        cx, cy = data["centroid"]
        state = data["state"]
        color = (0, 255, 0) if state == "MOVING" else (0, 0, 255)

        label = f"ID {object_id}: {data['class']} - {state}"

        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(
            frame,
            label,
            (cx - 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Track 2 - Situational Awareness", frame)
    out.write(frame)


    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

