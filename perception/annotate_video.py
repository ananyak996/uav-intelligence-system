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
MATCH_DISTANCE_THRESHOLD = 100    #max distance to consider same object (increased for better tracking)
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
# PERFORMANCE METRICS TRACKING
# ==================================

# Core metrics
total_frames = 0
total_detections = 0
unique_ids_created = 0
successful_matches = 0  # When an existing ID is matched
id_switches = 0  # When an object gets a new ID after disappearing
new_detections = 0  # Truly new objects (not ID switches)

# Motion classification
moving_count = 0
stationary_count = 0

# Tracking duration tracking
object_tracking_durations = {}  # id -> list of frame counts when object was active
active_objects_this_frame = set()

# ID switch detection - track recently disappeared objects
recently_disappeared = {}  # id -> (last_centroid, last_frame, class)



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

    total_frames += 1
    active_objects_this_frame = set()

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
            total_detections += 1

    used_ids = set()



# ==================================
# TRACKING & MOTION STATE
# ==================================


    for centroid, label in detections:
        matched_id = None
        min_dist = float("inf")

        # First try to match with active objects
        for object_id, data in objects.items():
            if data["class"] != label:
                continue

            dist = euclidean(centroid, data["centroid"])
            if dist < min_dist:
                min_dist = dist
                matched_id = object_id

        # Check if this might be a recently disappeared object (potential ID switch)
        potential_id_switch = False
        if matched_id is None or min_dist >= MATCH_DISTANCE_THRESHOLD:
            # Check recently disappeared objects for potential match
            for disappeared_id, (last_centroid, last_frame, disappeared_class) in recently_disappeared.items():
                if disappeared_class != label:
                    continue
                dist_to_disappeared = euclidean(centroid, last_centroid)
                # More lenient threshold for ID switch detection
                if dist_to_disappeared < MATCH_DISTANCE_THRESHOLD * 1.5:
                    if dist_to_disappeared < min_dist:
                        min_dist = dist_to_disappeared
                        matched_id = disappeared_id
                        potential_id_switch = True

        #existing object
        if matched_id is not None and min_dist < MATCH_DISTANCE_THRESHOLD:
            # If this is a recently disappeared object, recreate it
            if matched_id not in objects:
                # Recreate the object from recently_disappeared
                objects[matched_id] = {
                    "centroid": centroid,
                    "history": [centroid],
                    "class": label,
                    "disappeared": 0,
                    "state": "STATIONARY"
                }
                # Track ID switch
                id_switches += 1
                # Remove from recently disappeared since it's now tracked again
                if matched_id in recently_disappeared:
                    del recently_disappeared[matched_id]
            
            obj = objects[matched_id]
            obj["centroid"] = centroid
            obj["history"].append(centroid)

            if len(obj["history"]) > HISTORY_LENGTH:
                obj["history"].pop(0)

            obj["disappeared"] = 0
            used_ids.add(matched_id)
            active_objects_this_frame.add(matched_id)

            # Track successful match
            successful_matches += 1

            # Update tracking duration
            if matched_id not in object_tracking_durations:
                object_tracking_durations[matched_id] = []
            object_tracking_durations[matched_id].append(total_frames)

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
                "state": "STATIONARY",
                "first_frame": total_frames
            }
            used_ids.add(next_object_id)
            active_objects_this_frame.add(next_object_id)
            unique_ids_created += 1
            new_detections += 1
            
            # Initialize tracking duration
            object_tracking_durations[next_object_id] = [total_frames]
            
            next_object_id += 1

    #handling disappeared objects
    for object_id in list(objects.keys()):
        if object_id not in used_ids:
            objects[object_id]["disappeared"] += 1
            if objects[object_id]["disappeared"] > MAX_DISAPPEARED:
                # Store in recently disappeared for ID switch detection
                obj = objects[object_id]
                recently_disappeared[object_id] = (
                    obj["centroid"],
                    total_frames,
                    obj["class"]
                )
                # Clean up old entries (keep only last 30 frames worth)
                if len(recently_disappeared) > 50:
                    # Remove oldest entries
                    oldest_id = min(recently_disappeared.keys(), 
                                  key=lambda x: recently_disappeared[x][1])
                    del recently_disappeared[oldest_id]
                del objects[object_id]



# ==================================
# VISUALIZATION & OUTPUT
# ==================================


    for object_id, data in objects.items():
        cx, cy = data["centroid"]
        state = data["state"]
        color = (0, 255, 0) if state == "MOVING" else (0, 0, 255)

        # Track motion classification (count once per frame per object)
        if object_id in active_objects_this_frame:
            if state == "MOVING":
                moving_count += 1
            else:
                stationary_count += 1

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

# ==================================
# PERFORMANCE METRICS OUTPUT
# ==================================

# Calculate tracking accuracy (successful ID persistence)
# Tracking accuracy = (successful matches) / (successful matches + id_switches + new_detections)
total_tracking_attempts = successful_matches + id_switches + new_detections
tracking_accuracy = (successful_matches / total_tracking_attempts * 100) if total_tracking_attempts > 0 else 0

# Calculate average tracking duration per object
avg_tracking_duration = 0
if object_tracking_durations:
    total_duration = sum(len(frames) for frames in object_tracking_durations.values())
    avg_tracking_duration = total_duration / len(object_tracking_durations)

# Calculate average detections per frame
avg_detections_per_frame = total_detections / total_frames if total_frames > 0 else 0

# Calculate average objects per frame using tracking duration data
if object_tracking_durations and total_frames > 0:
    total_object_frames = sum(len(frames) for frames in object_tracking_durations.values())
    avg_objects_per_frame = total_object_frames / total_frames
else:
    avg_objects_per_frame = 0

# Print comprehensive metrics
print("\n" + "="*60)
print("TRACKING PERFORMANCE METRICS:")
print("="*60)
print(f"- Total frames: {total_frames}")
print(f"- Total detections: {total_detections}")
print(f"- Unique objects: {unique_ids_created}")
print(f"- Tracking accuracy: {tracking_accuracy:.2f}% (successful ID persistence)")
print(f"- Moving objects: {moving_count}")
print(f"- Stationary objects: {stationary_count}")
print(f"- Avg detections/frame: {avg_detections_per_frame:.2f}")
print("\n" + "-"*60)
print("DETAILED METRICS:")
print("-"*60)
print(f"- Successful matches: {successful_matches}")
print(f"- ID switches: {id_switches}")
print(f"- New detections: {new_detections}")
print(f"- Average tracking duration per object: {avg_tracking_duration:.2f} frames")
print(f"- Avg objects per frame: {avg_objects_per_frame:.2f}")
print("="*60 + "\n")
