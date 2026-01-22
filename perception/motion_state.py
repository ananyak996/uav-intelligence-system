import cv2

# Load video
video_path = "pedestrians_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

prev_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        continue

    # Frame differencing
    diff = cv2.absdiff(prev_gray, gray)

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw bounding boxes
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore small noise
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show results
    cv2.imshow("Detected Moving Objects", frame)
    cv2.imshow("Motion Mask", thresh)

    prev_gray = gray

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
