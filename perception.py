"""
Real-Time Object Detection with Proximity Warning System
==========================================================
Uses YOLOv8 (Ultralytics) + OpenCV to detect people, cars, motorcycles,
and bicycles from a webcam feed, estimate approximate distance based on
bounding box size, and raise a warning when an object enters a
"danger zone" (i.e., gets too close to the camera).

Install dependencies first:
    pip install ultralytics opencv-python

Run:
    python yolo_proximity_detector.py

Press 'q' to quit.
"""

import cv2
import time
from ultralytics import YOLO

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

# Classes we care about (COCO class names used by pretrained YOLOv8 models)
TARGET_CLASSES = {"person", "car", "motorcycle", "bicycle"}

# Confidence threshold for detections
CONF_THRESHOLD = 0.45

# Approximate known real-world widths (in meters) used for distance estimation.
# This is a rough pinhole-camera approximation, NOT a calibrated measurement.
REAL_WIDTH_M = {
    "person": 0.5,
    "car": 1.8,
    "motorcycle": 0.8,
    "bicycle": 0.6,
}

# Assumed focal length in pixels. For a real measurement you'd calibrate this
# using an object of known size at a known distance:
#   focal_length = (pixel_width * known_distance) / real_width
FOCAL_LENGTH_PX = 700

# Distance (in meters) below which we consider an object to be in the "danger zone"
DANGER_DISTANCE_M = 1.5
WARNING_DISTANCE_M = 3.0

# Webcam device index (0 = default laptop webcam)
CAMERA_INDEX = 0

# Model weights - 'yolov8n.pt' (nano) is fastest, good for real-time on CPU
MODEL_WEIGHTS = "yolov8n.pt"


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def estimate_distance(class_name, pixel_width):
    """
    Estimate approximate distance to an object using the pinhole camera model:
        distance = (real_width * focal_length) / pixel_width_in_image

    Returns None if the class has no known reference width or pixel_width is 0.
    """
    real_width = REAL_WIDTH_M.get(class_name)
    if not real_width or pixel_width <= 0:
        return None
    distance_m = (real_width * FOCAL_LENGTH_PX) / pixel_width
    return distance_m


def get_zone_color(distance_m):
    """Return a BGR color and zone label based on estimated distance."""
    if distance_m is None:
        return (200, 200, 200), "UNKNOWN"
    if distance_m <= DANGER_DISTANCE_M:
        return (0, 0, 255), "DANGER"       # Red
    elif distance_m <= WARNING_DISTANCE_M:
        return (0, 165, 255), "WARNING"    # Orange
    else:
        return (0, 200, 0), "SAFE"         # Green


# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------

def main():
    print("Loading YOLOv8 model...")
    model = YOLO(MODEL_WEIGHTS)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check CAMERA_INDEX or camera permissions.")
        return

    print("Webcam started. Press 'q' to quit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from webcam.")
            break

        # Run YOLOv8 inference (verbose=False keeps console output clean)
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        danger_alerts = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name not in TARGET_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pixel_width = x2 - x1

            distance_m = estimate_distance(class_name, pixel_width)
            color, zone_label = get_zone_color(distance_m)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Build label text
            if distance_m is not None:
                label = f"{class_name} {conf:.2f} | ~{distance_m:.1f}m | {zone_label}"
            else:
                label = f"{class_name} {conf:.2f} | box:{x2-x1}x{y2-y1}px"

            # Draw label background + text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Also show raw bounding box coordinates
            coord_text = f"({x1},{y1})-({x2},{y2})"
            cv2.putText(frame, coord_text, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

            if zone_label == "DANGER":
                danger_alerts.append((class_name, distance_m))

        # Print console warnings for anything in the danger zone
        for class_name, distance_m in danger_alerts:
            print(f"[ALERT] {class_name} detected at ~{distance_m:.1f}m - DANGER ZONE!")

        # On-screen alert banner
        if danger_alerts:
            cv2.putText(frame, "!!! DANGER: OBJECT TOO CLOSE !!!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        # FPS counter
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("YOLOv8 Proximity Detection - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()