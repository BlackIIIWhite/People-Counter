# Import necessary libraries
import ultralytics                   # For YOLO object detection framework
from ultralytics import YOLO         # Import YOLO class
import cv2                           # OpenCV for image and video processing
import cvzone                        # For drawing bounding boxes and UI elements
import math                          # For mathematical functions (like ceil)
import numpy as np                   # For numerical operations and array handling
from sort import *                   # SORT tracking algorithm for object tracking

# Initialize video capture from file
cap = cv2.VideoCapture('people.mp4')

# Set capture frame width and height (not strictly necessary when resizing later)
cap.set(3, 255)
cap.set(4, 255)

# Load YOLOv8 model (nano version)
model = YOLO("yolov8n.pt")

# Load and resize mask image for region-of-interest filtering
mask = cv2.imread('masks.png')
mask = cv2.resize(mask, (1280, 720))

# Define a line for counting vehicles crossing it
line_1 = [180, 300, 360, 300]  # [x1, y1, x2, y2]
line_2 = [580, 600, 860, 600]
# Initialize a list to hold unique IDs of counted vehicles
total_countUp = []
total_countDown = []

# List of COCO class names YOLO model can detect
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "Mobile", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Initialize SORT tracker with custom parameters
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Main processing loop
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If no frame is returned, video is over
    if not ret:
        break

    # Resize frame to 1280x720
    frame = cv2.resize(frame, (1280, 720))

    # Apply mask to frame (bitwise AND operation — keeps only ROI areas)
    imageRegion = cv2.bitwise_and(frame, mask)

    # Perform object detection using YOLO on the masked frame (silent mode)
    results = model(imageRegion, verbose=False)


    # Array to hold current frame's detections for tracker
    detections = np.empty((0, 5))

    # Iterate over YOLO results (usually one batch per frame)
    for r in results:
        boxes = r.boxes

        # Loop through each detected bounding box
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # Compute width and height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Get detection confidence, rounded to 2 decimal places
            conf = math.ceil((box.conf * 100)) / 100

            # Get detected class index
            cls = int(box.cls[0])


            # Process only vehicles of interest: Person5
            if class_names[cls] in ['person']:
                # Draw styled bounding box around vehicle
                cvzone.cornerRect(frame, (x1, y1, w, h), l=30, t=5, colorR=(0, 0, 0), colorC=(200, 200, 200))

                # Prepare detection array [x1, y1, x2, y2, confidence]
                currentArray = np.array([x1, y1, x2, y2, conf])

                # Append detection to detections array
                detections = np.vstack((detections, currentArray))

    # Update tracker with current frame's detections
    resulttracker = tracker.update(detections)

    # Draw the counting line on frame
    cv2.line(frame, (line_1[0], line_1[1]), (line_1[2], line_1[3]), (255, 255, 255), 5)
    cv2.line(frame, (line_2[0], line_2[1]), (line_2[2], line_2[3]), (255, 255, 255), 5)
    # Iterate over tracked objects
    for result in resulttracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        w, h = x2 - x1, y2 - y1

        # Draw bounding box for tracked object
        cvzone.cornerRect(frame, (x1, y1, w, h), l=30, t=5, colorR=(0, 150, 0), colorC=(200, 200, 200))

        # Display tracking ID
        cvzone.putTextRect(frame, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=2, colorT=(0, 0, 0), colorB=(255, 255, 255), colorR=(255, 255, 255))

        # Calculate object's center point
        cx, cy = x1 + w // 2, y1 + h // 2

        # Draw center point as a circle
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), 2)

        # Check if object crosses the counting line region
        if line_1[0] < cx < line_1[2] and line_1[1] - 20 < cy < line_1[3] + 30:
            # If this object's ID hasn't been counted before, count it
            if total_countUp.count(Id) == 0:
                total_countUp.append(Id)
        if line_2[0] < cx < line_2[2] and line_2[1] - 20 < cy < line_2[3] + 30:
            # If this object's ID hasn't been counted before, count it
            if total_countDown.count(Id) == 0:
                total_countDown.append(Id)

        # Display total count on frame
        cvzone.putTextRect(frame, f'Total Count Up: {len(total_countUp)} Total Count Down: {len(total_countDown)}', (50, 50))

    # Display the processed frame
    cv2.imshow("Frame", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup: close OpenCV windows
cv2.destroyAllWindows()
