from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import time
import os

# cap = cv2.VideoCapture(0)  # webcam
cap = cv2.VideoCapture("../Videos/both.mp4")  # for videos

cap.set(3, 1280)
cap.set(4, 720)

# create YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# mask = cv2.imread("mask3.png")
mask = cv2.imread("maskBare.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [335, 100, 600, 100]
limits1 = [100, 490, 720, 490]
totalCount = []
vehicleCrossingTimes = {}
vehicleSpeeds = {}

# Initialize availability variable
availability = True

start_time = time.time()

# Define the folder where you want to save the screenshots (relative path to your project folder)
screenshot_folder = "screenshots"

# Create the folder if it doesn't exist
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)


# Define a function to capture a screenshot and save it to the specified folder
def capture_screenshot(img1, a1, b1, a2, b2, id, speed):
    # Define a region of interest (ROI) around the vehicle
    roi = img1[b1:b2, a1:a2]

    # Generate the full path for the screenshot with a unique filename
    screenshot_filename = os.path.join(screenshot_folder, f"vehicle_{id}_speed_{speed:.2f}_m_s.png")

    # Save the screenshot
    cv2.imwrite(screenshot_filename, roi)

    # Display a message when a screenshot is captured
    print(f"Screenshot captured for vehicle {id} with speed {speed:.2f} m/s")

    # Display the message at the bottom of the image
    message = f"Vehicle {id} captured with speed {speed:.2f} m/s"
    message_position = (10, img1.shape[0] - 30)  # Position at the bottom of the image
    cv2.putText(img1, message, message_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


# Define traffic light colors and initial state
traffic_light_colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]  # Red, Yellow, Green
traffic_light_state = 2  # Start with Red


# Function to draw a traffic light with a black rectangle
def draw_traffic_light_with_rectangle(traffic, state):
    light_radius = 25
    light_spacing = 35
    light_x = 50
    light_y = 50
    rectangle_width = 120
    rectangle_height = 320

    # Draw the black rectangle
    cv2.rectangle(traffic, (light_x - rectangle_width // 2, light_y - rectangle_height // 2),
                  (light_x + rectangle_width // 2, light_y + rectangle_height // 2), (0, 0, 0), cv2.FILLED)

    for i, color in enumerate(traffic_light_colors):
        if i == state:
            cv2.circle(traffic, (light_x, light_y + i * (light_spacing + light_radius)), light_radius,
                       color, cv2.FILLED)
        else:
            cv2.circle(traffic, (light_x, light_y + i * (light_spacing + light_radius)), light_radius, color, 3)


while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    front_empty = True  # Variable to track if front part of the road is empty

    # getting bounding boxes for each result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]  # get the first element
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence level
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # for the class
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                # scale=1, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))

        # for the id
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=2, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if the bounding box is in the front part of the road
        if y2 > limits[1]:
            front_empty = False
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 0))
            # cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
            # scale=2, thickness=3, offset=10)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # limits of x and y
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # Check if the vehicle crosses from L1 to L2
        if limits[1] < cy < limits1[1] and limits[0] < cx < limits[2]:
            if id not in vehicleCrossingTimes:
                vehicleCrossingTimes[id] = cv2.getTickCount()  # Store the current time

        # Check if the vehicle crosses L2 and calculate speed
        if id in vehicleCrossingTimes and (cy > limits1[1] or cx > limits1[2]):
            # Calculate time elapsed
            timeElapsed = (cv2.getTickCount() - vehicleCrossingTimes[id]) / cv2.getTickFrequency()
            speed = 100 / timeElapsed  # Calculate speed (distance / time)
            del vehicleCrossingTimes[id]  # Remove the vehicle from the dictionary
            vehicleSpeeds[id] = speed  # Store the speed for the vehicle
            speed_text = f"Speed: {speed:.2f} m/s"
            cvzone.putTextRect(img, speed_text, (max(0, x1), max(35, y1)),
                               scale=1, thickness=2, offset=10)

            # Check if the speed exceeds 40 m/s and capture a screenshot
            if speed > 40.0:
                capture_screenshot(img, x1, y1, x2, y2, id, speed)

        # Display the speed if the vehicle is present in the frame
        if id in vehicleSpeeds:
            speed = vehicleSpeeds[id]
            speed_text = f"Speed: {speed:.2f} m/s"
            cvzone.putTextRect(img, speed_text, (max(0, x1), max(35, y1)),
                               scale=1, thickness=2, offset=10)

    # Update availability based on front part of the road
    if not front_empty:
        availability = False  # Set availability to False
        cv2.putText(img, "Unavailable", (img.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
    else:
        availability = True  # Set availability to True
        cv2.putText(img, "Available", (img.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

    # Draw the text in the top-left corner
    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (150, 50), scale=1, thickness=1,colorT=(0, 0, 0))

    # Draw the traffic light with a black rectangle
    draw_traffic_light_with_rectangle(img, traffic_light_state)

    # Calculate and display timer
    current_time = time.time()
    elapsed_time = current_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    timer_text = f"Time: {minutes:02d}:{seconds:02d}"
    cv2.putText(img, timer_text, (50, img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Change the traffic light state every 5 seconds
    if int(elapsed_time) % 5 == 0:
        traffic_light_state = (traffic_light_state + 1) % len(traffic_light_colors)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
