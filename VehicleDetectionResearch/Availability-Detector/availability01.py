import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import time
import os


def resize_image(img):
    return cv2.resize(img, (640, 360))


# cap = cv2.VideoCapture(0)  # webcam
cap_road = cv2.VideoCapture("../Videos/both.mp4")  # for road video
cap_road2 = cv2.VideoCapture("../Videos/both.mp4")  # for road2 video

cap_road.set(3, 640)
cap_road.set(4, 360)
cap_road2.set(3, 640)
cap_road2.set(4, 360)

# create YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# create YOLO model
customModel = YOLO("emptyFilled.pt")

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

customClassNames = ['bare-roads', 'filled-roads']

mask = cv2.imread("mask360.png")

# Tracking
tracker_road = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker_road2 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# limit lines
# limits = [335, 150, 673, 150]
# limits1 = [100, 460, 720, 460]

# limits = [335, 150, 673, 150]
limits = [130, 80, 350, 80]
limits1 = [10, 300, 350, 300]

total_count_road = []
total_count_road2 = []

vehicleCrossingTimes = {}
vehicleCrossingTimes1 = {}

vehicleSpeeds = {}
vehicleSpeeds1 = {}

# Initialize availability variable for the two videos
availability_road = True
availability_road2 = True
cumulative_availability = True

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

    # Check if the ROI is empty
    if not roi.any():
        print(f"No vehicle in ROI for vehicle {id}")
        return

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
    light_radius = 20
    light_spacing = 30
    light_x = traffic.shape[1]-40
    light_y = 30
    rectangle_width = 90
    rectangle_height = 260

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
    success_road, img_road = cap_road.read()
    success_road2, img_road2 = cap_road2.read()

    if not success_road or not success_road2:
        break

    img_road = resize_image(img_road)
    img_road2 = resize_image(img_road2)

    imgRegion_road = cv2.bitwise_and(img_road, mask)
    imgRegion_road2 = cv2.bitwise_and(img_road2, mask)

    results_road = model(imgRegion_road, stream=True)
    results_road2 = model(imgRegion_road2, stream=True)
    customResults = customModel(imgRegion_road, stream=True)
    customResults2 = customModel(imgRegion_road2, stream=True)

    detections_road = np.empty((0, 5))
    detections_road2 = np.empty((0, 5))

    for r in customResults:
        boxes = r.boxes
        for box in boxes:
            # Confidence level
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Check if confidence level is above 0.9
            if conf > 0.9:
                # Bounding boxes
                # open CV
                x1, y1, x2, y2 = box.xyxy[0]  # get the first element
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # cv Zone
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img_road, (x1, y1, w, h), l=15)

                # Class Name
                cls = int(box.cls[0])

                # The formatting and display of text.
                # The max is for displaying the confidence level text on the frame goes up
                cvzone.putTextRect(img_road, f'{customClassNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1,
                                   thickness=1)

    for r in customResults2:
        boxes = r.boxes
        for box in boxes:
            # Confidence level
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Check if confidence level is above 0.9
            if conf > 0.9:
                # Bounding boxes
                # open CV
                x1, y1, x2, y2 = box.xyxy[0]  # get the first element
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # cv Zone
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img_road2, (x1, y1, w, h), l=15)

                # Class Name
                cls = int(box.cls[0])

                # The formatting and display of text.
                # The max is for displaying the confidence level text on the frame goes up
                cvzone.putTextRect(img_road2, f'{customClassNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1,
                                   thickness=1)

    detections = np.empty((0, 5))
    front_empty = True  # Variable to track if front part of the road is empty

    # getting bounding boxes for each result
    for r in results_road:
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
                detections_road = np.vstack((detections_road, currentArray))

    for r in results_road2:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections_road2 = np.vstack((detections_road2, currentArray))

    resultsTracker_road = tracker_road.update(detections_road)
    resultsTracker_road2 = tracker_road2.update(detections_road2)

    cv2.line(img_road, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img_road, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 5)
    cv2.line(img_road2, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img_road2, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 5)

    front_empty_road = True
    front_empty_road2 = True # Variable to track if front part of the road is empty

    for result in resultsTracker_road:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img_road, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))

        # for the id
        # cvzone.putTextRect(img_road, f'{int(id)}', (max(0, x1), max(35, y1)),
        # scale=1, thickness=2, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img_road, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if the bounding box is in the front part of the road
        if y2 > limits[1]:
            front_empty_road = False
            cvzone.cornerRect(img_road, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 0))
            # cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
            # scale=2, thickness=3, offset=10)
            cv2.circle(img_road, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            cv2.line(img_road, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # limits of x and y
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if total_count_road.count(id) == 0:
                total_count_road.append(id)
                cv2.line(img_road, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

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
            cvzone.putTextRect(img_road, speed_text, (max(0, x1), max(35, y1)),
                               scale=1, thickness=2, offset=10)

            # Check if the speed exceeds 40 m/s and capture a screenshot
            if speed > 10.0:
                capture_screenshot(img_road, x1, y1, x2, y2, id, speed)

        # Display the speed if the vehicle is present in the frame
        if id in vehicleSpeeds:
            speed = vehicleSpeeds[id]
            speed_text = f"Speed: {speed:.2f} m/s"
            cvzone.putTextRect(img_road, speed_text, (max(0, x1), max(35, y1)),
                               scale=1, thickness=2, offset=10)

    for result in resultsTracker_road2:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img_road2, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 0))

        # for the id
        # cvzone.putTextRect(img_road2, f'{int(id)}', (max(0, x1), max(35, y1)),
        # scale=1, thickness=2, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img_road2, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if the bounding box is in the front part of the road
        if y2 > limits[1]:
            front_empty_road2 = False
            cvzone.cornerRect(img_road2, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 0))
            # cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
            # scale=2, thickness=3, offset=10)
            cv2.circle(img_road2, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            cv2.line(img_road2, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # limits of x and y
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if total_count_road2.count(id) == 0:
                total_count_road2.append(id)
                cv2.line(img_road2, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # Check if the vehicle crosses from L1 to L2
        if limits[1] < cy < limits1[1] and limits[0] < cx < limits[2]:
            if id not in vehicleCrossingTimes1:
                vehicleCrossingTimes1[id] = cv2.getTickCount()  # Store the current time

        # Check if the vehicle crosses L2 and calculate speed
        if id in vehicleCrossingTimes1 and (cy > limits1[1] or cx > limits1[2]):
            # Calculate time elapsed
            timeElapsed = (cv2.getTickCount() - vehicleCrossingTimes1[id]) / cv2.getTickFrequency()
            speed = 100 / timeElapsed  # Calculate speed (distance / time)
            del vehicleCrossingTimes1[id]  # Remove the vehicle from the dictionary
            vehicleSpeeds1[id] = speed  # Store the speed for the vehicle
            speed_text = f"Speed: {speed:.2f} m/s"
            cvzone.putTextRect(img_road2, speed_text, (max(0, x1), max(35, y1)),
                               scale=1, thickness=2, offset=10)

            # Check if the speed exceeds 40 m/s and capture a screenshot
            if speed > 10.0:
                capture_screenshot(img_road2, x1, y1, x2, y2, id, speed)

        # Display the speed if the vehicle is present in the frame
        if id in vehicleSpeeds1:
            speed = vehicleSpeeds1[id]
            speed_text = f"Speed: {speed:.2f} m/s"
            cvzone.putTextRect(img_road2, speed_text, (max(0, x1), max(35, y1)),
                               scale=1, thickness=2, offset=10)

    availability_road = front_empty_road
    availability_road2 = front_empty_road2

    cumulative_availability = availability_road and availability_road2

    # Update availability based on front part of the road
    if availability_road:
        cv2.putText(img_road, "Available", (img_road.shape[1] - 230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
    else:
        cv2.putText(img_road, "Unavailable", (img_road.shape[1] - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    if availability_road2:
        cv2.putText(img_road2, "Available", (img_road2.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
    else:
        cv2.putText(img_road2, "Unavailable", (img_road2.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)

    # Adjusting the position of the text as needed
    availability_position = (320, img_road.shape[0] - 340)
    font_size = 0.5  # Adjust the font size as needed
    cv2.putText(img_road, f"Cumulative Result: {'Available' if cumulative_availability else 'Unavailable'}",
                availability_position, cv2.FONT_HERSHEY_SIMPLEX, font_size,
                (0, 255, 0) if cumulative_availability else (255, 255, 255), 1, cv2.LINE_AA)

    # cvzone.putTextRect(img_road, f' Count: {len(total_count_road)}', (20, (img_road.shape[0] // 2) - 10), scale=0.8,
    # thickness=1)
    # cvzone.putTextRect(img_road2, f' Count: {len(total_count_road2)}', (20, (img_road.shape[0] // 2) - 10), scale=0.8,
    # thickness=1)
    # Draw the traffic light with a black rectangle
    draw_traffic_light_with_rectangle(img_road, traffic_light_state)

    # Calculate and display timer
    current_time = time.time()
    elapsed_time = current_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    timer_text = f"Time: {minutes:02d}:{seconds:02d}"
    # Display timer text on the right bottom of both videos
    timer_position = (img_road.shape[1] - 630, img_road.shape[0] - 200)  # Adjust the position as needed
    cv2.putText(img_road, timer_text, timer_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_road2, timer_text, timer_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    combined_img = np.hstack((img_road, img_road2))

    cv2.imshow("Combined Videos", combined_img)
    cv2.waitKey(1)

cap_road.release()
cap_road2.release()
cv2.destroyAllWindows()
