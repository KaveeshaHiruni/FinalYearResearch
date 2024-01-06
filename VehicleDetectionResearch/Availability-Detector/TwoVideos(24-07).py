import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


def resize_image(img):
    return cv2.resize(img, (640, 360))


# cap = cv2.VideoCapture(0)  # webcam
cap_road = cv2.VideoCapture("../Videos/road.mp4")  # for road video
cap_road2 = cv2.VideoCapture("../Videos/road2.mp4")  # for road2 video

cap_road.set(3, 640)
cap_road.set(4, 360)
cap_road2.set(3, 640)
cap_road2.set(4, 360)

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

mask = cv2.imread("mask360.png")

tracker_road = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker_road2 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [20, 100, 500, 100]

availability_road = True
availability_road2 = True
cumulative_availability = True

total_count_road = []
total_count_road2 = []

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

    detections_road = np.empty((0, 5))
    detections_road2 = np.empty((0, 5))

    for r in results_road:
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
    cv2.line(img_road2, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    front_empty_road = True
    front_empty_road2 = True

    for result in resultsTracker_road:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img_road, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if y2 > limits[1]:
            front_empty_road = False

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if id not in total_count_road:
                total_count_road.append(id)

    for result in resultsTracker_road2:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img_road2, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if y2 > limits[1]:
            front_empty_road2 = False

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if id not in total_count_road2:
                total_count_road2.append(id)

    availability_road = front_empty_road
    availability_road2 = front_empty_road2

    cumulative_availability = availability_road and availability_road2

    if availability_road:
        cv2.putText(img_road, "Available", (img_road.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
    else:
        cv2.putText(img_road, "Unavailable", (img_road.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

    if availability_road2:
        cv2.putText(img_road2, "Available", (img_road2.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
    else:
        cv2.putText(img_road2, "Unavailable", (img_road2.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2, cv2.LINE_AA)

    cv2.putText(img_road, f"Cumulative Availability: {'Available' if cumulative_availability else 'Unavailable'}",
                (50, img_road.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if cumulative_availability else (0, 0, 255), 2, cv2.LINE_AA)

    cvzone.putTextRect(img_road, f' Count: {len(total_count_road)}', (50, 50), scale=0.8, thickness=1)
    cvzone.putTextRect(img_road2, f' Count: {len(total_count_road2)}', (50, 50), scale=0.8, thickness=1)

    combined_img = np.hstack((img_road, img_road2))

    cv2.imshow("Combined Videos", combined_img)
    cv2.waitKey(1)

cap_road.release()
cap_road2.release()
cv2.destroyAllWindows()
