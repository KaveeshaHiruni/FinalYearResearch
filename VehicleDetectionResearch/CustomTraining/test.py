from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0)  # webcam
cap = cv2.VideoCapture("../Videos/bareRoad.mp4")  # for videos

cap.set(3, 1280)
cap.set(4, 720)

# create YOLO model
model = YOLO("bareRoadTest.pt.pt")

classNames = ['bare-roads']

mask = cv2.imread("maskEmptyFilled.png")

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    # getting bounding boxes for each result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Confidence level
            conf = box.conf[0]

            # Check if confidence is above 0.9
            if conf > 0.9:
                x1, y1, x2, y2 = box.xyxy[0]  # get the first element
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Bounding boxes
                # open CV
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # cv Zone
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)

                # Class Name
                cls = int(box.cls[0])

                # The formatting and display of text.
                # The max is for displaying the confidence level text on the frame goes up
                cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (max(0, x1), max(35, y1)), scale=0.6,
                                   thickness=1, offset=3)

    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
