from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0)  # webcam
cap = cv2.VideoCapture("../Videos/road.mp4")  # for videos

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

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    # getting bounding boxes for each result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding boxes
            # open CV
            x1, y1, x2, y2 = box.xyxy[0]  # get the first element
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # cv Zone
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9)

            # Confidence level
            conf = math.ceil((box.conf[0]*100))/100

            # Class Name
            cls = int(box.cls[0])

            # The formatting and display of text.
            # The max is for displaying the confidence level text on the frame goes up
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
