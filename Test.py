from gtts import gTTS
import numpy as np
import torch
import cv2
import os
import threading

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
objects = ['person','car']
cap = cv2.VideoCapture("Renew.mp4")

def speak(text):
    tts = gTTS(text, lang='en')
    tts.save("output.mp3")
    os.system("start output.mp3")
    threading.Timer(5, lambda: os.remove("output.mp3")).start()

def estimate_distance(bbox_width, known_width=0.5, focal_length=800):
    """
    Зайг тооцоолох функц. bbox_width нь bounding box-ын өргөн, 
    known_width нь тухайн объектыг мэдэгдэхүйц бодит өргөн (м), 
    focal_length нь камерын фокал урт (px).
    """
    distance = (known_width * focal_length) / bbox_width
    return distance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Дүрсийг YOLO загвар руу дамжуулах
    results = model(frame)

    # Илэрсэн объектуудыг хадгалах
    detected_objects = results.pandas().xyxy[0]

    for index, row in detected_objects.iterrows():
        obj_name = row['name']

        if obj_name in objects:  # Зөвхөн сонирхож буй объектыг авах
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            bbox_width = x2 - x1
            
            # Зайг тооцоолох
            distance = estimate_distance(bbox_width)
            
            # Зөвхөн 5-25 метрийн зайд буй объектуудыг харуулах
            if 0.25 <= distance <= 10:
                label = f"{obj_name.capitalize()}:{distance:.2f}"
                
                # Объектын талбайг хүрээлэх болон мэдээлэл бичих
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                print(label)

    # Дүрсийг харуулах
    cv2.imshow('Object Detection', frame)

    # 'q' товчийг дарж гаргах
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Камерыг чөлөөлөх
cap.release()
cv2.destroyAllWindows()
