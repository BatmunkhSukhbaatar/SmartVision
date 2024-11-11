from gtts import gTTS
import numpy as np
import torch
import cv2
import os
import threading

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
objects = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'dog', 'cat', 
           'traffic light', 'stop sign', 'fire hydrant', 'pedestrian crossing', 
           'road sign', 'sidewalk', 'lane-left', 'lane-right']

colors = {
    'person': (0, 255, 0),
    'car': (0, 0, 255),
    'bus': (247, 247, 0),
    'traffic light': (0, 255, 255),
    'stop sign': (255, 0, 0),
    'fire hydrant': (255, 140, 0),
    'pedestrian crossing': (138, 43, 226),
    'road sign': (75, 0, 130),
    'sidewalk': (128, 128, 128),
    'lane-left': (255, 215, 0),
    'lane-right': (0, 191, 255)  
}

cap = cv2.VideoCapture("Smart.MP4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter('Out.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width, frame_height))

colors = {
    'person': (0, 255, 0),
    'car': (0, 0, 255),
    'bus': (247, 247, 0),
    'traffic light': (0, 255, 255),
    'stop sign': (255, 0, 0),
    'fire hydrant': (255, 140, 0)
}

def speak(text):
    tts = gTTS(text, lang='en')
    tts.save("output.mp3")
    os.system("start output.mp3")
    threading.Timer(5, lambda: os.remove("output.mp3")).start()

def estimate_distance(bbox_width, known_width=0.5, focal_length=800):
    distance = (known_width * focal_length) / bbox_width
    return distance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detected_objects = results.pandas().xyxy[0]

    for index, row in detected_objects.iterrows():
        obj_name = row['name']

        if obj_name in objects:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            bbox_width = x2 - x1
            
            distance = estimate_distance(bbox_width)
            
            if 0.25 <= distance <= 10:
                label = f"{obj_name}:{distance:.2f}"

                color = colors.get(obj_name, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                print(label)

    output_video.write(frame)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()
