import cv2
import numpy as np
import os
import pyttsx3
import time
import threading

# File paths
weights_path = r"C:\AI-powered Audio Able Insight to Support Blind\yolo\yolov8.weights"
config_path = r"C:\Projects\AI-powered Audio Able Insight to Support Blind\yolo\yolov8.cfg"
classes_file = r"C:\Projects\AI-powered Audio Able Insight to Support Blind\yolo\coco.names"

# Load class names
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize Text-to-Speech
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load YOLO
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Error: {weights_path} not found!")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Error: {config_path} not found!")
if not os.path.exists(classes_file):
    raise FileNotFoundError(f"Error: {classes_file} not found!")

net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Camera parameters (calibrate these values)
KNOWN_HEIGHT = 1.7  # Height in meters
FOCAL_LENGTH = 600   # Focal length in pixels (calibrate this)

# Start video capture using mobile phone camera
cap = cv2.VideoCapture('http://192.168.58.170:8080/video')  # Replace with your actual IP address and port
last_spoken_time = {}
distance_buffer = []

frame = None

def read_frame():
    global frame
    while True:
        ret, current_frame = cap.read()
        if ret:
            frame = cv2.resize(current_frame, (320, 240))  # Lower the resolution for speed

# Start frame reading in a separate thread
thread = threading.Thread(target=read_frame)
thread.start()

try:
    while True:
        if frame is None:
            continue  # Wait until a frame is captured

        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if h > 0:
                    distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / h
                    distance_cm = distance * 100
                    
                    distance_buffer.append(distance_cm)
                    if len(distance_buffer) > 5:
                        distance_buffer.pop(0)
                    smoothed_distance = np.mean(distance_buffer)

                    current_time = time.time()
                    if label not in last_spoken_time or (current_time - last_spoken_time[label] >= 10):
                        speak(f"The {label} is {smoothed_distance:.2f} centimeters ahead of you")
                        last_spoken_time[label] = current_time

        cv2.imshow('Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Streaming stopped by Ctrl+C")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released and windows closed.")
