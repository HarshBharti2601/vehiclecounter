from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import time
import numpy as np
import threading

app = Flask(__name__)
socketio = SocketIO(app)

PIXEL_TO_METERS = 0.1
FPS = 30
MIN_WIDTH_RECT = 80
MIN_HEIGHT_RECT = 80
COUNT_LINE_POSITION = 550
COUNT_LINE_POSITION_EXIT = 800
OFFSET = 6

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        objectID = str(len(self.objects) + 1)
        self.objects[objectID] = (centroid, None)
        self.disappeared[objectID] = 0

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [self.objects[objID][0] for objID in objectIDs]

            D = np.zeros((len(objectIDs), len(inputCentroids)))

            for i in range(len(objectIDs)):
                for j in range(len(inputCentroids)):
                    D[i, j] = np.linalg.norm(objectCentroids[i] - inputCentroids[j])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = (inputCentroids[col], objectCentroids[row])
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects
    
ct = CentroidTracker()    

vehicle_counter = 0  # Initialize vehicle counter

def process_frame(frame):
    global vehicle_counter
    
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    img_sub = algo.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    dilatadata = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, np.ones((5, 5)))
    dilatadata = cv2.morphologyEx(dilatadata, cv2.MORPH_CLOSE, np.ones((5, 5)))

    contours, _ = cv2.findContours(dilatadata, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        validate_counter = (w >= MIN_WIDTH_RECT) and (h >= MIN_HEIGHT_RECT)
        if not validate_counter:
            continue
        rects.append((x, y, x + w, y + h))

    objects = ct.update(rects)
    
    vehicles_between = 0
    for objectID, (centroid, _) in objects.items():
        if centroid[1] < (COUNT_LINE_POSITION_EXIT + OFFSET) and centroid[1] > (COUNT_LINE_POSITION - OFFSET):
            vehicles_between += 1

    global vehicle_counter  # Use the global variable

    JAM_THRESHOLD = 0
    jam = vehicles_between >= JAM_THRESHOLD

    for objectID, (centroid, prevCentroid) in objects.items():
        if centroid[1] < (COUNT_LINE_POSITION + OFFSET) and centroid[1] > (COUNT_LINE_POSITION - OFFSET):
            cv2.line(frame, (25, COUNT_LINE_POSITION), (1200, COUNT_LINE_POSITION), (0, 127, 255), 3)
            vehicle_counter = len(objects)  # Update vehicle counter

        if centroid[1] < (COUNT_LINE_POSITION_EXIT + OFFSET) and centroid[1] > (COUNT_LINE_POSITION_EXIT - OFFSET):
            cv2.line(frame, (25, COUNT_LINE_POSITION_EXIT), (1200, COUNT_LINE_POSITION_EXIT), (0, 127, 255), 3)
            vehicle_counter = len(objects)  # Update vehicle counter
    

    print("Vehicle counter:",vehicle_counter)
    # Emit count to the connected clients
    socketio.emit('update_vehicle_count', {'vehicle_count': vehicle_counter})

    # cv2.imshow('Frame', frame) # Removed to not show the frame
    # cv2.waitKey(1) # Not needed since we are not displaying the frame

def process_video(cap):
    global stop_thread  # Define a flag to control the thread's execution
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_thread:  # Check if the video ended or stop flag is True
            break

        process_frame(frame)

    cap.release()

@app.route('/vehicle_counter', methods=['GET'])
def get_vehicle_counter():
    global vehicle_counter
    return jsonify({'vehicle_counter': vehicle_counter})

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route('/stop_video', methods=['GET'])  # Endpoint to stop the video processing
def stop_video():
    global stop_thread
    stop_thread = True  # Set the flag to True to stop the thread
    return jsonify({'status': 'Video processing stopped'})

if __name__ == '__main__':
    cap = cv2.VideoCapture('video.mp4')
    algo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    stop_thread = False  #
    video_thread = threading.Thread(target=process_video, args=(cap,))
    video_thread.start()
    socketio.run(app, debug=True)
