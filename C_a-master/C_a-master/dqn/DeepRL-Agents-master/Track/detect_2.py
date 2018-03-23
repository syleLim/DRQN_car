from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
#import argparse
import imutils
import time
import cv2

import client_App

col = -1
width = -1
row = -1
height = -1
frame = None
frame2 = None
inputmode = False
rectangle = False
trackWindow = None
roi_hist = None
roi = None

obstacle_points = []
target_point = None
obstacle_box_color = (0, 0, 255)
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]

if tracker_type == 'BOOSTING':
	tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
	tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
	tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
	tracker = cv2.TrackerTLD_create()
else :
	tracker = cv2.TrackerMedianFlow_create()


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt','MobileNetSSD_deploy.caffemodel')
 
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")


def onMouse(event, x, y, flags, param) :
	global frame, inputmode, rectangle, col, row, frame2, width, height, trackWindow, tracker, roi

	if inputmode : 
		if event == cv2.EVENT_LBUTTONDOWN :
			trackWindow = None
			#print('DOWN')
			rectangle = True
			col, row = x, y
		elif event == cv2.EVENT_MOUSEMOVE :
			#print('MOVE')
			if rectangle :
				#print('Move - rec+true')
				frame = frame2.copy()
				cv2.rectangle(frame, (col, row), (x, y), (0, 255, 0), 2)
				cv2.imshow('frame', frame)

		elif event == cv2.EVENT_LBUTTONUP :
			#print('UP')
			inputmode = False
			rectangle = False
			cv2.rectangle(frame, (col, row), (x, y), (0, 255, 0), 2)
			height, width = abs(row - y), abs(col - x)
			trackWindow = (col, row, width, height)
			roi = frame[row : row + height, col : col+width]
			ok = tracker.init(frame, trackWindow)
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
			cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
	
	return

cap = VideoStream('http://192.168.137.2:8080/?action=stream').start()
time.sleep(2.0)
fps = FPS().start()

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', onMouse, param = (frame, frame2))


# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = cap.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

		# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
 
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > 0.2:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
 
			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	


	if trackWindow is not None :
		ok, trackWindow = tracker.update(frame)
		if ok :
			x, y, w, h = trackWindow
			x, y, w, h = int(x), int(y), int(w), int(h)
			#target_point = {'row' : int((2*y+h)/2), 'col' : int((2*x+w)/2)}
			cv2.rectangle(frame, (x, y), (x+w, y+w), (0, 255, 0), 3)
			is_game_start = True
		else : 
			cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

	show_frame = cv2.resize(frame, None, fx = 2, fy = 2)

	cv2.imshow('frame', show_frame)

	key = cv2.waitKey(60) & 0xFF

	

	if key == ord('i') :
		print('select target')
		inputmode = True
		frame2 = frame.copy()

		while inputmode :
			cv2.imshow('frame', frame)
			cv2.waitKey(0)
	
	client_App.forward_fun()

	fps.update() ### Idont know where it locatied

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
cap.stop()