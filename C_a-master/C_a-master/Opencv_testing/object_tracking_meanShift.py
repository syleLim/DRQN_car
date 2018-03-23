import numpy as np
import cv2

tracking_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']

def Tracking_type (cap, tracking_type) : 
	
	if tracking_type == 'BOOSTING' :
		tracker = cv2.TrackerBoosting_create()
	elif tracking_type == 'MIL' :
		tracker = cv2.TrackerMIL_create()
	elif tracking_type == 'KCF' :
		tracker = cv2.TrackerKCF_create()
	elif tracking_type == 'TLD' :
		tracker = cv2.TrackerTLD_create()
	elif tracking_type == 'MEDIANFLOW' :
		tracker = cv2.TrackerMedianFlow_create()
	elif tracking_type == 'GOTURN' :
		tracker = cv2.TrackerGOTURN_create()

	if not cap.isOpened() :
		print("video isnt open")
		sys.exit()

	ret, frame = cap.read()

	if not ret :
		print("video cannot be read")
		sys.exit()

	detect_box = (0, 0, 100, 100)#Set initial detect_box
	
	############################################
	detect_box = cv2.selectROI(frame, False) #??
	#############################################

	ret = tracker.init(frame, detect_box)

	while True :
		#read frame again
		ret, frame = cap.read()

		if not ret :
			break

		timer = cv2.GetTickCout()

		ret, detect_box = tracker.update(frame)

		#################################################
		fps = cv2.GetTickFrequency() / (cv2.GetTickCout() - timer) #??
		#################################################

		if ret :
			p1 = (int(detect_box[0]), int(detect_box[1]))
			p2 = (int(detect_box[0]+detect_box[2]), int(detect_box[1] + detect_box[3]))
			cv2.rectangle(frame, p1, p2 (255, 0, 0), 2, 1)
		else :
			cv2.putText(frame, "Tracking Failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

		cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
	
		# Display FPS on frame
		cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
		# Display result
		cv2.imshow("Tracking", frame)
 
		# Exit if ESC pressed
		k = cv2.waitKey(1) & 0xff
		if k == 'q' : break

def Drone_tracking_ex (cap) :
	ret, frame = cap.read()

	#Set Roi
	c, r, w, h = 300, 300, 70, 70
	track_window = (c, r, w, h)

	#mask / histogram be made
	roi = frame[r : r+h, c: c+w]
	hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))

	roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
	cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
	term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)

	while True :
		ret, frame = cap.read()

		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

		ret, track_window = cv2.meanShift(dst, track_window, term_crit)

		x, y, w, h = track_window
		cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
		cv2.putText(frame, 'Tracked', (x-25, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.CV_AA)

		cv2.imshow('Tracking', frame)

		if cv2.waitKey(1) & 0xFF == ord('q') :
			break

	cap.release()
	cv2.destoryAllWindows()

if __name__ == "__main__" :
	cap = cv2.VideoCapture(0)

	Drone_tracking_ex(cap)

