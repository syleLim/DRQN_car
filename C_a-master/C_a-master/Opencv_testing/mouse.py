import numpy as np
import cv2
import video_camera

col, width, row, height = -1, -1, -1, -1

frame = None
frame2 = None
inputmode = False
rectangle = False
trackWindow = None
roi_hist = None

def onMouse(event, x, y, flags, param) :
	global col, width, row, height, frame, frame2, inputmode
	global rectangle, roi_hist, trackWindow

	if inputmode : 
		if event == cv2.EVENT_LBUTTONDOWN :
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
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
			cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
	
	return

def mean_Shift():
	global frame, frame2, inputmode, trackWindow, roi_hist

	cap = cv2.VideoCapture(0)

	ret, frame = cap.read()
	width = cap.get(3)
	height = cap.get(4)

	cv2.namedWindow('frame')
	cv2.setMouseCallback('frame', onMouse, param= (frame, frame2))

	termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

	while True :
		ret, frame = cap.read()
				
		if not ret : break

#		width = cap.get(3)
#		height = cap.get(4)

		if trackWindow is not None :
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
			## Maybe this window is track size
			ret, trackWindow = cv2.meanShift(dst, trackWindow, termination)

			x, y, w, h = trackWindow
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

			cv2.line(frame, (int(width/2), int(height)), (int((2*x+w)/2), int((2*y+h)/2)), (0, 255, 0), 2)

		cv2.imshow('frame', frame)

		key = cv2.waitKey(60) & 0xFF

		if key == 27 : break

		if key == ord('i') :
			print('Select for CamShift')
			inputmode = True
			frame2 = frame.copy()

			while inputmode :
				cv2.imshow('frame', frame)
				cv2.waitKey(0)
	cap.release()
	cv2.destroyAllWindows()

def cam_Shift() :
	global frame, frame2, inputmode, trackWindow, roi_hist, out

	cap = cv2.VideoCapture(0)

	ret, frame = cap.read()

	cv2.namedWindow('frame')
	cv2.setMouseCallback('frame', onMouse, param = (frame, frame2))

	termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

	while True :
		ret, frame = cap.read()

		if not ret : break

		width = cap.get(3)
		height = cap.get(4)

		if trackWindow is not None :
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
			## Maybe this window is track size
			ret, trackWindow = cv2.CamShift(dst, trackWindow, termination)

			pts = cv2.boxPoints(ret)
			pts = np.int0(pts)
			cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
			
			cv2.line(frame, (int(width/2), int(height)), (int((pts[0][0]+pts[2][0])/2), int((pts[0][1]+pts[2][1])/2)), (0, 255, 0), 2)

		cv2.imshow('frame', frame)

		key = cv2.waitKey(60) & 0xFF

		if key == 27 : break

		if key == ord('i') :
			print('Select for CamShift')
			inputmode = True
			frame2 = frame.copy()

			while inputmode :
				cv2.imshow('frame', frame)
				cv2.waitKey(0)

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__' :
	cam_Shift()
