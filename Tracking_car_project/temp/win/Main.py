import cv2
import agent_app
import time


while True :
	f = agent_app.camera_fun()

	cv2.imshow('frame', f)

	time.sleep(3)

	#agent_app.do_move(0)



