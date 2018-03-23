
import cv2
import numpy as np



def Video_load(cap) : 
	cap = cv2.VideoCapture(1) #0 = read_Cemra / file_name = read_file

	return cap

## Video Load and Change
def Video_resize(cap) : 
	width = cap.get(3)
	height = cap.get(4)

	cap.set(3, 500)  #width = 500
	cap.set(4, 600)  #height = 500

	while(True) :
		is_img_in, frame = cap.read() #is_video_in == ret, if video is in sucesss, ret is True / when end vdeo, return False

		gray = cv2.cvtColor(frame, cv.COLOR_BGR2GRAY) # GrayScaling
		cv2.imshow('webcam', frame)

		if cv2.waitKey(1)&0xFF == ord('q'):
			break							#input q, finish

	cap.release()
	cv2.destroyAllWindows()

## Video Save other file
def Vedio_save (cap) : 
	fourcc = cv2.VideoWroter_fourcc(*'DIVX')  # cv2.VideoWriter(output, Codec, frame, size(=width, height))
	out = cv2.VideoWriter('output.avi', fourcc, 25.0, (500, 500))

	while( cap.isOpen()) : # video read check
		ret, frame = cap.read()

		if ret : 

			frame = cv2.flip(frame, 0)

			out.write(frame)

			cv2.imshow('frame', frame)

			if cv2.waitKey(0) & 0xFF == ord('q'):
				break
		else : 
			break

	cap.release()
	out.release()
	cv2.destroyAllWindows()