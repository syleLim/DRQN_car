import cv2

cap = cv2.VideoCapture(0)

def image_data () :
	ret, frame = cap.read()

	if ret is None :
		print('camera _ error')
		return

	return frame


