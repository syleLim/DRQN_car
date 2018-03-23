import cv2
import picamera
import io
import numpy as np


### Way to cv2 in picamera (in raspberry pi)
class pi_camera :
    stream = io.BytesIO()
	#Get the picture (low resolution, so it should be quite fast)
    #Here you can also specify other parameters (e.g.:rotate the image)
    with picamera.PiCamera() as camera :
    	camera.resolution = (160, 160)
    	camera.capture(stream, format = 'jpeg')

	#Convert the picture into a numpy array
	buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

	#Now creates an OpenCV image
	image = cv2.imdecode(buff, 1)

    cv2.imwrite('result.jpg',image)

    def something(self) :

        ## Connected Server
        HOST = '192.168.0.159'    # Server(Raspberry Pi) IP address
        PORT = 21567
        BUFSIZ = 1024             # buffer size
        ADDR = (HOST, PORT)

        tcpCliSock = socket(AF_INET, SOCK_STREAM)   # Create a socket
        tcpCliSock.connect(ADDR)                    # Connect with the server