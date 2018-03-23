from Tkinter import *
from socket import *      # Import necessary modules
import os

top = Tk()   # Create a top window for show image stream
top.title('Raspberry Pi Smart Video Car Calibration')

HOST = '192.168.0.159'    # Server(Raspberry Pi) IP address
PORT = 21567
BUFSIZ = 1024             # buffer size
ADDR = (HOST, PORT)

tcpCliSock = socket(AF_INET, SOCK_STREAM)   # Create a socket
tcpCliSock.connect(ADDR)                    # Connect with the server



#### For Moving Send data ####
def run(event):
	global runbtn
	print 'motor ', runbtn
	if runbtn == 'Stop':
		tcpCliSock.send('motor_stop')
		runbtn = 'Run'
	elif runbtn == 'Run':
		tcpCliSock.send('motor_run')
		runbtn = 'Stop'



	

if __name__ == '__main__' :
	main()