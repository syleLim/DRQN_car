#!/usr/bin/env python
import RPi.GPIO as GPIO
import video_dir
import car_dir
import motor
from socket import *
from time import ctime          # Import necessary modules   
import numpy as np

ctrl_cmd = ['forward', 'backward', 'left', 'right', 'stop', 'read cpu_temp', 'home', 'distance', 'x+', 'x-', 'y+', 'y-', 'xy_home']

busnum = 1          # Edit busnum to 0, if you uses Raspberry Pi 1 or 0

HOST = ''           # The variable of HOST is null, so the function bind( ) can be bound to all valid addresses.
PORT = 21565
BUFSIZ = 1024       # Size of the buffer
ADDR = (HOST, PORT)

tcpSerSock = socket(AF_INET, SOCK_STREAM)    # Create a socket.
tcpSerSock.bind(ADDR)    # Bind the IP address and port number of the server. 
tcpSerSock.listen(5)     # The parameter of listen() defines the number of connections permitted at one time. Once the 
                         # connections are full, others will be rejected. 

video_dir.setup(busnum=busnum)
car_dir.setup(busnum=busnum)
motor.setup(busnum=busnum)     # Initialize the Raspberry Pi GPIO connected to the DC motor. 
video_dir.home_x_y()
car_dir.home()

while True:
	print ('Waiting for connection...')
	# Waiting for connection. Once receiving a connection, the function accept() returns a separate 
	# client socket for the subsequent communication. By default, the function accept() is a blocking 
	# one, which means it is suspended before the connection comes.
	tcpCliSock, addr = tcpSerSock.accept() 
	print ('...connected from :'+ str(addr))     # Print the IP address of the client connected with the server.

	while True:
		data = ''
		data = tcpCliSock.recv(BUFSIZ)    # Receive data sent from the client. 
		# Analyze the command received and control the car accordingly.
		if not data:
			break

		print('data : ' + str(data))

		elif data == ctrl_cmd[0]:
			
			motor.forward()
		elif data == ctrl_cmd[1]:
			
			motor.backward()
		elif data == ctrl_cmd[2]:
			
			car_dir.turn_left()
		elif data == ctrl_cmd[3]:
			
			car_dir.turn_right()
		elif data == ctrl_cmd[6]:
			
			car_dir.home()
		elif data == ctrl_cmd[4]:
			
			motor.ctrl(0)
		elif data == ctrl_cmd[5]:
			
			temp = cpu_temp.read()
			tcpCliSock.send('[%s] %0.2f' % (ctime(), temp))
		elif data == ctrl_cmd[8]:
			
			video_dir.move_increase_x()
		elif data == ctrl_cmd[9]:
			
			video_dir.move_decrease_x()
		elif data == ctrl_cmd[10]:
			
			video_dir.move_increase_y()
		elif data == ctrl_cmd[11]:
			
			video_dir.move_decrease_y()
		elif data == ctrl_cmd[12]:
			
			video_dir.home_x_y()
		elif data[0:5] == 'speed':     # Change the speed
			
			numLen = len(data) - len('speed')
			if numLen == 1 or numLen == 2 or numLen == 3:
				tmp = data[-numLen:]
			
				spd = int(tmp)
			
				if spd < 24:
					spd = 24
				motor.setSpeed(spd)
		elif data[0:5] == 'turn=':	#Turning Angle
			
			angle = data.split('=')[1]
			try:
				angle = int(angle)
				car_dir.turn(angle)
			except:
				print('Error: angle ='+ str(angle))
		elif data[0:8] == 'forward=':
			
			spd = data[8:]
			try:
				spd = int(spd)
				motor.forward(spd)
			except:
				print('Error speed ='+str(spd))
                elif data[0:9] == 'backward=':
                        
                        spd = data.split('=')[1]
			try:
				spd = int(spd)
	                        motor.backward(spd)
			except:
				print('ERROR, speed ='+str(spd))

		else:
			print ('Command Error! Cannot recognize command: ' + str(data))

tcpSerSock.close()


