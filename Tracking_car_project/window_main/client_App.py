#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from Tkinter import *
from socket import *      # Import necessary modules

HOST = '192.168.43.135'    # Server(Raspberry Pi) IP address
PORT = 22
BUFSIZ = 1024             # buffer size
ADDR = (HOST, PORT)

tcpCliSock = socket(AF_INET, SOCK_STREAM)   # Create a socket
tcpCliSock.connect(ADDR)                    # Connect with the server

def forward_fun():
  tcpCliSock.send(b'forward')

def backward_fun():
  tcpCliSock.send(b'backward')

def left_fun():
  tcpCliSock.send(b'left')

def right_fun():
  tcpCliSock.send(b'right')

def stop_fun():
  tcpCliSock.send(b'stop')

def home_fun():
  tcpCliSock.send(b'home')

# =============================================================================
# Exit the GUI program and close the network connection between the client 
# and server.
# =============================================================================
def quit_fun():
  #top.quit()
  tcpCliSock.send('stop')
  tcpCliSock.close()


def changeSpeed(ev=None):
  tmp = 'speed'
  global spd
  spd = 10
  data = tmp + str(spd)  # Change the integers into strings and combine them with the string 'speed'. 
  print('sendData = %s' % data)
  tcpCliSock.send(b'speed10')  # Send the speed data to the server(Raspberry Pi)

#label = Label(top, text='Speed:', fg='red')  # Create a label
#label.grid(row=6, column=0)                  # Label layout

#speed = Scale(top, from_=0, to=100, orient=HORIZONTAL, command=changeSpeed)  # Create a scale
#speed.set(50)
#speed.grid(row=6, column=1)

def do_move(action):
  if action == 0 :
    stop_fun()
  elif action == 1 :
    forward_fun()
  elif action == 2 :
    backward_fun()
  elif action == 3 :
    right_fun()
  elif action == 4 :
    left_fun()




