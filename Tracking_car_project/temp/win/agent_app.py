#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from Tkinter import *
from socket import *      # Import necessary modules
import numpy as np
import time
import sys
import struct
from io import StringIO

try :
  import cPickle as pickle
except ImportError :
  import pickle

HOST = '192.168.43.20'    # Server(Raspberry Pi) IP address
PORT = 21565
BUFSIZ = 1024            # buffer size
ADDR = (HOST, PORT)

tcpCliSock = socket(AF_INET, SOCK_STREAM)   # Create a socket
tcpCliSock.connect(ADDR)                    # Connect with the server

def forward_fun():
  tcpCliSock.send(bytes('forward', encoding='utf-8'))

def backward_fun():
  tcpCliSock.send(bytes('backward', encoding='utf-8'))

def left_fun():
  tcpCliSock.send(bytes('left', encoding='utf-8'))

def right_fun():
  tcpCliSock.send(bytes('right', encoding='utf-8'))

def stop_fun():
  tcpCliSock.send(bytes('stop', encoding='utf-8'))

def home_fun():
  tcpCliSock.send(bytes('home', encoding='utf-8'))

def camera_fun():
  tcpCliSock.send(bytes('get_img', encoding='utf-8'))



  b_frame = tcpCliSock.recv(BUFSIZ)
  #how to find it
  
  #frame = np.frombuffer(b_frame, dtype = np.int32, offset = 16)
  #b_frame = tcpCliSock.recv(BUFSIZ)

  #print(b_frame)
  count = 0
  b_img = b''
  while True :
    b_frame = tcpCliSock.recv(BUFSIZ)
    b_img += b_frame
    

    count +=1
    if count == 110 : break
    print(count)

    

  print('-----')

  frame = pickle.loads(b_img)

  print('end pic')

  # #frame = np.load(StringIO(data.decode('utf-8')))['frame']

  #print(frame)

  print(frame.shape)
  #print(frame)

  return frame

# =============================================================================
# Exit the GUI program and close the network connection between the client 
# and server.
# =============================================================================
def quit_fun():
  top.quit()
  tcpCliSock.send('stop')
  tcpCliSock.close()

spd = 50

def changeSpeed(ev=None):
  tmp = 'speed'
  global spd
  spd = speed.get()
  data = tmp + str(spd)  # Change the integers into strings and combine them with the string 'speed'. 
  print('sendData = %s' % data)
  tcpCliSock.send(data)  # Send the speed data to the server(Raspberry Pi)

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