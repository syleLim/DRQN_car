import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time

#from helper import *

from Sim import Sim as sim
from Sim import r_Sim as r_sim

class Qnetwork():
	def __init__(self,h_size,rnn_cell,myScope):
		#The network recieves a frame from the game, flattened into an array.
		#It then resizes it and processes it through four convolutional layers.
		self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32)
		self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
		self.conv1 = slim.convolution2d( \
			inputs=self.imageIn,num_outputs=32,\
			kernel_size=[8,8],stride=[4,4],padding='VALID', \
			biases_initializer=None,scope=myScope+'_conv1')
		self.conv2 = slim.convolution2d( \
			inputs=self.conv1,num_outputs=64,\
			kernel_size=[4,4],stride=[2,2],padding='VALID', \
			biases_initializer=None,scope=myScope+'_conv2')
		self.conv3 = slim.convolution2d( \
			inputs=self.conv2,num_outputs=64,\
			kernel_size=[3,3],stride=[1,1],padding='VALID', \
			biases_initializer=None,scope=myScope+'_conv3')
		self.conv4 = slim.convolution2d( \
			inputs=self.conv3,num_outputs=h_size,\
			kernel_size=[7,7],stride=[1,1],padding='VALID', \
			biases_initializer=None,scope=myScope+'_conv4')
		
		self.trainLength = tf.placeholder(dtype=tf.int32)
		#We take the output from the final convolutional layer and send it to a recurrent layer.
		#The input must be reshaped into [batch x trace x units] for rnn processing, 
		#and then returned to [batch x units] when sent through the upper levles.
		self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
		self.convFlat = tf.reshape(slim.flatten(self.conv4),[self.batch_size,self.trainLength,h_size])
		self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
		self.rnn,self.rnn_state = tf.nn.dynamic_rnn(\
				inputs=self.convFlat,cell=rnn_cell,dtype=tf.float32,initial_state=self.state_in,scope=myScope+'_rnn')
		self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
		#The output from the recurrent player is then split into separate Value and Advantage streams
		self.streamA,self.streamV = tf.split(self.rnn,2,1)
		self.AW = tf.Variable(tf.random_normal([h_size//2,5]))
		self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
		self.Advantage = tf.matmul(self.streamA,self.AW)
		self.Value = tf.matmul(self.streamV,self.VW)
		
		self.salience = tf.gradients(self.Advantage,self.imageIn)
		#Then combine them together to get our final Q-values.
		self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
		self.predict = tf.argmax(self.Qout,1)
		
		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
		self.actions_onehot = tf.one_hot(self.actions,5,dtype=tf.float32)
		
		self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
		
		self.td_error = tf.square(self.targetQ - self.Q)
		
		#In order to only propogate accurate gradients through the network, we will mask the first
		#half of the losses for each trace as per Lample & Chatlot 2016
		self.maskA = tf.zeros([self.batch_size,self.trainLength//2])
		self.maskB = tf.ones([self.batch_size,self.trainLength//2])
		self.mask = tf.concat([self.maskA,self.maskB],1)
		self.mask = tf.reshape(self.mask,[-1])
		self.loss = tf.reduce_mean(self.td_error * self.mask)
		
		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
	def __init__(self, buffer_size = 1000):
		self.buffer = []
		self.buffer_size = buffer_size
	
	def add(self,experience):
		if len(self.buffer) + 1 >= self.buffer_size:
			self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
		self.buffer.append(experience)
			
	def sample(self,batch_size,trace_length):
		sampled_episodes = random.sample(self.buffer,batch_size)
		sampledTraces = []
		for episode in sampled_episodes:
			point = np.random.randint(0,len(episode)+1-trace_length)
			sampledTraces.append(episode[point:point+trace_length])
		sampledTraces = np.array(sampledTraces)
		return np.reshape(sampledTraces,[batch_size*trace_length,5])

class Agent :
	def __init__(self) :
		#Setting the training parameters
		self.batch_size = 4 #How many experience traces to use for each training step.
		self.trace_length = 8 #How long each experience trace will be when training
		self.update_freq = 5 #How often to perform a training step.
		self.y = .99 #Discount factor on the target Q-values
		self.startE = 1 #Starting chance of random action
		self.endE = 0.1 #Final chance of random action
		self.anneling_steps = 10000 #How many steps of training to reduce startE to endE.
		self.num_episodes = 10000 #How many episodes of game environment to train network with.
		self.pre_train_steps = 10000 #How many steps of random actions before training begins.
		self.load_model = True #Whether to load a saved model.
		self.path = "./drqn" #The path to save our model to.
		self.h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
		self.max_epLength = 300 #The max allowed length of our episode.
		self.time_per_step = 1 #Length of each step used in gif creation
		self.summaryLength = 100 #Number of epidoes to periodically save for analysis
		self.tau = 0.001

		# for Tracking
		self.cap = None
		self.col = -1
		self.width = -1
		self.row = -1
		self.height = -1
		self.frame = None
		self.frame2 = None
		self.inputmode = False
		self.rectangle = False
		self.trackWindow = None
		self.roi_hist= None
		self.roi = None
		self.caffe_model_path = './MobileNetSSD_deploy.caffemodel'
		self.prorotxt_path = './MobileNetSSD_deploy.prototxt.txt'
		self.net = None
		self.obstacle_points = []
		self.target_point = None
		self.obstacle_box_color = (0, 0, 255)
		self.tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
		self.tracker_type = self.tracker_types[1]

		if self.tracker_type == 'BOOSTING':
			self.tracker = cv2.TrackerBoosting_create()
		if self.tracker_type == 'MIL':
			self.tracker = cv2.TrackerMIL_create()
		if self.tracker_type == 'KCF':
			self.tracker = cv2.TrackerKCF_create()
		if self.tracker_type == 'TLD':
			self.tracker = cv2.TrackerTLD_create()
		else :
			self.tracker = cv2.TrackerMedianFlow_create()

		self.game = sim(200, True)

	def processState(self, states) :
		return np.reshape(states,[21168])

	def updateTargetGraph(self, tfVars, tau) :
		total_vars = len(tfVars)
		op_holder = []
		for idx,var in enumerate(tfVars[0:total_vars//2]):
			op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
		return op_holder

	def updateTarget(self, op_holder, sess) :
		for op in op_holder:
			sess.run(op)

	def Check_path(path) :
		if not os.path(path) :
			os.makedirs(path)

	# def saveToCenter(self, i, rList, bufferArray, summaryLength, h_size, sess, mainQN, time_per_step) :
	#   with open('.Center/log.csv', 'a') as myfile :
	#       state_display = (np.zeros([1, h_size]), np.zeros([1, h_size]))
	#       imagesS = []

	#       for idx, z in enumerate(np.vstack(bufferArray[:, 0])) :

	#           img, state_display = sess.run([])



	def Train(self) :
		tf.reset_default_graph()
		#We define the cells for the primary and target q-networks
		cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size,state_is_tuple=True)
		cellT = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size,state_is_tuple=True)
		mainQN = Qnetwork(self.h_size,cell,'main')
		targetQN = Qnetwork(self.h_size,cellT,'target')

		init = tf.global_variables_initializer()

		saver = tf.train.Saver(max_to_keep=5)

		trainables = tf.trainable_variables()

		targetOps = self.updateTargetGraph(trainables,self.tau)

		myBuffer = experience_buffer()

		#Set the rate of random action decrease. 
		e = self.startE
		stepDrop = (self.startE - self.endE)/self.anneling_steps


		#create lists to contain total rewards and steps per episode
		jList = []
		rList = []
		total_steps = 0

		#Make a path for our model to be saved in.
		if not os.path.exists(self.path):
			os.makedirs(self.path)

		##Write the first line of the master log-file for the Control Center
		with open('./Center/log.csv', 'a') as myfile:
			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			wr.writerow(['Episode','Length','Reward','IMG','LOG','SAL'])    
		self.net = cv2.dnn.readNetFromCaffe(self.prorotxt_path ,self.caffe_model_path)
		CLASSES = ['bottle', "background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor" ]

		with tf.Session() as sess:
			if self.load_model == True:
				print ('Loading Model...')
				ckpt = tf.train.get_checkpoint_state(self.path)
				saver.restore(sess,ckpt.model_checkpoint_path)
			sess.run(init)
   
			self.updateTarget(targetOps, sess) #Set the target network to be equal to the primary network.
			for i in range(self.num_episodes):

				self.col = -1
				self.width = -1
				self.row = -1
				self.height = -1
				self.frame = None
				self.frame2 = None
				self.inputmode = False
				self.rectangle = False
				self.trackWindow = None
				self.roi_hist = None
				self.roi = None

				self.cap = VideoStream('').start()
				time.sleep(2.0)
				fps = FPS().start()

				cv2.namedWindow('frame')
				cv2.setMouseCallback('frame', self.onMouse, param = (self.frame, self.frame2))

				episodeBuffer = []
				#Reset environment and get first new observation
				sP = self.game.Reset()
				s = self.processState(sP)
				d = False
				rAll = 0
				j = 0
				state = (np.zeros([1,self.h_size]),np.zeros([1,self.h_size])) #Reset the recurrent layer's hidden state
				#The Q-Network
				while j < self.max_epLength: 
					j+=1


					is_game_start = False
					self.frame = self.cap.read()
					#print(self.frame)

					self.frame = imutils.resize(self.frame, width = 200, height = 200)

					(h, w) = self.frame.shape[:2]
					blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 0.007843, (300, 300), 127.5)

					self.net.setInput(blob)
					detections = self.net.forward()

					self.obstacle_points = []
					for x in np.arange(0, detections.shape[2]) :
						confidence = detections[0, 0, x, 2]

						if confidence > 0.2 :  ### set for changing
							idx = int(detections[0, 0, x, 1])
							box = detections[0, 0, x, 3:7] * np.array([w, h, w, h])
							(startX, startY, endX, endY) = box.astype('int')

							label = "{}: {:.2f}%".format('obstacle', confidence * 100)
							cv2.rectangle(self.frame, (startX, startY), (endX, endY), self.obstacle_box_color, 2)
							self.obstacle_points.append({'row' : startY, 'col' : startX, 'row_size' : endY - startY, 'col_size' : endX - startX})

					if self.trackWindow is not None :
						ok, self.trackWindow = self.tracker.update(self.frame)

						if ok :
							x, y, w, h = self.trackWindow
							x, y, w, h = int(x), int(y), int(w), int(h)
							self.target_point = {'row' : int((2*y+h)/2), 'col' : int((2*x+w)/2)}
							cv2.rectangle(self.frame, (x, y), (x+w, y+w), (0, 255, 0), 3)
							is_game_start = True
						else : 
							cv2.putText(self.frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
							self.target_point = {'row' : -1, 'col' : -1}

					show_frame = cv2.resize(self.frame, None, fx = 2, fy = 2)

					cv2.imshow('frame', show_frame)

					print(self.target_point)

					key = cv2.waitKey(60) & 0xFF

					if key == ord('i') :
						print('select target')
						self.inputmode = True
						self.frame2 = self.frame.copy()

						while self.inputmode :
							cv2.imshow('frame', self.frame)
							cv2.waitKey(0)

					fps.update() ### Idont know where it locatied

					if not is_game_start :
						epi -=1
						continue
					else : 
						self.game.Update(self.target_point, self.obstacle_points)

					#Choose an action by greedily (with e chance of random action) from the Q-network
					if np.random.rand(1) < e or total_steps < self.pre_train_steps:
						state1 = sess.run(mainQN.rnn_state, feed_dict={mainQN.scalarInput:[s/255.0], mainQN.trainLength:1, mainQN.state_in:state, mainQN.batch_size:1})
						a = np.random.randint(0,5)
					else:
						a, state1 = sess.run([mainQN.predict,mainQN.rnn_state],\
							feed_dict={mainQN.scalarInput:[s/255.0],mainQN.trainLength:1,mainQN.state_in:state,mainQN.batch_size:1})
						a = a[0]

					s1P,r,d = self.game.Step(a)
					s1 = self.processState(s1P)
					total_steps += 1
					episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
					if total_steps > self.pre_train_steps:
						if e > self.endE:
							e -= stepDrop

						if total_steps % (self.update_freq) == 0:
							self.updateTarget(targetOps,sess)
							#Reset the recurrent layer's hidden state
							state_train = (np.zeros([self.batch_size,self.h_size]),np.zeros([self.batch_size,self.h_size])) 
					
							trainBatch = myBuffer.sample(self.batch_size, self.trace_length) #Get a random batch of experiences.
							#Below we perform the Double-DQN update to the target Q-values
							Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3]/255.0),mainQN.trainLength:self.trace_length,mainQN.state_in:state_train,mainQN.batch_size: self.batch_size})
							Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3]/255.0),targetQN.trainLength:self.trace_length,targetQN.state_in:state_train,targetQN.batch_size: self.batch_size})
							end_multiplier = -(trainBatch[:,4] - 1)
							doubleQ = Q2[range(self.batch_size*self.trace_length),Q1]
							targetQ = trainBatch[:,2] + (self.y*doubleQ * end_multiplier)
							#Update the network with our target values.
							sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]/255.0),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1],mainQN.trainLength:self.trace_length, mainQN.state_in:state_train,mainQN.batch_size:self.batch_size})
					rAll += r
					s = s1
					sP = s1P
					state = state1
					if d == True:

						break
				print(str(i) + '_th scorce : ' + str(rAll) + '/ episode : ' + str(j))
				self.game.Print_action_log()
				#Add the episode to the experience buffer
				bufferArray = np.array(episodeBuffer)
				episodeBuffer = list(zip(bufferArray))
				myBuffer.add(episodeBuffer)
				jList.append(j)
				rList.append(rAll)
				f = open('graph.txt', 'a')
				f.write(str(rAll))
				f.write('\n')
				f.close()

				#Periodically save the model. 
				if i % 90 == 0 and i != 0:
					saver.save(sess,self.path+'/model-'+str(i)+'.cptk')
					print ("Saved Model")
				if len(rList) % self.summaryLength == 0 and len(rList) != 0:
					print (total_steps, np.mean(rList[-self.summaryLength:]), e)
					#self.saveToCenter(i, rList, jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),self.summaryLength,self.h_size,sess, mainQN, self.time_per_step)
			saver.save(sess,self.path+'/model-'+str(i)+'.cptk')


	def onMouse(self, event, x, y, flags, param) :
		if self.inputmode : 
			if event == cv2.EVENT_LBUTTONDOWN :
				#print('DOWN')
				self.rectangle = True
				self.col, self.row = x, y

			elif event == cv2.EVENT_MOUSEMOVE :
				#print('MOVE')
				if self.rectangle :
					#print('Move - rec+true')
					self.frame = self.frame2.copy()
					cv2.rectangle(self.frame, (self.col, self.row), (x, y), (0, 255, 0), 2)
					cv2.imshow('frame', self.frame)

			elif event == cv2.EVENT_LBUTTONUP :
				#print('UP')
				self.inputmode = False
				self.rectangle = False
				cv2.rectangle(self.frame, (self.col, self.row), (x, y), (0, 255, 0), 2)
				self.height, self.width = abs(self.row - y), abs(self.col - x)
				self.trackWindow = (self.col, self.row, self.width, self.height)
				self.roi = self.frame[self.row : self.row + self.height, self.col : self.col+self.width]
				ok = self.tracker.init(self.frame, self.trackWindow)
				# roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
				# self.roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
				# cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
	
		return

	def Replay(self) :
		e = 0.01 #The chance of chosing a random action
		num_episodes = 10000 #How many episodes of game environment to train network with.
		load_model = True #Whether to load a saved model.
		path = "./drqn" #The path to save/load our model to/from.
		h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
		h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
		max_epLength = 50 #The max allowed length of our episode.
		time_per_step = 1 #Length of each step used in gif creation
		summaryLength = 100 #Number of epidoes to periodically save for analysis

		tf.reset_default_graph()
		cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
		cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
		mainQN = Qnetwork(h_size,cell,'main')
		targetQN = Qnetwork(h_size,cellT,'target')

		init = tf.global_variables_initializer()

		saver = tf.train.Saver(max_to_keep=2)

		game = r_sim(200)

		#create lists to contain total rewards and steps per episode
		jList = []
		rList = []
		total_steps = 0

		#Make a path for our model to be saved in.
		if not os.path.exists(path):
			os.makedirs(path)

		##Write the first line of the master log-file for the Control Center
		with open('./Center/log.csv', 'a') as myfile:
			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			wr.writerow(['Episode','Length','Reward','IMG','LOG','SAL'])    
		print('load_detector...')
		self.net = cv2.dnn.readNetFromCaffe(self.prorotxt_path ,self.caffe_model_path)
		CLASSES = ['bottle', "background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor" ]
			
		with tf.Session() as sess:
			print ('Loading Model...')
			ckpt = tf.train.get_checkpoint_state(path)
			saver.restore(sess,ckpt.model_checkpoint_path)
		
			for i in range(num_episodes):
				#Set Video
				self.col = -1
				self.width = -1
				self.row = -1
				self.height = -1
				self.frame = None
				self.frame2 = None
				self.inputmode = False
				self.rectangle = False
				self.trackWindow = None
				self.roi_hist = None
				self.roi = None

				self.cap = VideoStream('http://192.168.137.2:8080/?action=stream').start()
				time.sleep(2.0)
				fps = FPS().start()

				cv2.namedWindow('frame')
				cv2.setMouseCallback('frame', self.onMouse, param = (self.frame, self.frame2))

				episodeBuffer = []
				#Reset environment and get first new observation
				sP = game.Reset()
				s = self.processState(sP)
				d = False
				rAll = 0
				j = 0
				state = (np.zeros([1,h_size]),np.zeros([1,h_size]))
				#The Q-Network
				while True : #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
					j+=1
					#Choose an action by greedily (with e chance of random action) from the Q-network
					is_game_start = False
					self.frame = self.cap.read()
					#print(self.frame)

					self.frame = imutils.resize(self.frame, width = 200, height = 200)

					(h, w) = self.frame.shape[:2]
					blob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 0.007843, (300, 300), 127.5)

					self.net.setInput(blob)
					detections = self.net.forward()

					self.obstacle_points = []
					for x in np.arange(0, detections.shape[2]) :
						confidence = detections[0, 0, x, 2]

						if confidence > 0.2 :  ### set for changing
							idx = int(detections[0, 0, x, 1])
							box = detections[0, 0, x, 3:7] * np.array([w, h, w, h])
							(startX, startY, endX, endY) = box.astype('int')

							label = "{}: {:.2f}%".format('obstacle', confidence * 100)
							cv2.rectangle(self.frame, (startX, startY), (endX, endY), self.obstacle_box_color, 2)
							self.obstacle_points.append({'row' : startY, 'col' : startX, 'row_size' : endY - startY, 'col_size' : endX - startX})

					if self.trackWindow is not None :
						ok, self.trackWindow = self.tracker.update(self.frame)

						if ok :
							x, y, w, h = self.trackWindow
							x, y, w, h = int(x), int(y), int(w), int(h)
							self.target_point = {'row' : int((2*y+h)/2), 'col' : int((2*x+w)/2)}
							cv2.rectangle(self.frame, (x, y), (x+w, y+w), (0, 255, 0), 3)

							is_game_start = True					
					

						else : 
							cv2.putText(self.frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
							self.target_point = {'row' : -1, 'col' : -1}


					if self.target_point is not None and self.target_point['row'] == -1 :
						is_game_start = True

					show_frame = cv2.resize(self.frame, None, fx = 2, fy = 2)

					cv2.imshow('frame', show_frame)

					print(self.target_point)

					key = cv2.waitKey(60) & 0xFF

					if key == ord('i') :
						print('select target')
						self.inputmode = True
						self.frame2 = self.frame.copy()

						while self.inputmode :
							cv2.imshow('frame', self.frame)
							cv2.waitKey(0)

					fps.update() ### Idont know where it locatied

					print(is_game_start)

					if not is_game_start :
						j -=1
						continue
					else : 
						game.Update(self.target_point, self.obstacle_points)
					
					a, state1 = sess.run([mainQN.predict,mainQN.rnn_state],\
						feed_dict={mainQN.scalarInput:[s/255.0],mainQN.trainLength:1,\
						mainQN.state_in:state,mainQN.batch_size:1})

					#a = game.getting_fake_action()


					a = a[0]

					print('a : ' + str(a))

					s1P,r,d = game.Step(a)
					s1 = self.processState(s1P)
					total_steps += 1
					episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
					rAll += r
					s = s1
					sP = s1P
					state = state1
					if d == True:
		
						break
		
				bufferArray = np.array(episodeBuffer)
				jList.append(j)
				rList.append(rAll)

				#Periodically save the model. 
				if len(rList) % summaryLength == 0 and len(rList) != 0:
					print (total_steps,np.mean(rList[-summaryLength:]), e)
					saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
						summaryLength,h_size,sess,mainQN,time_per_step)
		print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
		