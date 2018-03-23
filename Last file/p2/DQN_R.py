import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim

#from helper import *

from Sim import Sim as sim

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
		self.load_model = False #Whether to load a saved model.
		self.path = "./drqn" #The path to save our model to.
		self.h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
		self.max_epLength = 300 #The max allowed length of our episode.
		self.time_per_step = 1 #Length of each step used in gif creation
		self.summaryLength = 100 #Number of epidoes to periodically save for analysis
		self.tau = 0.001

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
	# 	with open('.Center/log.csv', 'a') as myfile :
	# 		state_display = (np.zeros([1, h_size]), np.zeros([1, h_size]))
	# 		imagesS = []

	# 		for idx, z in enumerate(np.vstack(bufferArray[:, 0])) :

	# 			img, state_display = sess.run([])



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
		

		with tf.Session() as sess:
			if self.load_model == True:
				print ('Loading Model...')
				ckpt = tf.train.get_checkpoint_state(self.path)
				saver.restore(sess,ckpt.model_checkpoint_path)
			sess.run(init)
			self.updateTarget(targetOps, sess) #Set the target network to be equal to the primary network.
			for i in range(self.num_episodes):
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
					#Choose an action by greedily (with e chance of random action) from the Q-network
					if np.random.rand(1) < e or total_steps < self.pre_train_steps:
						state1 = sess.run(mainQN.rnn_state, feed_dict={mainQN.scalarInput:[s/255.0], mainQN.trainLength:1, mainQN.state_in:state, mainQN.batch_size:1})
						a = self.game.Get_action()
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


	def saveToCenter(self, i,rList,jList,bufferArray,summaryLength,h_size,sess,mainQN,time_per_step):
		with open('./Center/log.csv', 'a') as myfile:
			state_display = (np.zeros([1,h_size]),np.zeros([1,h_size]))
			imagesS = []
			for idx,z in enumerate(np.vstack(bufferArray[:,0])):
				img,state_display = sess.run([mainQN.salience,mainQN.rnn_state],\
					feed_dict={mainQN.scalarInput:np.reshape(bufferArray[idx,0],[1,21168])/255.0,\
					mainQN.trainLength:1,mainQN.state_in:state_display,mainQN.batch_size:1})
				imagesS.append(img)
			imagesS = (imagesS - np.min(imagesS))/(np.max(imagesS) - np.min(imagesS))
			imagesS = np.vstack(imagesS)
			imagesS = np.resize(imagesS,[len(imagesS),84,84,3])
			luminance = np.max(imagesS,3)
			imagesS = np.multiply(np.ones([len(imagesS),84,84,3]),np.reshape(luminance,[len(imagesS),84,84,1]))
			#self.make_gif(np.ones([len(imagesS),84,84,3]),'./Center/frames/sal'+str(i)+'.gif',duration=len(imagesS)*time_per_step,true_image=False,salience=True,salIMGS=luminance)

			images = zip(bufferArray[:,0])
			images.append(bufferArray[-1,3])
			images = np.vstack(images)
			images = np.resize(images,[len(images),84,84,3])
			#self.make_gif(images,'./Center/frames/image'+str(i)+'.gif',duration=len(images)*time_per_step,true_image=True,salience=False)

			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			wr.writerow([i,np.mean(jList[-100:]),np.mean(rList[-summaryLength:]),'./frames/image'+str(i)+'.gif','./frames/log'+str(i)+'.csv','./frames/sal'+str(i)+'.gif'])
			myfile.close()
		with open('./Center/frames/log'+str(i)+'.csv','w') as myfile:
			state_train = (np.zeros([1,h_size]),np.zeros([1,h_size]))
			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			wr.writerow(["ACTION","REWARD","A0","A1",'A2','A3','V'])
			a, v = sess.run([mainQN.Advantage,mainQN.Value],\
				feed_dict={mainQN.scalarInput:np.vstack(bufferArray[:,0])/255.0,mainQN.trainLength:len(bufferArray),mainQN.state_in:state_train,mainQN.batch_size:1})
			wr.writerows(zip(bufferArray[:,1],bufferArray[:,2],a[:,0],a[:,1],a[:,2],a[:,3],v[:,0]))
	
#This code allows gifs to be saved of the training episode for use in the Control Center.
	def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
		import moviepy.editor as mpy
  
		def make_frame(t):
			try :
				x = images[int(len(images)/duration*t)]
			except:
				x = images[-1]

			if true_image:
				return x.astype(np.uint8)
			else:
				return ((x+1)/2*255).astype(np.uint8)
  
		def make_mask(t):
			try:
				x = salIMGS[int(len(salIMGS)/duration*t)]
			except:
				x = salIMGS[-1]
			return x

		clip = mpy.VideoClip(make_frame, duration=duration)
		if salience == True:
			mask = mpy.VideoClip(make_mask, ismask=True,duration= duration)
			clipB = clip.set_mask(mask)
			clipB = clip.set_opacity(0)
			mask = mask.set_opacity(0.1)
			mask.write_gif(fname, fps = len(images) / duration,verbose=False)
			#clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
		else:
			clip.write_gif(fname, fps = len(images) / duration,verbose=False)
