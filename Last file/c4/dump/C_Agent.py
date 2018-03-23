from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time

from C_Sim import Sim as sim
from DQN import DQN
from DQN import experience_buffer


class Agent :
    def __init__(self, flag) :
        self.batch_size = 64 #How many experiences to use for each training step.
        self.update_freq = 4 #How often to perform a training step.
        self.y = .99 #Discount factor on the target Q-values
        self.startE = 1 #Starting chance of random action
        self.endE = 0.1 #Final chance of random action
        self.annealing_steps = 10000. #How many steps of training to reduce startE to endE.
        self.num_episodes = 10000 #How many episodes of game environment to train network with.
        self.pre_train_steps = 10000 #How many steps of random actions before training begins.
        self.max_epLength = 300 #The max allowed length of our episode.
        self.load_model = False #Whether to load a saved model.
        self.path = "./dqn" #The path to save our model to.
        self.h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
        self.tau = 0.001 #Rate to update target network toward primary network
        self.action_num = 5

        tf.reset_default_graph()
        self.mainQN = DQN(self.h_size, self.action_num)
        self.targetQN = DQN(self.h_size, self.action_num)

        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

        self.trainables = tf.trainable_variables()

        self.targetOps = self.updateTargetGraph(self.trainables, self.tau)

        self.myBuffer = experience_buffer()

        # Set the rate of random action decrease. 
        self.e = self.startE
        self.stepDrop = (self.startE - self.endE)/self.annealing_steps

        # create lists to contain total rewards and steps per episode
        self.jList = []
        self.rList = []
        self.total_steps = 0

        self.game = sim(200, True)

        self.is_Train = flag

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
        self.tracker_type = self.tracker_types[2]

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


    ### Help function
    def processState(self, states):
        return np.reshape(states,[21168])

    def updateTargetGraph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx,var in enumerate(tfVars[0:total_vars//2]):
            op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
        return op_holder

    def updateTarget(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)

    def Check_path(path) :
        if not os.path(path) :
            os.makedirs(path)

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


    def Train(self) :
        if not self.is_Train :
            print('load image_model ...')
            self.net = cv2.dnn.readNetFromCaffe(self.prorotxt_path ,self.caffe_model_path)

        with tf.Session() as sess :
            sess.run(self.init)
            if self.load_model == True :
                print('load_model ...')
                ckpt = tf.train.get_checkpoint_state(path)
                saver.restore(sess, ckpt.model_checkpoint_path)

            for i in range(self.num_episodes) :
                if not self.is_Train :
                    CLASSES = ['bottle']
                    #["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat",
                    # "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant",
                    # "sheep","sofa", "train", "tvmonitor"
                    
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

                    self.cap = VideoStream(src=0).start()
                    time.sleep(2.0)
                    fps = FPS().start()

                    cv2.namedWindow('frame')
                    cv2.setMouseCallback('frame', self.onMouse, param = (self.frame, self.frame2))

                    #termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

                episode_buffer = experience_buffer()

                state = self.game.Reset()
                state = self.processState(state)

                dead = False
                reward_all = 0
                epi =0

                while epi < self.max_epLength :
                    epi +=1

                    if not self.is_Train :
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
                            # hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                            # dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
                            ## Maybe this window is track size
                        #     ret, self.trackWindow = cv2.meanShift(dst, self.trackWindow, termination)

                        #     x, y, w, h = self.trackWindow
                        #     self.target_point = {'row' : int((2*y+h)/2), 'col' : int((2*x+w)/2)}
                        #     cv2.rectangle(self.frame, (x, y), (x+w, y+w), (0, 255, 0), 3)
                        #     is_game_start = True                            
                        # else : 
                        #     self.target_point = {'row' : -1, 'col' : -1} #in Sim m_row == -1 is_show = False
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
                            self.game.Update_ob_points(self.target_point, self.obstacle_points)

                    if np.random.rand(1) < self.e or self.total_steps < self.pre_train_steps :
                        action = self.game.Get_action()
                    else :
                        action = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput : [state]})[0]

                    state_1, reward, dead = self.game.Step(action)
                    state_1 = self.processState(state_1)
                    self.total_steps += 1
                    episode_buffer.add(np.reshape(np.array([state, action, reward, state_1, dead]), [1, 5]))

                    if self.total_steps > self.pre_train_steps :
                        if self.e > self.endE :
                            self.e -= self.stepDrop

                        if self.total_steps % (self.update_freq) == 0 :
                            train_batch = self.myBuffer.sample(self.batch_size)

                            Q_1 = sess.run(self.mainQN.predict,feed_dict={self.mainQN.scalarInput:np.vstack(train_batch[:,3])})
                            Q_2 = sess.run(self.targetQN.Qout,feed_dict={self.targetQN.scalarInput:np.vstack(train_batch[:,3])})
                            end_mutiplier = -(train_batch[:,4] -1)
                            doubleQ = Q_2[range(self.batch_size), Q_1]
                            targetQ = train_batch[:,2] + (self.y * doubleQ * end_mutiplier)
                            
                            _ = sess.run(self.mainQN.updateModel, feed_dict={self.mainQN.scalarInput : np.vstack(train_batch[:,0]), self.mainQN.targetQ : targetQ, self.mainQN.actions : train_batch[:,1]})

                            self.updateTarget(self.targetOps, sess)

                    reward_all += reward
                    state = state_1

                    if dead == True :
                        break

                self.myBuffer.add(episode_buffer.buffer)
                self.jList.append(epi)
                self.rList.append(reward_all)

                f = open('./graph.txt', 'a')
                f.write(str(i)+'_th Game_End = Reward : ' + str(reward_all) + '/ Episode : ' + str(epi))
                f.write('\n')
                f.close() 


                self.game.Print_action_log()
                print(str(i)+'_th Game_End = Reward : ' + str(reward_all) + '/ Episode : ' + str(epi))

                if i % 100 == 0 :
                    self.saver.save(sess, self.path + '/model-'+str(i)+'.ckpt')
                    print('save model')

                if len(self.rList) % 10 == 0 :
                    print(self.rList)
                    print(len(self.rList))
                    print(self.total_steps)
                    print(np.mean(self.rList[-10:]))
                    print(self.e)
                    f_2 = open('./reward_mean.txt', 'a')
                    f_2.write(str(i)+'th : ' + str(np.mean(self.rList[-10:])))
                    f_2.write('\n')
                    f_2.close()

                if not self.is_Train :
                    cv2.destroyAllWindows()

            self.saver.save(sess, self.path + '/model-'+str(i)+'.ckpt')

        print("Percent of succesful episodes: " + str(sum(self.rList)/self.num_episodes) + "%")

        rMat = np.resize(np.array(self.rList),[len(self.rList)//100,100])
        rMean = np.average(rMat,1)
        plt.plot(rMean)


    def Play(self) :
        if not self.is_Train :
            print('load image_model ...')
            self.net = cv2.dnn.readNetFromCaffe(self.prorotxt_path ,self.caffe_model_path)

        with tf.Session() as sess :
            sess.run(self.init)
            if self.load_model == True :
                print('load_model ...')
                ckpt = tf.train.get_checkpoint_state(path)
                saver.restore(sess, ckpt.model_checkpoint_path)

            for i in range(self.num_episodes) :
                if not self.is_Train :
                    CLASSES = ['bottle']
                    #["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat",
                    # "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant",
                    # "sheep","sofa", "train", "tvmonitor"
                    
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

                    self.cap = VideoStream(src=0).start()
                    time.sleep(2.0)
                    fps = FPS().start()

                    cv2.namedWindow('frame')
                    cv2.setMouseCallback('frame', self.onMouse, param = (self.frame, self.frame2))

                    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

                episode_buffer = experience_buffer()

                state = self.game.Reset()
                state = self.processState(state)

                dead = False
                reward_all = 0

                while True :
                    if not self.is_Train :
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
                            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                            dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
                            ## Maybe this window is track size
                            ret, self.trackWindow = cv2.meanShift(dst, self.trackWindow, termination)

                            x, y, w, h = self.trackWindow
                            self.target_point = {'row' : int((2*y+h)/2), 'col' : int((2*x+w)/2)}
                            cv2.rectangle(self.frame, (x, y), (x+w, y+w), (0, 255, 0), 3)
                            is_game_start = True                            
                        else : 
                            self.target_point = {'row' : -1, 'col' : -1} #in Sim m_row == -1 is_show = False

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
                            continue
                        else : 
                            self.game.Update_ob_points(self.target_point, self.obstacle_points)

                    action = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput : [state]})[0]

                    state_1, reward, dead = self.game.Step(action)

                    if dead == True :
                        break

                self.jList.append(epi)
                self.rList.append(reward_all)

                f = open('./play_graph.txt', 'a')
                f.write(str(i)+'_th Game_End = Reward : ' + str(reward_all))
                f.write('\n')
                f.close() 
                self.game.Print_action_log()
                print(str(i)+'_th Game_End = Reward : ' + str(reward_all))

                if not self.is_Train :
                    cv2.destroyAllWindows()

        print("Percent of succesful episodes: " + str(sum(self.rList)/self.num_episodes) + "%")

        rMat = np.resize(np.array(self.rList),[len(self.rList)//100,100])
        rMean = np.average(rMat,1)
        plt.plot(rMean)