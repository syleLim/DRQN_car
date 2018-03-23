import tensorflow as tf
import numpy as np
import random
import time
import cv2

from Car_Sim import Car
from DQN import DQN

class agent_car :
    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT) :
        ### For Train
        self.target_pos = {'m_ row' : -1, 'm_col' : -1}
        self.obstacles_pos = list()

        self.MAX_EPISODE = 1000
        self.episode = 0
        # target network update at 1000 times
        self.TARGET_UPDATE_INTERVAL = 100
        # train each 4 frames
        self.TRAIN_INTERVAL = 4
        # After stack the datas and time, start training (with epsilon decrease)
        self.OBSERVE = 100

        # action :  0: not /  1: front /  2: back / 3: right / 4:left
        self.NUM_ACTION = 5
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.sess = None
        self.game = None
        self.brain = None
        self.rewards = None
        self.saver = None
        self.writer = None
        self.summary_merged = None

        # decide action decision time using dqn after this time
        self.epsilon = 1.0
        # number of frames
        self.time_step = 0
        self.total_reward_list = []

        ## Setting URL or Video Clip
        self.cap = None

        ### For Shift
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

        ## Flag for test : die count
        self.count_for_die = 0

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
                roi = self.frame[self.row : self.row + self.height, self.col : self.col+self.width]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                self.roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
                cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
    
        return

    def Train(self) :
        
        ## Session and brain / Game setting
        self.sess = tf.Session()
        self.game = Car(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.target_pos, self.obstacles_pos)
        self.brain = DQN(self.sess, self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.NUM_ACTION)

        ## Tensorflow model making
        self.rewards = tf.placeholder(tf.float32, [None])
        tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(self.rewards))
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.summary_merged = tf.summary.merge_all()

        ## Brain target network playing
        self.brain.update_target_network()

        # decide action decision time using dqn after this time
        self.epsilon = 1.0
        # number of frames
        self.time_step = 0
        self.total_reward_list = []

        while True:
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

            self.cap = cv2.VideoCapture(0)

            ## frame Window Setting
            ret, self.frame = self.cap.read()
            self.cap.set(3, self.SCREEN_WIDTH)
            self.cap.set(4, self.SCREEN_HEIGHT)

            ### Window Setting
            cv2.namedWindow('frame')
            cv2.setMouseCallback('frame', self.onMouse, param = (self.frame, self.frame2))

            ## MeanShift Setting
            termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            
            is_show = False

            terminal = False  # Flag for die
            total_reward = 0
            is_game_start = False

            ## gmae reset and get state (=poisition data)
            state = self.game.Reset(is_show, self.target_pos, self.obstacles_pos)
            self.brain.init_state(state)  # set state of dqn
            self.episode +=1

            if self.episode > self.MAX_EPISODE :
                break

            while not terminal:
                ### may be tensorflow cant be possible smme times.
                #detector.get_img(pre_frame)

                #frame, obstacles_pos = detector.process_img()
                ret, self.frame = self.cap.read()

                ## if video dont open, break train
                if not ret :
                    print('video_end some happen, please check')

                if self.trackWindow is not None :
                    hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                    dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
                    ## Maybe this window is track size
                    ret, self.trackWindow = cv2.meanShift(dst, self.trackWindow, termination)

                    x, y, w, h = self.trackWindow
                    self.target_pos = {'m_row' : int((y+h)/2), 'm_col' : int((x+w)/2)}
                    cv2.rectangle(self.frame, (x, y), (x+w, y+w), (0, 255, 0), 3)

                    cv2.line(self.frame, (int(self.SCREEN_WIDTH /2), int(self.SCREEN_HEIGHT)), (int((2*x+w)/2), int((2*y+h)/2)), (0, 255, 0), 2)
                    is_game_start = True
                    is_show = True
                else : 
                    self.target_pos = {'m_row' : -1, 'm_col' : -1} #in Sim m_row == -1 is_show = False
                    is_show = False

                cv2.imshow('frame', self.frame)

                key = cv2.waitKey(60) & 0xFF

                if key == ord('i') :
                    print('select target')
                    self.inputmode = True
                    self.frame2 = self.frame.copy()

                    while self.inputmode :
                        cv2.imshow('frame', self.frame)
                        cv2.waitKey(0)
                        

                ### TODO : how to get obstacle?

                print('is_game_start : ' + str(is_game_start))
                

                if is_game_start == False :
                    continue

                if np.random.rand() < self.epsilon:
                    action = self.game.get_action()
                else:
                    action = self.brain.get_action()

                ### TODO : i have to send message to car!!
                print("action_choice : " + str(action))

                # epsion decrease 
                if self.episode > self.OBSERVE:
                    epsilon -= 1 / 1000

                # game updates, get data (state / reward / is_gameover)
                state, reward, terminal = self.game.Update(is_show, self.target_pos, self.obstacles_pos)
                
                total_reward += reward

                # brain save this state
                self.brain.remember(state, action, reward, terminal)

                # After little time, In interval, do training
                if self.time_step > self.OBSERVE and self.time_step % self.TRAIN_INTERVAL == 0:
                    self.brain.train()

                # In interval, do training
                if self.time_step % self.TARGET_UPDATE_INTERVAL == 0:
                    self.brain.update_target_network()

                self.time_step += 1

                self.count_for_die +=1
                print(self.count_for_die)

                ### Only for Testing
                if self.count_for_die > 50 :
                    terminal = True
                    self.count_for_die = 0
                

            print('Number of game: %d  Score: %d' % (self.episode + 1, total_reward))

            self.total_reward_list.append(total_reward)

            if self.episode % 10 == 0:
                summary = self.sess.run(self.summary_merged, feed_dict={self.rewards: self.total_reward_list})
                self.writer.add_summary(summary, self.time_step)
                self.total_reward_list = []

            if self.episode % 100 == 0:
                self.saver.save(self.sess, 'model/dqn.ckpt', global_step=self.time_step)

            if self.episode > self.MAX_EPISODE :
                break

            self.cap.release()
            cv2.destroyAllWindows()
            self.cap = None
        