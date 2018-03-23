import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import time
import random

class Car:
	def __init__(self, screen_width, screen_height, target_pos, obstacles_pos): 

		self.screen_height = screen_height
		self.screen_width = screen_width       #Set Camera width == 640

		self.middle_point = { 'row' : int(self.screen_height/2), 'col' : int(self.screen_width/2) }  #goal point
		self.player_point = { 'row' : int((3/2) * self.screen_height), 'col' : int(self.screen_width) }

		# data for sending dqn
		self.car_map = np.zeros((self.screen_width, self.screen_height))

		self.target_pos = target_pos
		self.obstacles_pos = obstacles_pos

		# for reward
		self.total_reward = 0.
		self.current_reward = 0.
		self.total_game = 0

		# flag for target is hiding or not
		self.is_show = False

		#flag for deciding defeat obstacle
		self.obstacles_num = len(self.obstacles_pos)
		self.pre_obstacles_num = len(self.obstacles_pos)

		#flag for checking stablity
		self.is_stable = False
		self.missing_count = 0

	
	# target_pos is composed m_row, m_col, h_size
	# obstacle pos is composed row, col , size
	# get data for car image = car_image_process is first 
	def Update_map(self, is_tartget_show, target_pos, obstacles_pos) :
		#initial map
		self.car_map = np.zeros((self.screen_height, self.screen_width))
		"""
		for i in range(self.screen_height) :
			for j in range(self.screen_width) :
				self.car_map[i, j] = 0                 i dont know what is fast
		"""
		
		#target just have middle_point setting

		if is_tartget_show :
			self.target_pos = target_pos
			self.car_map[self.target_pos['m_row'], self.target_pos['m_col']] = 1
			self.is_show = True
			self.missing_count = 0
		else :
			self.target_pos['m_row'] = -1
			self.target_pos['m_col'] = -1
			self.is_show = False
			self.missing_count += 1
		
		self.obstacles_pos = obstacles_pos
		self.obstacles_num = len(obstacles_pos)
		#obstacle is set with their size
		if self.obstacles_num is not 0 :
			for obstacle_pos in self.obstacles_pos :
				for i in ragne(obstacle_pos['size']) :
					for j in range(obstacle_pos['size']) :
						self.car_map[obstacle_pos['row']+i, obstacle_pos['col'] + j] = 2
		else :
			self.obstacles_pos = list()

	def Game_over(self) :
		reward = 0
		is_over = False

		if self.missing_count > 100 :
			is_over = True
			reward = -500
			print('long time no see')

		if is_over == False :
			if self.car_map[self.player_point['row'], self.player_point['col']] == 2 :
				print('meet_obstacle !')
				is_over = True
				reward = -100
		if is_over :
			self.total_reward +=self.current_reward

		return is_over, reward
		

	#data update
	def Update(self, is_target_show, target_pos, obstacles_pos) :
		#first, update map and car
		self.Update_map(is_target_show, target_pos, obstacles_pos)
		# move_action = self.Car_move(flag)

		is_over, die_reward = self.Game_over()

		# TODO : Check Score
		target_reward = self.Score_fit_middle() + self.Score_in_view()
		obstacle_reward = self.Score_defeat_obstacle()
		
		if is_over :
			reward = die_reward
		else :
			reward = target_reward + obstacle_reward
			self.current_reward += reward
 
		return self.car_map, reward, is_over


	#if epsilon is lower checking side all
	def  get_action(self) : 
		if self.target_pos['m_row'] < self.middle_point['row'] : #have to go front
			if self.target_pos['m_col'] < self.middle_point['col'] : #have to go left
				temp_action = [0, 1, 4]
				action = random.choice(temp_action)
			elif self.target_pos['m_col'] > self.middle_point['col'] :
				temp_action = [0, 1, 3]
				action = random.choice(temp_action)
			else :
				temp_action = [0, 1]
				action = random.choice(temp_action)

		elif self.target_pos['m_row'] > self.middle_point['row'] : #have to go back
			if self.target_pos['m_col'] < self.middle_point['col'] : #have to go left
				temp_action = [0, 2, 4]
				action = random.choice(temp_action)
			elif self.target_pos['m_col'] > self.middle_point['col'] :
				temp_action = [0, 2, 3]
				action = random.choice(temp_action)
			else :
				temp_action = [0, 2]
				action = random.choice(temp_action)
		else : 
			if self.target_pos['m_col'] < self.middle_point['col'] : #have to go left
				temp_action = [0, 4]
				action = random.choice(temp_action)
			elif self.target_pos['m_col'] > self.middle_point['col'] : 
				temp_action = [0, 3]
				action = random.choice(temp_action)
			else : 
				action = 0

		print('use get_action, action is :' + str(action))
		# is a flag
		return action 

	#Moving Signal Send
	def Move_not(self) :
		self.is_stable = True
		

	def Move_forward(self) : 
		self.is_stable = False
		#TODO : call car go right (may be send the server)
		

	def Move_back(self) :
		self.is_stable = False
		#TODO : call car go right (may be send the server)
		

	def Move_right(self) :
		self.is_stable = False
		#TODO : call car go right (may be send the server)
		

	def Move_left(self) :
		self.is_stable = False
		#TODO : call car go right (may be send the server)
		
	################ For Scoring ################
	def Score_fit_middle(self) :
		reward = 0

		if self.middle_point['row'] == self.target_pos['m_row'] and self.middle_point['col'] == self.target_pos['m_col'] :
			reward = 100

		return reward

	def Score_in_view(self) :
		if self.is_show :
			reward = 1
		else :
			self.reward = 0

		return reward

	def Score_defeat_obstacle(self) :
		reward = 0

		if self.pre_obstacles_num < self.obstacles_num :
			reward = self.obstacles_num - self.pre_obstacles_num

		self.pre_obstacles_num = self.obstacles_num

		return reward



	def Reset(self, is_target_show, target_pos, obstacles_pos) :
		self.current_reward = 0
		self.total_game += 1

		self.middle_point = { 'row' : int(self.screen_height/2), 'col' : int(self.screen_width/2) }  #goal point
		self.player_point = { 'row' : int((3/4) * self.screen_height), 'col' : int(self.screen_width/2) }

		self.is_show = False

		#flag for checking stablity
		self.is_stable = False

		self.target_pos = target_pos
		self.obstacles_pos = obstacles_pos

		#### For Map ####
		self.car_map = np.zeros((self.screen_height, self.screen_width))

		self.Update_map(is_target_show, target_pos, obstacles_pos)

		#flag for deciding defeat obstacle
		self.obstacles_num = len(self.obstacles_pos)
		self.pre_obstacles_num = len(self.obstacles_pos)

		return self.car_map

