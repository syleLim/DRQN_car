import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import time
import random
import math
import itertools
import scipy.misc

#import client_App

class Sim :
	def __init__(self, screen, Mode_Sim) :
		# Check is Sim or RealPlay
		self.Mode_Sim = Mode_Sim
		if self.Mode_Sim :
			self.Map = None
			self.vertical_speed = 5
			self.horizontal_speed_v = 4
			self.horizontal_speed_h = 1
			self.g_target_point = None
			self.g_obstacle_points = None
			self.g_player_point = None
			self.move_count = 0
			self.target_move = False
		
		# view Map _ course
		self.row = screen
		self.col = screen

		# Train Map _ course
		self.divide = 10
		self.t_row = int(screen/self.divide)
		self.t_col = int(screen/self.divide)

		self.player_point = None
		self.target_point = None
		self.obstacle_points = []
		self.obstacle_num = 15
		## obstacle['row', 'col', 'row_size', 'col_size']

		# object init for Train Map
		self.t_middle_point = None
		self.t_player_point = None
		self.t_target_point = None
		self.t_obstacle_points = []
		## t_obstacle['row', 'col', 'row_end', 'col_end']

		# flag setting
		self.action = 5
		self.is_stable = False ## for score, if stable == true, score +1
		self.is_target_not_view = False
		self.is_target_out = False
		self.print_count = 0

		# for log
		self.stable_count = 0
		self.missing_count = 0
		self.goal_count = 0
		self.collision_count = 0
		self.action_list = []

		self.goal_reward = 20
		self.stable_reward = 1
		self.missing_panalty = -3
		self.collison_panalty = -5
		self.not_view_count = 0

		self.Reset()


	def Reset(self) :
		if self.Mode_Sim :
			self.Map = game_Map(2*self.row, 2*self.col)
			self.g_player_point = {'row' : int((3/4)*self.row), 'col' : int(self.col/2)}
			self.g_target_point = {'row' : np.random.randint(185, 215), 'col' : np.random.randint(185, 215) }
			self.g_obstacle_points = []
			for i in range(self.obstacle_num) :
				self.g_obstacle_points.append(self.Make_obstacle())

		# object init for view_Map
		self.player_point = {'row' : self.row-1, 'col' : int(self.col/2)}
		self.target_point = {'row' : self.g_target_point['row'] - 100, 'col' : self.g_target_point['col'] - 100}
		self.obstacle_points = []
		
		# object init for Train Map
		self.t_middle_point = {'row' : int(self.t_row/2), 'col' : int(self.t_col/2)}
		self.t_player_point = {'row' : self.t_row, 'col' : int(self.t_col/2)}
		self.t_target_point = {'row' : -1, 'col' : -1}
		self.t_obstacle_points = []

		self.is_stable = False
		self.is_target_out = False
		self.is_target_not_view = False

		state = self.Make_State()

		self.stable_count = 0
		self.missing_count = 0
		self.goal_count = 0
		self.collision_count = 0
		self.action_list = []

		return state

	def Step(self, action) :


		self.Move(action)

		if self.Mode_Sim :
			self.Target_move()

			self.Check_target_out()

			if not self.is_target_out:
				self.Map.Update(self.g_target_point, self.g_obstacle_points)
				self.Check_obstacle_out()

				k = -1
				
				target_find = False

				for i in range(self.row) :
					for j in range(self.col) :
						if self.Map.Map[int((1/2)*self.row) + i, int((1/2)*self.col) + j] == 1 :
							self.target_point['row'] = i
							self.target_point['col'] = j
							self.is_target_not_view = False
							target_find = True

				for i in range(self.row + 49) : 
					for j in range(self.col +24) :
						pos_row = int((1/2)*self.row) - 49 + i
						pos_col = int((1/2)*self.col) - 24 + j
						if int(self.Map.Map[pos_row, pos_col]/1000000) == 1 :
							k +=1
							if len(self.obstacle_points) < k+1 :
								self.obstacle_points.append({'row' : -1, 'col' : -1, 'row_size' : 0, 'col_size' : 0})

							temp = self.Map.Map[pos_row, pos_col]
							temp = temp - 1000000
							row_end = int(temp/1000)
							col_end = temp - 1000*row_end

							#print('temp' + str(temp))

							if pos_row < int((1/2)*self.row) :
								row_size = row_end - int((1/2)*self.row)
								pos_row = 0
							else :
								row_size = row_end - pos_row
								pos_row  = pos_row - int((1/2)*self.row)
							
							if pos_col < int((1/2) * self.col) :
								col_size = col_end - int((1/2)*self.col)
								pos_col = 0
							else : 
								col_size = col_end - pos_col
								pos_col = pos_col - int((1/2)*self.col)
						
							self.obstacle_points[k]['row'] = pos_row
							self.obstacle_points[k]['col'] = pos_col
							self.obstacle_points[k]['row_size'] = row_size
							self.obstacle_points[k]['col_size'] = col_size

				self.obstacle_points = self.obstacle_points[:k+1]

				if not target_find :
					self.target_point['row'] = -1
					self.target_point['col'] = -1
					self.is_target_not_view = True
				
		self.Get_Target_Train_pos()
		self.Get_Obstacle_Train_pos()

		#print('action : ' +str(action))
		#print('target : ' +str(self.target_point))
		#print('player_point' + str(self.player_point))
		#print('obstacle')
		#for i in self.obstacle_points :
		#	print(i)
		
		#print('t_target : ' +str(self.t_target_point))
		#print('t_middle : ' + str(self.t_middle_point))
		#print('t_player_ point' + str(self.t_player_point))
		#print('t_obstacle')
		#for i in self.t_obstacle_points :
		#	print(i)

		state = self.Make_State()
		reward = self.Check_goal() + 1 ### one frame is +1 point
		dead_flag, dead_panalty = self.Check_over()

		self.print_count +=1
		"""
		if self.print_count%20 == 0 :
			print('t_middle_point' + str(self.t_middle_point))

			print('Game_map_target')
			print('tartget : ' +str(self.g_target_point))
			print('obstacle')
			for i in self.g_obstacle_points :
				print(i) 

			print('current state')
			print('target : ' + str(self.target_point))
			print('t_target_point : ' + str(self.t_target_point))

			print('obstacle : len = ' +str(len(self.obstacle_points))+ " / t_len = " + str(len(self.t_obstacle_points)))
			for i in self.obstacle_points :
				print(i) 
			for i in self.t_obstacle_points :
				print(i)
		"""

		return state, reward+dead_panalty, dead_flag


	def Make_State(self) : 
		state = np.ones((self.t_row+2, self.t_col+2, 3), dtype=np.int32)

		state[1:-1, 1:-1, :] = 0

		state[self.t_middle_point['row']+1:self.t_middle_point['row']+2 , self.t_middle_point['col']+1:self.t_middle_point['col']+2, 2] = 1

		if self.is_target_not_view :
			pass
		else :
			state[self.t_target_point['row']:self.t_target_point['row']+1, self.t_target_point['col']:self.t_target_point['col']+1, 1] = 1

		for obstacle in self.t_obstacle_points :
			if obstacle['row'] == -1 or obstacle['col'] == -1 :
				continue
			else :
				state[obstacle['row']:obstacle['row']+obstacle['row_end'], obstacle['col']:obstacle['col']+obstacle['col_end'], 0] = 1

		b = scipy.misc.imresize(state[:, :, 0], [84, 84, 1], interp='nearest')
		c = scipy.misc.imresize(state[:, :, 1], [84, 84, 1], interp='nearest')
		d = scipy.misc.imresize(state[:, :, 2], [84, 84, 1], interp='nearest')
		state = np.stack([b, c, d], axis=2)

		return state
		

	def Check_goal(self) :
		stable_reward = 0
		goal_reward = 0
		collison_panalty = 0
		missing_panalty = 0
		not_view_panalty = 0
		is_missing = False
		is_collision = False

		if self.is_stable :
			stable_reward = self.stable_reward
			self.is_stable = False
			self.stable_count +=1
			#print('get_stable_point')

		if self.t_middle_point['row']+1 == self.t_target_point['row'] and self.t_middle_point['col']+1 == self.t_target_point['col'] :
			#print('goal point get')
			goal_reward = self.goal_reward
			self.goal_count +=1

		for obstacle in self.t_obstacle_points :
			for i in range(obstacle['row_end']) :
				for j in range(obstacle['col_end']) :
					if self.t_player_point['row'] == obstacle['row'] + i and self.t_player_point['col'] == obstacle['col'] + j :
						#print('collision panalty')
						collison_panalty = self.collison_panalty
						self.collision_count +=1
						
				if is_missing and is_collision :
					break
			if is_missing and is_collision :
				break

		if self.is_target_not_view :
			#print('not_view panalty')
			self.missing_count +=1
			not_view_panalty = self.missing_panalty
			self.not_view_count +=1
		else :
			self.not_view_count = 0

		return stable_reward + goal_reward + collison_panalty + missing_panalty + not_view_panalty

	def Check_over(self) :
		if self.is_target_out :
			print('target_out in game_Map')
			return True, -100

		for obstacle in self.obstacle_points :
			for i in range(obstacle['row_size']) :
				for j in range(obstacle['col_size']) :
					if self.player_point['row'] == obstacle['row'] + i and self.player_point['col'] == obstacle['col'] + j :
						print('real_collapse')
						return True, -100
		
		if self.not_view_count > 50 :
			return True, -100

		return False, 0

		
	def Make_obstacle(self) :
		row = np.random.randint(30, 200)
		if row < 80 :
			col = np.random.randint(100, 300)
		else :
			col = random.choice([np.random.randint(70, 140), np.random.randint(250, 320)])
		row_size = np.random.randint(50, 80)
		obstacle = {'row' : row , 'col' : col , 'row_size' : row_size  , 'col_size' : int(row_size/2) }

		return obstacle

	def Move(self, flag) :
		if flag == 0 : 
			self.Move_Stable()
		elif flag == 1 :
			self.Move_forward()
		elif flag == 2 :
			self.Move_back()
		elif flag == 3 :
			self.Move_right()
		elif flag == 4 :
			self.Move_left()

		self.action_list.append(flag)


	def Move_Stable(self) :
		self.is_stable = True

	def Move_forward(self) :
		if self.Mode_Sim :
			self.g_target_point['row'] += self.vertical_speed

			if len(self.g_obstacle_points) is not 0 :
				for obstacle in self.g_obstacle_points :
					obstacle['row'] += self.vertical_speed
		else :
			#client_App.forward_fun()
			pass

	def Move_back(self) :
		if self.Mode_Sim :
			self.g_target_point['row'] -= self.vertical_speed

			if len(self.g_obstacle_points) is not 0 :
				for obstacle in self.g_obstacle_points :
					obstacle['row'] -= self.vertical_speed
		else :
			#client_App.backward_fun()
			pass

	def Move_right(self) :
		if self.Mode_Sim :
			self.g_target_point['col'] -= self.horizontal_speed_h
			self.g_target_point['row'] += self.horizontal_speed_v

			if len(self.g_obstacle_points) is not 0 :
				for obstacle in self.g_obstacle_points :
					obstacle['col'] -= self.horizontal_speed_h
					obstacle['row'] += self.horizontal_speed_v
		else :
			#client_App.right_fun()
			pass

	def Move_left(self) :
		if self.Mode_Sim :
			self.g_target_point['col'] += self.horizontal_speed_h
			self.g_target_point['row'] += self.horizontal_speed_v

			if len(self.g_obstacle_points) is not 0 :
				for obstacle in self.g_obstacle_points :
					obstacle['col'] += self.horizontal_speed_h
					obstacle['row'] += self.horizontal_speed_v
		else :
			#client_App.left_fun()
			pass


	# Train pos Check
	def Get_Target_Train_pos(self) :
		row_interval = self.divide
		col_interval = self.divide
		t_row = 1
		t_col = 1

		while True :
			if self.target_point['row'] < 0 or self.target_point['row'] > self.row :
				self.t_target_point['row'] = -1
				self.is_target_not_view = True
				break
			
			if self.target_point['row'] < row_interval :
				self.is_target_not_view = False
				self.t_target_point['row'] = t_row
				break
			else :
				row_interval += self.divide
				t_row +=1

				if t_row > self.t_row :
					self.t_target_point['row'] = -1
					self.is_target_not_view = True
					break

		while True : 
			if self.target_point['col'] < 0 or self.target_point['col'] > self.col :
				self.t_target_point['col'] = -1
				self.is_target_not_view = True
				break
			
			if self.target_point['col'] < col_interval :
				self.is_target_not_view = False
				self.t_target_point['col'] = t_col
				break
			else :
				col_interval += self.divide
				t_col +=1

				if t_col > self.t_row :
					self.t_target_point['col'] = -1
					self.is_target_not_view = True
					break

	def Get_Obstacle_Train_pos(self) :
		# init t_obstacle if obstacle num change -> it is easy (have to change pop or append decount)
		if len(self.obstacle_points) is not len(self.t_obstacle_points) :
			self.t_obstacle_points = []

			for i in range(len(self.obstacle_points)) :
				self.t_obstacle_points.append({'row' : -1, 'col' : -1, 'row_end' : -1, 'col_end': -1})

		if len(self.obstacle_points) is not 0 :
			i = -1

			for obstacle in self.obstacle_points :
				i +=1
				row_interval = self.divide
				col_interval = self.divide
				t_row = 1
				t_row_end = 2
				t_col = 1
				t_col_end = 2

				while True :
					if obstacle['row'] < 0 or obstacle['row'] >= self.row :
						self.t_obstacle_points[i]['row'] = -1
						break
			
					if obstacle['row'] < row_interval :
						self.t_obstacle_points[i]['row'] = t_row

						while True:
							if obstacle['row'] + obstacle['row_size'] < row_interval :

								self.t_obstacle_points[i]['row_end'] = t_row_end - t_row
								break
							else : 
								row_interval += self.divide
								t_row_end +=1

								if t_row_end > self.t_row :
									self.t_obstacle_points[i]['row_end'] = t_row_end - t_row - 1
									break
						break
					else :
						row_interval += self.divide
						t_row +=1
						t_row_end +=1

						if t_row > self.t_row :
							self.t_obstacle_points[i]['row'] = -1
							break

				while True : 
					if obstacle['col'] < 0 or obstacle['col'] >= self.col :
						self.t_obstacle_points[i]['col'] = -1
						break
			
					if obstacle['col'] < col_interval :
						self.t_obstacle_points[i]['col'] = t_col

						while  True:
							if obstacle['col'] + obstacle['col_size'] < col_interval :
								self.t_obstacle_points[i]['col_end'] = t_col_end - t_col
								break
							else :
								col_interval += self.divide
								t_col_end +=1

								if t_col_end > self.t_col :
									self.t_obstacle_points[i]['col_end'] = t_col_end - t_col - 1
									break

						break
					else :
						col_interval += self.divide
						t_col +=1
						t_col_end +=1

						if t_col > self.t_row :
							self.t_obstacle_points[i]['col'] = -1
							break

	def Check_target_out(self) : 
		if self.g_target_point['row'] >= 2*self.row or self.g_target_point['col'] >= 2*self.col or self.g_target_point['row'] < 0 or self.g_target_point['col'] < 0:
			self.is_target_out = True

	def Check_obstacle_out(self) :
		if len(self.g_obstacle_points) is not 0 :
			delete_count = 0

			for obstacle in self.g_obstacle_points :
				if obstacle['row'] >= int((3/2)*self.row) or obstacle['col'] >= 2*self.col or obstacle['row'] < 0 or obstacle['col'] < 0:
					self.g_obstacle_points.remove(obstacle)
					delete_count +=1

			if delete_count is not 0 :
				for i in range(delete_count) :
					self.g_obstacle_points.append(self.Make_obstacle())

	def Target_move(self) :
		self.move_count +=1

		if self.target_move :
			if self.move_count > 20 :
				self.target_move = False
				self.move_count = 0
		else :
			if self.move_count > 30 :
				self.target_move = True
				self.move_count = 0
			
		if self.target_move :
			self.g_target_point['row'] += np.random.randint(-1, 5)
			self.g_target_point['col'] += np.random.randint(-5, 5)

	def Print_action_log(self) :
		print('stable : ' + str(self.stable_count) +' /goal : ' + str(self.goal_count) + ' /missing : ' + str(self.missing_count) +' /collision : '  +str(self.collision_count))
		

	def Update_ob_points(self, target_point, obstacle_points) :
		self.target_point = target_point
		self.obstacle_points = obstacle_points

	def Get_action(self) :
		if self.t_target_point['row'] < self.t_middle_point['row'] :
			if self.t_target_point['col'] > self.t_middle_point['col'] :
				return random.choice([1, 3])
			else :
				return random.choice([1, 4])

		if self.t_target_point['row'] > self.t_middle_point['row'] :
			return 2

		if self.t_target_point['row'] == self.t_middle_point['row'] and self.t_target_point['col'] == self.t_middle_point['col'] :
			return 0

		return np.random.randint(0, 5)


class game_Map :
	def __init__(self, row, col) :
		self.row = row
		self.col = col
		self.Map = np.zeros((row, col), dtype=np.int32)


	def Update(self, target_point, obstacle_points) :
		self.Map = np.zeros((self.row, self.col), dtype=np.int32)

		if target_point['row'] < 0 or target_point['row'] >= self.row or target_point['col'] < 0 or target_point['col'] >= self.col :
			pass
		else :
			self.Map[target_point['row'], target_point['col']] = 1

		if len(obstacle_points) is not 0 :
			for obstacle in obstacle_points :
				if obstacle['row'] < 0 or obstacle['row'] >= self.row or obstacle['col'] < 0 or obstacle['col'] >= self.col :
					continue

				if obstacle['row'] + obstacle['row_size'] > self.row :
					row_end = self.row-1
				else :
					row_end = obstacle['row'] + obstacle['row_size']

				if obstacle['col'] + obstacle['col_size'] > self.col :
					col_end = self.col-1
				else :
					col_end = obstacle['col'] + obstacle['col_size']

				self.Map[obstacle['row'], obstacle['col']] = 1000000 + 1000*row_end + col_end
