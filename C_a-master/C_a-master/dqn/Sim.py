import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import time
import random
import math

class Sim:
    def __init__(self, screen_width, screen_height): 
        self.screen_height = screen_height
        self.screen_width = screen_width       #Set Camera width == 640
        self.view_height = int(screen_height/2)
        self.view_width = int(screen_width/2)

        self.middle_point = { 'row' : int(self.screen_height/2), 'col' : int(self.screen_width/2) }  #goal point
        self.player_point = { 'row' : int((3/4) * self.screen_height), 'col' : int(self.screen_width/2) }

        # for reward
        self.total_reward = 0.
        self.current_reward = 0.
        self.total_game = 0

        # flag for target is hiding or not
        self.is_show = True
        self.not_view_count = 0

        #flag for checking stablity
        self.is_stable = False

        #### For Map ####
        self.game_map = np.zeros((self.screen_height, self.screen_width))
        # data for sending dqn
        self.car_map = np.zeros((self.view_height, self.view_width))

        self.is_target_move = True
        self.target_move_count = 0

        self.target_pos = self.Make_target()
        self.obstacles_pos = list()
        for i in range(6) :
            self.obstacles_pos.append(self.Make_obstacle())

        self.view_obstacles_pos = list()
        self.Check_view_obstacle()

        self.Update_game_map()
        self.Update_car_map(self.is_show, self.target_pos, self.obstacles_pos)

        #flag for deciding defeat obstacle
        self.obstacles_num = len(self.view_obstacles_pos)
        self.pre_obstacles_num = len(self.view_obstacles_pos)

        ###for sim ###
        self.speed = 5

        ###for check
        self.print_count = 0


    # target_pos is composed m_row, m_col, h_size
    # obstacle pos is composed row, col , size
    # get data for car image = car_image_process is first 
    def Update_game_map(self) :
        #initial map
        self.game_map = np.zeros((self.screen_height, self.screen_width))
        """
        for i in range(self.screen_height) :
            for j in range(self.screen_width) :
                self.car_map[i, j] = 0                 i dont know what is fast
        """
        
        #target just have middle_point setting

        self.Move_target()
                       
        #obstacle is set with their size
        for obstacle_pos in self.obstacles_pos :
            for i in range(obstacle_pos['size']) :
                for j in range(obstacle_pos['size']) :
                    if (obstacle_pos['row'] + i) < self.screen_height and (obstacle_pos['col'] + j) < self.screen_width :
                        self.game_map[obstacle_pos['row']+i, obstacle_pos['col'] + j] = 2

        #initialize
        if self.target_pos['m_row'] is not -1 and self.target_pos['m_col'] is not -1:
            self.is_show = True
        else :
            self.is_show = False
        

        # if same in obstacle = is_show = false
        for obstacle_pos in self.obstacles_pos :
            for i in range(obstacle_pos['size']) :
                for j in range(obstacle_pos['size']) :
                    if self.target_pos['m_row'] == obstacle_pos['row'] + i :
                        if self.target_pos['m_col'] == obstacle_pos['col'] + j :
                            self.is_show = False
                            print('not view target with a obstacle')

        if (self.target_pos['m_row']) < (int((1/4) * self.screen_height)) or (self.target_pos['m_row']) > (int((3/4) * self.screen_height)) or (self.target_pos['m_col']) < (int((1/4) * self.screen_width)) or (self.target_pos['m_col'] ) > (int((3/4) * self.screen_height)) :
            self.is_show = False
            print('not view target!')

        print(self.target_pos)
        if self.is_show :
            self.game_map[self.target_pos['m_row'], self.target_pos['m_col']] = 1
        

    def Update_car_map(self, is_tartget_show, target_pos, obstacles_pos) :
        self.car_map = np.zeros((self.view_height, self.view_width))
        
        for i in range(self.view_height) :
            for j in range(self.view_width) :
                self.car_map[i, j] = self.game_map[i + int(self.view_height/2), j+int(self.view_width/2)]
        
        self.Check_view_obstacle()

        self.obstacles_num = len(self.view_obstacles_pos)

    #data update
    def Update(self, flag) :
        #Moving Map
        self.Update_game_map()
        self.Update_car_map(self.is_show, self.target_pos, self.obstacles_pos)

        self.Car_move(flag)

        target_reward = self.Score_fit_middle() + self.Score_in_view()
        obstacle_reward = self.Score_defeat_obstacle()

        is_over, die_reward = self.Game_over()

        if is_over :
            reward = die_reward
        else :
            reward = target_reward + obstacle_reward
            self.current_reward += reward

        if self.print_count%10 == 0 :
            self.Data_print()
        
        self.print_count +=1
        #print(is_over)
    
        #time.sleep(3)
        return self.car_map, reward, is_over

    ################ For Movind ################
    def Car_move(self, flag) :
        if flag == 0 :
            self.Move_not()
            print('stable')
        elif flag == 1 :
            self.Move_forward()
            print('move_front')
        elif flag == 2 :
            self. Move_back()
            print('move back')
        elif flag == 3 :
            self.Move_right()
            print('move right')
        elif flag == 4 :
            self.Move_left()
            print('move left')
        else :
            print('Flag_Error')

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
       
        # is a flag
        return action 

    #Moving Signal Send
    def Move_not(self) :
        self.is_stable = True

    def Move_forward(self) : 
        self.is_stable = False

        d = self.Get_distance(self.target_pos, self.middle_point)

        if d < self.speed :
            self.speed = int(d)

        self.target_pos['m_row'] +=self.speed

        remove_count = 0
        for obstacle_pos in self.obstacles_pos :
            obstacle_pos['row'] += self.speed

            if obstacle_pos['row'] < 0 or obstacle_pos['row'] > (4/5)*self.screen_height :
                self.obstacles_pos.remove(obstacle_pos)
                remove_count  +=1
        if remove_count != 0  :
            for i in range(remove_count) :
                self.obstacles_pos.append(self.Make_obstacle())

        self.Check_view_obstacle()

        self.speed = 5
        

    def Move_back(self) :
        self.is_stable = False

        d = self.Get_distance(self.target_pos, self.middle_point)

        if d < self.speed :
            self.speed = int(d)
        
        self.target_pos['m_row'] -= self.speed

        remove_count = 0
        for obstacle_pos in self.obstacles_pos :
            obstacle_pos['row'] -= self.speed

            if obstacle_pos['row'] < 0 or obstacle_pos['row'] > (4/5)*self.screen_height :
                self.obstacles_pos.remove(obstacle_pos)
                remove_count  +=1
        if remove_count != 0  :
            for i in range(remove_count) :
                self.obstacles_pos.append(self.Make_obstacle())
        
        self.Check_view_obstacle()

        self.speed = 5
        

    def Move_right(self) :
        self.is_stable = False

        # TODO : have to move perfect / front and back 
        d = self.Get_distance(self.target_pos, self.middle_point)

        if d < self.speed :
            self.speed = int(d)
        
        self.target_pos['m_col'] -= self.speed

        remove_count = 0
        for obstacle_pos in self.obstacles_pos :
            obstacle_pos['col'] += self.speed

            if obstacle_pos['col'] < 0 or obstacle_pos['col'] > self.screen_width :
                self.obstacles_pos.remove(obstacle_pos)
                remove_count +=1
        if remove_count != 0  :
            for i in range(remove_count) :
                self.obstacles_pos.append(self.Make_obstacle())

        self.Check_view_obstacle()

        self.speed = 5
        

    def Move_left(self) :
        self.is_stable = False
        
        # TODO : have to move perfect / front and back 
        d = self.Get_distance(self.target_pos, self.middle_point)

        if d < self.speed :
            self.speed = int(d)
        
        self.target_pos['m_col'] += self.speed

        remove_count = 0
        for obstacle_pos in self.obstacles_pos :
            obstacle_pos['col'] += self.speed

            if obstacle_pos['col'] < 0 or obstacle_pos['col'] > self.screen_width :
                self.obstacles_pos.remove(obstacle_pos)
                remove_count +=1
        if remove_count != 0  :
            for i in range(remove_count) :
                self.obstacles_pos.append(self.Make_obstacle())
            
        self.Check_view_obstacle()

        self.speed = 5
        
    ################ For Scoring ################
    def Score_fit_middle(self) :
        reward = 0

        if self.middle_point['row'] == self.target_pos['m_row'] and self.middle_point['col'] == self.target_pos['m_col'] :
            print(' fit score ! ')
            reward = 100

        return reward

    def Score_in_view(self) :
        if self.is_show :
            self.not_view_count = 0
            reward = 1
        else :
            reward = -1
            self.not_view_count +=1

        return reward
    def Score_stable(self) :
        if self.is_stable :
            reward = 1

    def Score_defeat_obstacle(self) :
        reward = 0

        if self.pre_obstacles_num < self.obstacles_num :
            reward = (self.obstacles_num - self.pre_obstacles_num) * 5
            print('defeat ob !  reward : ' + str(reward))

        self.pre_obstacles_num = self.obstacles_num

        return reward

    def Game_over(self) : 
        is_over, reward = self.Check_target_out()

        if is_over == False :
            if self.game_map[self.player_point['row'], self.player_point['col']] == 2 :
                print('meet_obstacle !')
                is_over = True
                reward = -100

        if is_over == False :
            if self.not_view_count > 150 :
                is_over = True
                reward = -500

        if is_over :
            self.total_reward +=self.current_reward

        return is_over, reward

    def Reset(self) :
        self.current_reward = 0
        self.total_game += 1

        self.middle_point = { 'row' : int(self.screen_height/2), 'col' : int(self.screen_width/2) }  #goal point
        self.player_point = { 'row' : int((3/4) * self.screen_height), 'col' : int(self.screen_width/2) }

        self.is_show = True
        self.not_view_count = 0

        #flag for checking stablity
        self.is_stable = False

        #### For Map ####
        self.game_map = np.zeros((self.screen_height, self.screen_width))
        # data for sending dqn
        self.car_map = np.zeros((self.view_height, self.view_width))

        self.is_target_move = True
        self.target_move_count = 0

        self.target_pos = self.Make_target()
        self.obstacles_pos = list()
        for i in range(8) :
            self.obstacles_pos.append(self.Make_obstacle())

        self.view_obstacles_pos = list()
        self.Check_view_obstacle()

        self.Update_game_map()
        self.Update_car_map(self.is_show, self.target_pos, self.obstacles_pos)

        #flag for deciding defeat obstacle
        self.obstacles_num = len(self.view_obstacles_pos)
        self.pre_obstacles_num = len(self.view_obstacles_pos)

        self.print_count = 0

        return self.car_map

    ##### For Simulation ####
    ##### Target Setting #####
    def Make_target(self) :
        target = {'m_row' : int(random.randrange(190, 210)), 'm_col' : int(random.randrange(190, 210))}

        return target

    def Move_target(self) :
        if self.target_move_count < 50 :
            if self.is_target_move :
                #print('target_stop')
                self.target_move_count = 0
                self.is_target_move = False
            else :
                self.target_move_count = 0
                #print('target_move')
                self.is_target_move = True

        self.target_move_count +=1

        if self.is_target_move :
            self.target_pos['m_row'] += int(random.randrange(-10, 8))
            self.target_pos['m_col'] += int(random.randrange(-15, 15))
    
    ####obstacle Setting ####
    def Make_obstacle(self) :
        obstacle = {'row' : int(random.randrange(10, 100)) , 'col' : random.choice([int(random.randrange(10, 110)), int(random.randrange(290,390))]), 'size' : int(random.randrange(50, 70))}

        return obstacle

    def Check_view_obstacle(self) :
        #initial view_obstacle
        self.view_obstacles_pos.clear()

        for obstacle_pos in self.obstacles_pos :
            if obstacle_pos['row'] + obstacle_pos['size'] > (1/4)*self.screen_height or obstacle_pos['row'] < (3/4)*self.screen_height :
                if obstacle_pos['col'] + obstacle_pos['size'] > (1/4)*self.screen_width or obstacle_pos['col']  < (3/4)*self.screen_width :
                    self.view_obstacles_pos.append(obstacle_pos)               

    def Check_obstacle_size(self, obstacle_pos) :
        distance = self.Get_distance(self.target_pos, obstacle_pos)

        if distance > 80 :
            return 4
        elif distance > 65 :
            return 10
        elif distance > 40 :
            return 16
        elif distance > 25 :
            return 22
        else :
            return 28 


    def Get_distance(self, target_pos, obstacle_pos) :
        distance = math.sqrt(math.pow(target_pos['m_row'] - obstacle_pos['row'], 2) + math.pow(target_pos['m_col'] - obstacle_pos['col'], 2))

        return distance

    def Check_target_out(self) :
        if self.target_pos['m_row'] < 0 or self.target_pos['m_row'] > self.screen_height :
            print('miss target')
            return True, -500

        if self.target_pos['m_col'] < 0 or self.target_pos['m_col'] > self.screen_width : 
            print('miss target')
            return True, -500
        
        return False, 0

    def Data_print(self) :
        if self.print_count > 1 :
            print('target_pos : ' + str(self.target_pos))
            print('obstacle_pos : ')
            for ob in self.obstacles_pos :
                print(ob)

            print('score : ' + str(self.current_reward))
            print('view _ob _ len' + str(len(self.view_obstacles_pos)))