import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import time
import random
import math


class Sim :
    def __init__(self, screen_width, screen_height, show_game=False): 
        self.screen_width = screen_width       #Set Camera width == 640
        self.screen_height = screen_height     #Set Camera height == 640

        self.game_xy = np.zeros((self.screen_width, self.screen_height, 2))
        for i in range(self.screen_width) :
            for j in range(self.screen_height) :
                self.game_xy[i, j, 0] = j
                self.game_xy[i, j, 1] = i

        """
        [[0, 0], [1, 0], [2, 0] ...]
        [[0, 1], [1, 1], [2, 1] ...]
        [[0, 2], [1, 2], [2, 2] ...]
        ...
        """ 

        
        self.target_point = {'x' : int(self.screen_width/2), 'y' : int(self.screen_height/2) }  #trace point
        
        self.obstacles = list() # in obstacle = {'x' : 0, 'y' : 0,  's' : 1, 'in" : False}
        for i in range(5) :
            self.obstacles.append(self.make_obstacle())
        #TODO : obstacle initial
        
        self.middle_point = { 'x' : int(self.screen_height/2), 'y' : int(self.screen_height/2) }  #goal point

        # target moving effect
        self.is_target_move = True
        self.speed = 3
        self.degree = -18
        self.is_target_show = True
        self.is_target_bound = True

        # for reward
        self.not_show_time = 0

        self.total_reward = 0.
        self.current_reward = 0.
        self.total_game = 0
        self.show_game = show_game

        if show_game:
            self.fig, self.axis = self._prepare_display()

        self.game_map = np.zeros([self.screen_width, self.screen_height], dtype=int)
        self.view_width = int(screen_width/2)
        self.view_height = int(screen_height/2)
        self.view_map = np.zeros([self.view_width, self.view_height, 2], dtype=int)

        for i in range(0, self.view_width) :
            for j in range(0, self.view_height) :
                self.view_map[i, j, 0] = int(i + self.view_width/2)
                self.view_map[i, j, 1] = int(j + self.view_height/2)

        self.player_point = {'x' : int(self.screen_width/2), 'y' : int((3/2) * self.view_height)}

        # self.left_rotation_map = np.zeros([self.screen_width, self.screen_height, 2], dtype=int)
        # for i in range(screen_width) :
        #     for j in range(screen_height) :
        #         self.left_rotation_map[i, j , 0] = j - int(screen_width/2)
        #         self.left_rotation_map[i, j , 1] = -i + self.screen_height
        # for i in range(screen_width) :
        #     for j in range(screen_height) :
        #         self.left_rotation_map[i ,j, 0] = int(math.cos(math.pi/self.degree)*self.left_rotation_map[i, j, 0] - math.sin(math.pi/self.degree)*self.left_rotation_map[i, j, 1])
        #         self.left_rotation_map[i ,j, 1] = int(math.sin(math.pi/self.degree)*self.left_rotation_map[i, j, 0] + math.cos(math.pi/self.degree)*self.left_rotation_map[i, j, 1])
        # for i in range(screen_width) :
        #     for j in range(screen_height) :
        #         self.left_rotation_map[i, j , 0] = self.left_rotation_map[i, j , 0] + int(screen_width/2)
        #         self.left_rotation_map[i, j , 1] = -( self.left_rotation_map[i, j , 1] - self.screen_height) # TODO : it is right????
        # # TODO : Tanspos -> rotation -> re_transpos

        # self.right_rotation_map = np.zeros([self.screen_width, self.screen_height, 2], dtype=int)
        # for i in range(self.screen_width) :
        #     for j in range(self.screen_height) :
        #         self.right_rotation_map[i, j , 0] = j - int(screen_width/2)
        #         self.right_rotation_map[i, j , 1] = -i + self.screen_height
        # for i in range(screen_width) :
        #     for j in range(screen_height) :
        #         self.right_rotation_map[i ,j, 0] = int(math.cos(-math.pi/self.degree)*self.right_rotation_map[i, j, 0] - math.sin(-math.pi/self.degree)*self.right_rotation_map[i, j, 1])
        #         self.right_rotation_map[i ,j, 1] = int(math.sin(-math.pi/self.degree)*self.right_rotation_map[i, j, 0] + math.cos(-math.pi/self.degree)*self.right_rotation_map[i, j, 1])
        # for i in range(screen_width) :
        #     for j in range(screen_height) :
        #         self.right_rotation_map[i, j , 0] = self.right_rotation_map[i, j , 0] + int(screen_width/2)
        #         self.right_rotation_map[i, j , 1] = -(self.right_rotation_map[i, j , 1] - self.screen_height) # TODO : it is right????

        print('player_point : ( ' + str(self.player_point['x']) + ' , ' + str(self.player_point['y']) +' )')
        print('target_point : ( ' + str(self.target_point['x']) + ' , ' + str(self.target_point['y']) +' )')

        print('Obstacle info')
        ttt = 0
        for obstacle in self.obstacles :
            print('obtacle_' +str(ttt) + " : ( " + str(obstacle['x']) + ' : ' + str(obstacle['y']) + " )")
            ttt +=1

        #for checking data
        time.sleep(10)

        # f = open('./right.txt', 'w')
        # for i in range(self.screen_width) :
        #     for j in range(self.screen_height) :
        #         f.write(str(self.right_rotation_map[i, j]) +  ", ")
        #     f.write('\n')
        # f.close()

        # f = open('./left.txt', 'w')
        # for i in range(self.screen_width) :
        #     for j in range(self.screen_height) :
        #         f.write(str(self.left_rotation_map[i, j]) +', ')
        #     f.write('\n')
        # f.close()


    
    #Target is random move
    def Target_move(self) :
        if self.is_target_move :
            self.target_point['x'] += random.randrange(-2, 2) * self.speed
            self.target_point['y'] += random.randrange(-2, 1) * self.speed

        if self.target_point['x'] < 0 or self.target_point['y'] < 0 or self.target_point['x'] > self.screen_width or self.target_point['y'] > self.screen_height :
            self.is_target_show = False
            self.is_target_show = False

        # TODO : if target disappear (out of window range or to backside of obstacle) -> have to arrage model

    
    def Move_right(self) : 
        self.target_point['x'] = (int(math.cos(math.pi/self.degree) * (self.target_point['x'] - int(self.screen_width/2)) - math.sin(math.pi/self.degree) * (-self.target_point['y'] + self.screen_height) )+int(self.screen_width/2))
        self.target_point['y'] = int(-((math.sin(math.pi/self.degree) * (self.target_point['x'] - int(self.screen_width/2)) + math.cos(math.pi/self.degree) * (-(self.target_point['y'] + self.screen_height))) - self.screen_height))

        if self.target_point['x'] < 0 or self.target_point['y'] < 0 or self.target_point['x'] > self.screen_width or self.target_point['y'] > self.screen_height :
            self.is_target_show = False
            self.is_target_show = False        

        for obstacle in self.obstacles :
            obstacle['x'] = (int(math.cos(math.pi/self.degree) * (obstacle['x'] - int(self.screen_width/2)) - math.sin(math.pi/self.degree) * (-obstacle['y'] + self.screen_height) )+int(self.screen_width/2))
            obstacle['y'] = int(-((math.sin(math.pi/self.degree) * (obstacle['x'] - int(self.screen_width/2)) + math.cos(math.pi/self.degree) * (-(obstacle['y'] + self.screen_height))) - self.screen_height))

            if obstacle['x'] > self.screen_width or obstacle['y'] > self.screen_height or obstacle['x'] < 0 or obstacle['y'] < 0 :
                self.obstacles.remove(obstacle)
                self.obstacles.append(self.make_obstacle())

            
            
            obstacle['in'] = self.check_in(obstacle)
            obstacle['s'] = self.obstacle_size(obstacle)


    def Move_left(self) :
        self.target_point['x'] = int(math.cos(-math.pi/self.degree) * (self.target_point['x'] - int(self.screen_width/2)) - math.sin(-math.pi/self.degree) * (-self.target_point['y']) )+int(self.screen_width/2)
        self.target_point['y'] = int(-((math.sin(-math.pi/self.degree) * (self.target_point['x'] - int(self.screen_width/2)) + math.cos(-math.pi/self.degree) * (-(self.target_point['y'] + self.screen_height))) - self.screen_height))

        if self.target_point['x'] < 0 or self.target_point['y'] < 0 or self.target_point['x'] > self.screen_width or self.target_point['y'] > self.screen_height :
            self.is_target_show = False
            self.is_target_show = False

        for obstacle in self.obstacles :
            obstacle['x'] = (int(math.cos(-math.pi/self.degree) * (obstacle['x'] - int(self.screen_width/2)) - math.sin(-math.pi/self.degree) * (-obstacle['y'] + self.screen_height) )+int(self.screen_width/2))
            obstacle['y'] = int(-((math.sin(-math.pi/self.degree) * (obstacle['x'] - int(self.screen_width/2)) + math.cos(-math.pi/self.degree) * (-(obstacle['y'] + self.screen_height))) - self.screen_height))

            if obstacle['x'] > self.screen_width or obstacle['y'] > self.screen_height or obstacle['x'] < 0 or obstacle['y'] < 0 :
                self.obstacles.remove(obstacle)
                self.obstacles.append(self.make_obstacle())
            
            obstacle['in'] = self.check_in(obstacle)
            obstacle['s'] = self.obstacle_size(obstacle)


    def Move_front(self) :
        # all point move
        for obstacle in self.obstacles :
            obstacle['y'] +=1 * self.speed

            if  obstacle['x'] > self.screen_width or obstacle['y'] > self.screen_height or obstacle['x'] < 0 or obstacle['y'] < 0 :
                self.obstacles.remove(obstacle)
                self.obstacles.append(self.make_obstacle())

            obstacle['in'] = self.check_in(obstacle)
            obstacle['s'] = self.obstacle_size(obstacle)

        self.target_point['y'] +=1

        if self.target_point['x'] < 0 or self.target_point['y'] < 0 or self.target_point['x'] > self.screen_width or self.target_point['y'] > self.screen_height :
            self.is_target_show = False
            self.is_target_show = False


    def Move_back(self) :
        # all point move
        for obstacle in self.obstacles :
            obstacle['y'] -=1


            if  obstacle['x'] > self.screen_width or obstacle['y'] > self.screen_height or obstacle['x'] < 0 or obstacle['y'] < 0 :
                self.obstacles.remove(obstacle)
                self.obstacles.append(self.make_obstacle())

            obstacle['in'] = self.check_in(obstacle)
            obstacle['s'] = self.obstacle_size(obstacle)

        self.target_point['y'] -=1

        if self.target_point['x'] < 0 or self.target_point['y'] < 0 or self.target_point['x'] > self.screen_width or self.target_point['y'] > self.screen_height :
            self.is_target_show = False
            self.is_target_show = False


    def Check_goal(self) :
        reward = 0

        if self.middle_point['x'] == self.target_point['x'] and self.middle_point['y'] == self.target_point['y'] :
            print('point meet')
            reward = 10

        return reward

    def Check_show(self) :
        if self.is_target_bound :
            reward = 1
        else :
            reward = -1
        
        if not self.is_target_show :
            self.not_show_time += 1

        return reward




    def Check_avoid(self) :
        reward = 0 

        for obstacle in self.obstacles :
            if obstacle['in'] :
                if self.get_distance(obstacle['x'], obstacle['y'], player_point['x'], player_point['y']) > 20 :  #check distace to far away -> change in 10 / apart 20
                    reward = 1  # check reward point

        return reward



    def Update_car(self, flag) : 
        if flag == 0 :
            pass 
        elif flag == 1:
            self.Move_front()
        elif flag == 2 :
            self.Move_back()
        elif flag == 3 :
            self.Move_left()
        elif flag == 4 :
            self.Move_right()
        else :
            pass
                        
    def Get_state(self) :
        for i in range(self.screen_width) :
            for j in range(self.screen_height) :
                self.game_map[i, j] = 0        
        
        state = np.zeros([self.view_width, self.view_height])
        
        for obstacle in self.obstacles :
            for k in range(-obstacle['s'], obstacle['s']) :
                if obstacle['x']+k < self.screen_width and obstacle['x']+k >= 0 and obstacle['y']+k < self.screen_height and obstacle['y']+k >= 0:
                    self.game_map[int(obstacle['x']+k), int(obstacle['y']+k)] = 2

        if self.is_target_show :
            if self.game_map[int(self.target_point['x']), int(self.target_point['y'])] == 2 :
                # TODO : Have to change value
                self.is_target_show = False
                self.is_target_bound = False
            else :
                self.game_map[int(self.target_point['x']), int(self.target_point['y'])] = 1
                self.is_target_show = True

        for i in range(self.view_width) :
            for j in range(self.view_height) :
                #print(self.view_map[i, j, 0])
                state[i, j] = self.game_map[int(self.view_map[i, j, 0]), int(self.view_map[i, j, 1])]

                # TODO : have change value

        if self.target_point['x'] < int((3/2) * self.view_width) and self.target_point['x'] > int((1/2) * self.view_width) :
            if self.target_point['y'] < int((3/2) * self.view_height) and self.target_point['y'] > int((1/2) * self.view_height) :
                self.is_target_bound = True

        return state
        
    # if meet the obstacle, it is die
    def Is_game_over(self) :
        if self.game_map[self.player_point['x'], self.player_point['y']] == 2 :
            self.total_reward += self.current_reward
            return True

        if self.not_show_time > 100 :
            return True, -1000

        return False, -200            


    def Update(self, flag) :
        # car and target move
        self.Target_move()
        self.Update_car(flag)

        stable_reward = self.Check_goal()
        avoid_reward = self.Check_avoid()   #TODO : avoid obstacle point
        show_reward = self.Check_show()
        is_game_over, score = self.Is_game_over()

        if is_game_over : 
            reward = score
        else :
            reward = stable_reward + avoid_reward + show_reward
            self.current_reward += reward

        #print('target_point : ')
        #print(self.target_point)
        #print(self.not_show_time)

        if self.show_game :
            self._draw_screen()
            print('player_point : ( ' + str(self.player_point['x']) + ' , ' + str(self.player_point['y']) +' )')
            print('target_point : ( ' + str(self.target_point['x']) + ' , ' + str(self.target_point['y']) +' )')

            print('Obstacle info')
            ttt = 0
            for obstacle in self.obstacles :
                print('obtacle_' +str(ttt) + " : ( " + str(obstacle['x']) + ' : ' + str(obstacle['y']) + " )")
                ttt +=1


        #print('target_point["x"] : ' + str(self.target_point['x']))

        return self.Get_state(), reward, is_game_over
        #TODO : retrun reward and something


    #TODO : show display
    #TODO : Obstacle Initial
    #TODO : Obstacle update
    #TODO : how to count the reward when avoid obstacles

    def Reset(self) :
        self.current_reward = 0
        self.total_game += 1

        self.player_point = {'x' : int(self.screen_width/2), 'y' : int((3/2) * self.view_height)}
        self.target_point = {'x' : int(self.screen_width/2), 'y' : int(self.screen_height/2)}  #trace point
        self.obstacles = list()    #using obstacle class?
        for i in range(5) :
            self.obstacles.append(self.make_obstacle())
        self.middle_point = { 'x' : int(self.screen_height/2), 'y' : int(self.screen_height/2)}  #goal point
        #if game over = data reset
        #TODO : obstacle initial
        self.not_show_time = 0
        self.is_target_bound = True
        self.is_target_show = True

        return self.Get_state()  # Why???


    def _draw_screen(self):
        title = " Avg. Reward: %d Reward: %d Total Game: %d" % (self.total_reward / self.total_game, self.current_reward, self.total_game)

        #self.axis.clear()
        self.axis.set_title(title, fontsize=12)
        #print('title')

        # road = patches.Rectangle((self.road_left - 1, 0),
        #                          self.road_width + 1, self.screen_height,
        #                          linewidth=0, facecolor="#333333")
        # 자동차, 장애물들을 1x1 크기의 정사각형으로 그리도록하며, 좌표를 기준으로 중앙에 위치시킵니다.
        # 자동차의 경우에는 장애물과 충돌시 확인이 가능하도록 0.5만큼 아래쪽으로 이동하여 그립니다.
        if self.target_point['x'] < self.screen_width and self.target_point['x'] > 0 and self.target_point['y'] < self.screen_height and self.target_point['y'] > 0 :
            target = patches.Rectangle((self.target_point["x"] - 3, self.target_point["y"] - 3), 6, 6, linewidth=0, facecolor="#FF5733")
            self.axis.add_patch(target)
            #print(target)
            
        for obstacle in self.obstacles :
            if obstacle['x'] - obstacle['s'] < 0 :
                x = 0
            else :
                x = obstacle['x'] - obstacle['s']/2

            if obstacle['y'] - obstacle['s'] < 0 :
                y = 0
            else :
                y = obstacle['y'] - obstacle['s']/2                

            if obstacle['x'] + obstacle['s'] > self.screen_width :
                x_ = self.screen_width - obstacle['x']
            else :
                x_ = obstacle['s']/2                

            if obstacle['y'] + obstacle['s'] > self.screen_height :
                y_ = self.screen_height - obstacle['y']
            else :
                y_ = obstacle['s']/2                
            
            ob = patches.Rectangle((x, y), x_, y_, linewidth=0, facecolor="#00FF00")
            #print(x)
            #print(y)
            #print(x_)
            #print(y_)
            self.axis.add_patch(ob)

        self.fig.canvas.draw()
        # 게임의 다음 단계 진행을 위해 matplot 의 이벤트 루프를 잠시 멈춥니다.
        plt.pause(0.0001)

    def _prepare_display(self):
        """게임을 화면에 보여주기 위해 matplotlib 으로 출력할 화면을 설정합니다."""
        fig, axis = plt.subplots(figsize=(10, 10))
        fig.set_size_inches(10, 10)
        # 화면을 닫으면 프로그램을 종료합니다.
        fig.canvas.mpl_connect('close_event', exit)
        plt.axis((0, self.screen_width, 0, self.screen_height))
        plt.tick_params(top='off', right='off',
                        left='off', labelleft='off',
                        bottom='off', labelbottom='off')

        plt.draw()
        # 게임을 진행하며 화면을 업데이트 할 수 있도록 interactive 모드로 설정합니다.
        plt.ion()
        plt.show()

        print('pre_pal_end')
        return fig, axis



# for easy work 
    def get_distance(self, x_1, y_1, x_2, y_2) :
        distance = math.sqrt(math.pow(x_1 - x_2, 2) + math.pow(y_1 - y_2, 2))

        return distance

    def make_obstacle(self) :
        obstacle = {'x' : random.randrange(int(self.screen_width*(1/10)), int(self.screen_width*(9/10))+1), 'y': self.screen_height*(2/10), 's' : random.randrange(10, 20), 'in' : False}

        return obstacle

    def check_in(self, obstacle):
        if self.get_distance(self.player_point['x'], self.player_point['y'], obstacle['x'], obstacle['y']) < 15 :
            return True

        return False

    def obstacle_size(self, obstacle) :
        var =  self.get_distance(obstacle['x'], obstacle['y'], self.player_point['x'], self.player_point['y'])

        if var > 30 :
            return 10
        elif var > 20 :
            return 14
        elif var > 10 :
            return 17
        else :
            return 20

    def  get_action(self) : 
        if self.target_point['x'] > self.middle_point['x'] :
            if self.target_point['y'] < self.middle_point['y'] :
                temp_action = [0, 1, 4]
                action = random.choice(temp_action)
            else :
                temp_action = [0, 2, 4]
                action = random.choice(temp_action)

        elif self.target_point['x'] < self.middle_point['x'] :
            if self.target_point['y'] < self.middle_point['y'] :
                temp_action = [0, 1, 3]
                action = random.choice(temp_action)
            else : 
                temp_action = [0, 2, 3]
                action = random.choice(temp_action)
        else : 
            if self.target_point['y'] < self.middle_point['y'] :
                temp_action = [0, 1]
                action = random.choice(temp_action)
            elif self.target_point['y'] > self.middle_point['y'] : 
                temp_action = [0, 2]
                action = random.choice(temp_action)
            else : 
                action = 0

        return action
