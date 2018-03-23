import tensorflow as tf
import numpy as np
import random
import time

from Sim import Sim
from dqn import DQN



tf.app.flags.DEFINE_boolean("train", False, "학습모드. 게임을 화면에 보여주지 않습니다.")
FLAGS = tf.app.flags.FLAGS

# number of training
MAX_EPISODE = 1000
# target network update at 1000 times
TARGET_UPDATE_INTERVAL = 100
# train each 4 frames
TRAIN_INTERVAL = 5
# After stack the datas and time, start training (with epsilon decrease)
OBSERVE = 100

# action: 0: stop, 1: front, 2: back, 3: right, 4:left
NUM_ACTION = 5
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
VIEW_WIDTH = 200
VIEW_HEIGHT = 200


def train():
    print('dqn_setting')
    sess = tf.Session()

    game = Sim(SCREEN_WIDTH, SCREEN_HEIGHT)
    brain = DQN(sess, VIEW_WIDTH, VIEW_HEIGHT, NUM_ACTION)

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    # initialize target network
    brain.update_target_network()

    # decide action decision time using dqn after this time
    epsilon = 1.0
    # number of frames
    time_step = 0
    total_reward_list = []

    # Starting
    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        # gmae reset and get state (=poisition data)
        state = game.Reset()
        brain.init_state(state)  # set state of dqn

        while not terminal:
            # choice random action when epsilon < random var, else ( :epilon > random var), choice with dqn
            # Later time, unless use random action
            if np.random.rand() < epsilon:
                action = game.get_action()
            else:
                action = brain.get_action()

            # epsion decrease 
            if episode > OBSERVE:
                epsilon -= 1 / 1000

            # game updates, get data (state / reward / is_gameover)
            state, reward, terminal = game.Update(action)
            total_reward += reward

            # brain save this state
            brain.remember(state, action, reward, terminal)


            # After little time, In interval, do training
            if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
                brain.train()

            # In interval, do training
            if time_step % TARGET_UPDATE_INTERVAL == 0:
                brain.update_target_network()

            time_step += 1


        print('Number of game: %d  Score: %d' % (episode + 1, total_reward))

        f = open('ac.txt', 'rw')

        f.write('Number of game: %d  Score: %d ::\n' % (episode + 1, total_reward))

        f.close()

        total_reward_list.append(total_reward)

        if episode % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        if episode % 100 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step=time_step)


def replay():
    print('dqn_setting')
    sess = tf.Session()

    game = Sim(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=True)
    brain = DQN(sess, VIEW_WIDTH, VIEW_HEIGHT, NUM_ACTION)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    # start game
    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.Reset()
        brain.init_state(state)

        while not terminal:
            action = brain.get_action()
            print('action_choice : ' + str(action))

            # get data
            state, reward, terminal = game.Update(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

            # show the play
            time.sleep(10)

        print('Number of game: %d Score: %d' % (episode + 1, total_reward))


def main(self):
    if FLAGS.train:
        train()
    else:
        replay()


if __name__ == '__main__':
    tf.app.run()