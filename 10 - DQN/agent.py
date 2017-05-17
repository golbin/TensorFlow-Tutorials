# 게임 구현과 DQN 모델을 이용해 게임을 실행하고 학습을 진행합니다.
import tensorflow as tf
import numpy as np
import time

from game import Game
from model import DQN


tf.app.flags.DEFINE_boolean("train", False, "학습모드. 게임을 화면에 보여주지 않습니다.")
FLAGS = tf.app.flags.FLAGS

# action: 0: 좌, 1: 유지, 2: 우
n_action = 3
screen_width = 6
screen_height = 10


def main(_):
    game = Game(screen_width, screen_height, show_game=not FLAGS.train)
    state = game.get_state()
    brain = DQN(n_action, screen_width, screen_height, state)

    while 1:
        game.reset()
        gameover = FLAGS.train

        print(" Avg. Reward: %d, Total Game: %d" % (
                    game.total_reward / game.total_game, game.total_game))

        while not gameover:
            # DQN 모델을 이용해 실행할 액션을 결정합니다.
            action = brain.get_action(FLAGS.train)

            # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
            reward, gameover = game.proceed(np.argmax(action))

            # 위에서 결정한 액션에 따른 현재 상태를 가져옵니다.
            # 상태는 screen_width x screen_height 크기의 화면 구성입니다.
            state = game.get_state()

            # DQN 으로 학습을 진행합니다.
            brain.step(state, action, reward, gameover)

            # 학습모드가 아닌 경우, 게임 진행을 인간이 인지할 수 있는 속도로^^; 보여줍니다.
            if not FLAGS.train:
                time.sleep(0.3)

if __name__ == '__main__':
    tf.app.run()
