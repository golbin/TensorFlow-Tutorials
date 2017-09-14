# 게임 구현과 DQN 모델을 이용해 게임을 실행하고 학습을 진행합니다.
import tensorflow as tf
import numpy as np
import random
import time

from game import Game
from model import DQN


tf.app.flags.DEFINE_boolean("train", False, "학습모드. 게임을 화면에 보여주지 않습니다.")
FLAGS = tf.app.flags.FLAGS

# 최대 학습 횟수
MAX_EPISODE = 10000
# 1000번의 학습마다 한 번씩 타겟 네트웍을 업데이트합니다.
TARGET_UPDATE_INTERVAL = 1000
# 4 프레임마다 한 번씩 학습합니다.
TRAIN_INTERVAL = 4
# 학습 데이터를 어느정도 쌓은 후, 일정 시간 이후에 학습을 시작하도록 합니다.
OBSERVE = 100

# action: 0: 좌, 1: 유지, 2: 우
NUM_ACTION = 3
SCREEN_WIDTH = 6
SCREEN_HEIGHT = 10


def train():
    print('뇌세포 깨우는 중..')
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=False)
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    # 타겟 네트웍을 초기화합니다.
    brain.update_target_network()

    # 다음에 취할 액션을 DQN 을 이용해 결정할 시기를 결정합니다.
    epsilon = 1.0
    # 프레임 횟수
    time_step = 0
    total_reward_list = []

    # 게임을 시작합니다.
    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        # 게임을 초기화하고 현재 상태를 가져옵니다.
        # 상태는 screen_width x screen_height 크기의 화면 구성입니다.
        state = game.reset()
        brain.init_state(state)

        while not terminal:
            # 입실론이 랜덤값보다 작은 경우에는 랜덤한 액션을 선택하고
            # 그 이상일 경우에는 DQN을 이용해 액션을 선택합니다.
            # 초반엔 학습이 적게 되어 있기 때문입니다.
            # 초반에는 거의 대부분 랜덤값을 사용하다가 점점 줄어들어
            # 나중에는 거의 사용하지 않게됩니다.
            if np.random.rand() < epsilon:
                action = random.randrange(NUM_ACTION)
            else:
                action = brain.get_action()

            # 일정 시간이 지난 뒤 부터 입실론 값을 줄입니다.
            # 초반에는 학습이 전혀 안되어 있기 때문입니다.
            if episode > OBSERVE:
                epsilon -= 1 / 1000

            # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
            state, reward, terminal = game.step(action)
            total_reward += reward

            # 현재 상태를 Brain에 기억시킵니다.
            # 기억한 상태를 이용해 학습하고, 다음 상태에서 취할 행동을 결정합니다.
            brain.remember(state, action, reward, terminal)

            if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
                # DQN 으로 학습을 진행합니다.
                brain.train()

            if time_step % TARGET_UPDATE_INTERVAL == 0:
                # 타겟 네트웍을 업데이트 해 줍니다.
                brain.update_target_network()

            time_step += 1

        print('게임횟수: %d 점수: %d' % (episode + 1, total_reward))

        total_reward_list.append(total_reward)

        if episode % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        if episode % 100 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step=time_step)


def replay():
    print('뇌세포 깨우는 중..')
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=True)
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    # 게임을 시작합니다.
    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        while not terminal:
            action = brain.get_action()

            # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

            # 게임 진행을 인간이 인지할 수 있는 속도로^^; 보여줍니다.
            time.sleep(0.3)

        print('게임횟수: %d 점수: %d' % (episode + 1, total_reward))


def main(_):
    if FLAGS.train:
        train()
    else:
        replay()


if __name__ == '__main__':
    tf.app.run()
