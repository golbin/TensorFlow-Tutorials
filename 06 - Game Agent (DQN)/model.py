# -*- coding: utf-8 -*-
# 알파고를 만든 구글의 딥마인드의 논문을 참고한 DQN 모델을 생성합니다.
# 딥마인드의 논문에서는 신경망 모델을 CNN 모델을 사용하지만, 여기서는 간단히 기본적인 신경망 모델을 사용합니다.
# http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html

import tensorflow as tf
import numpy as np
import random
from collections import deque


class DQN:

    # 다음에 취할 액션을 DQN 을 이용해 결정할 시기를 결정합니다.
    # get_action 참고
    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.01
    EXPLORE = 1000.
    # 학습 데이터를 어느정도 쌓은 후, 일정 시간 이후에 학습을 시작하도록 합니다.
    OBSERVE = 100.
    # 학습에 사용할 플레이결과를 얼마나 많이 저장해서 사용할지를 정합니다.
    # (플레이결과 = 게임판의 상태 + 취한 액션 + 리워드 + 종료여부)
    REPLAY_MEMORY = 50000
    # 학습시 사용/계산할 상태값(정확히는 replay memory)의 갯수를 정합니다.
    BATCH_SIZE = 50
    # 과거의 상태에 대한 가중치를 줄이는 역할을 합니다.
    GAMMA = 0.99

    def __init__(self, n_action, n_width, n_height, state):
        self.n_action = n_action
        self.n_width = n_width
        self.n_height = n_height

        self.time_step = 0
        self.epsilon = self.INITIAL_EPSILON
        # 게임의 상태. 게임판의 상태를 말합니다.
        # 학습으로 계산할 상태는 현재 게임판과, 과거 세 번의 게임판, 총 네 가지의 상태를 사용합니다.
        self.state_t = np.stack((state, state, state, state), axis=1)[0]
        # 게임 플레이결과를 저장할 메모리
        self.memory = deque()

        # 게임의 상태를 입력받을 변수
        # [게임 상태의 갯수(현재+과거+과거..), 각 시점의 게임의 상태(게임판의 크기)]
        self.input_state = tf.placeholder(tf.float32, [None, len(self.state_t), self.n_width * self.n_height])
        # 각각의 상태를 만들어낸 액션의 값들입니다.
        self.input_action = tf.placeholder(tf.float32, [None, self.n_action])
        # DQN 의 가장 핵심적인 값이며 Q_action 을 사용하는데 계산할 값 입니다. train 함수를 참고하세요.
        self.input_Y = tf.placeholder(tf.float32, [None])

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.rewards = tf.placeholder(tf.float32, [None])
        tf.summary.scalar('avg.reward', tf.reduce_mean(self.rewards))

        self.Q_value, self.train_op = self.build_model()

        self.saver, self.session = self.init_session()
        self.writer = tf.summary.FileWriter('logs', self.session.graph)
        self.summary = tf.summary.merge_all()

    def init_session(self):
        saver = tf.train.Saver()
        session = tf.InteractiveSession()

        ckpt = tf.train.get_checkpoint_state('model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print "다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print "새로운 모델을 생성하는 중 입니다."
            session.run(tf.global_variables_initializer())

        return saver, session

    def write_logs(self, reward):
        if self.time_step % 100 == 0:
            summary = self.summary.eval(feed_dict={self.rewards: reward})
            self.writer.add_summary(summary, self.global_step.eval())

        if self.time_step % 10000 == 0:
            self.saver.save(self.session, 'model/dqn.ckpt', global_step=self.time_step)

    def build_model(self):
        # 계산 속도와 편의성을 위해 CNN 을 사용하지 않고, input_state 값을 flat 하게 만들어 계산합니다.
        n_input = len(self.state_t) * self.n_width * self.n_height
        state = tf.reshape(self.input_state, [-1, n_input])

        W1 = tf.Variable(tf.truncated_normal([n_input, 128], stddev=0.01))
        b1 = tf.Variable(tf.constant(0.01, shape=[128]))
        L1 = tf.nn.relu(tf.matmul(state, W1) + b1)

        W2 = tf.Variable(tf.truncated_normal([128, 256], stddev=0.01))
        b2 = tf.Variable(tf.constant(0.01, shape=[256]))
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

        W3 = tf.Variable(tf.truncated_normal([256, self.n_action], stddev=0.01))
        b3 = tf.Variable(tf.constant(0.01, shape=[self.n_action]))
        Q_value = tf.matmul(L2, W3) + b3

        # DQN 의 손실 함수를 구성하는 부분입니다. 다음 수식을 참고하세요.
        # Perform a gradient descent step on (y_j-Q(ð_j,a_j;θ))^2
        Q_action = tf.reduce_sum(tf.mul(Q_value, self.input_action), axis=1)
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_action))
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost, global_step=self.global_step)

        return Q_value, train_op

    def train(self):
        # 게임 플레이를 저장한 메모리에서 배치 사이즈만큼을 샘플링하여 가져옵니다.
        minibatch = random.sample(self.memory, self.BATCH_SIZE)

        state = [data[0] for data in minibatch]
        action = [data[1] for data in minibatch]
        reward = [data[2] for data in minibatch]
        next_state = [data[3] for data in minibatch]

        Y = []
        Q_value = self.Q_value.eval(feed_dict={self.input_state: next_state})

        # DQN 의 손실 함수에 사용할 핵심적인 값을 계산하는 부분입니다. 다음 수식을 참고하세요.
        # if episode is terminates at step j+1 then r_j
        # otherwise r_j + γ*max_a'Q(ð_(j+1),a';θ')
        for i in range(0, self.BATCH_SIZE):
            if minibatch[i][4]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * np.max(Q_value[i]))

        self.train_op.run(feed_dict={
            self.input_Y: Y,
            self.input_action: action,
            self.input_state: state
        })

        self.write_logs(reward)

    def step(self, state, action, reward, terminal):
        # 학습데이터로 현재의 상태만이 아닌, 과거의 상태까지 고려하여 계산하도록 하였고,
        # 이 모델에서는 과거 3번 + 현재 = 총 4번의 상태를 계산하도록 하였으며,
        # 새로운 상태가 들어왔을 때, 가장 오래된 상태를 제거하고 새로운 상태를 넣습니다.
        next_state = np.append(self.state_t[1:, :], state, axis=0)
        # 플레이결과, 즉, 액션으로 얻어진 상태와 보상등을 메모리에 저장합니다.
        self.memory.append((self.state_t, action, reward, next_state, terminal))

        # 저장할 플레이결과의 갯수를 제한합니다.
        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        # 일정시간 이상 반복이 이루어진 이후에(데이터가 쌓인 이후에) 학습을 시작합니다.
        if self.time_step > self.OBSERVE:
            self.train()

        self.state_t = next_state
        self.time_step += 1

    def get_action(self, train=False):
        # action 과 Q_value 는 one-hot 벡터를 이용합니다.
        action = np.zeros(self.n_action)

        # 학습 초기에는 액션을 랜덤한 값으로 결정합니다.
        # 이후 학습을 진행하면서 점진적으로 더 많은 결정을 DQN 이 하도록 합니다.
        if train and random.random() <= self.epsilon:
            index = random.randrange(self.n_action)
        else:
            Q_value = self.Q_value.eval(feed_dict={self.input_state: [self.state_t]})[0]
            index = np.argmax(Q_value)

        action[index] = 1

        # 학습이 일정시간 이상 지났을 때부터 입실론 값을 점진적으로 줄이며,
        # 얼마나 단계적으로, 또 얼마나 많이 액션값을 DQN 에 맡길지를 결정하기 위한 로직입니다.
        if self.epsilon > self.FINAL_EPSILON and self.time_step > self.OBSERVE:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE

        return action
