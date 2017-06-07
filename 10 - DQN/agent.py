import gym
import time
import threading
import numpy as np
import tensorflow as tf
from scipy.misc import imresize

from brain import Brain


# 이미지 사이즈를 줄이고, 흑백 사진으로 만듭니다.
def preprocess(screen, width, height):
    # 완전한 그레이스케일이 아닌 적당한 수준의 흑백으로 만듭니다.
    gray = screen.astype('float32').mean(2)
    # 이미지를 리사이즈하고 수치를 보정합니다. 0~1.0 사이의 값으로 만듭니다.
    processed = imresize(gray, (width, height)).astype('float32') * (1. / 255)

    return processed


def trainer(brain):
    # 컴퓨터의 성능에 따라 학습 빈도를 적절히 제어 할 필요가 있습니다.
    # 원래는 게임 몇 프레임마다 학습을 한 번씩 하게 해야 하지만,,
    # 일반 컴퓨터/CPU로 학습을 시키는 경우 학습 속도가 느려 학습과정을 보기 어려워
    # 적당한 미니배치 사이즈로 계속 학습을 시키도록 했습니다.
    # 컴퓨터가 빠르거나 여러개의 GPU를 사용할 수 있다면
    # 게임 안에 학습 과정을 넣거나, 게임을 여러개를 한 번에 돌리는 방법을 써봐도 좋겠습니다.
    steps = 1
    while True:
        if len(brain.memory) > 5000:
            if steps == 1:
                print('Trainer: 학습 시작!')

            brain.train()
            steps += 1
        else:
            time.sleep(1)

        if steps % 100 == 0:
            print('Trainer: 타겟 네트웍 업데이트. Step: {}'.format(steps))
            brain.update_target_network()


MAX_EPISODE = 9999999
SCREEN_WIDTH = 84
SCREEN_HEIGHT = 84
ENV_NAME = 'Breakout-v0'
# ENV_NAME = 'Freeway-v0'
# ENV_NAME = 'Pong-v0'

env = gym.make(ENV_NAME)

sess = tf.Session()

print('뇌세포 깨우는 중..')
brain = Brain(session=sess,
              width=SCREEN_WIDTH, height=SCREEN_HEIGHT,
              n_action=env.action_space.n)

sess.run(tf.global_variables_initializer())

# 타겟 네트웍을 초기화합니다.
brain.update_target_network()

# 학습을 시키는 trainer 함수를 쓰레드로 돌립니다. 게임과 학습을 동시에 진행합니다.
# 컴퓨터가 빠르거나 반대로 너무 느리면 쓰레드 대신 게임 반복문 안에서 순차적으로 학습시키는 것이 좋을 수 있습니다.
train_thread = threading.Thread(target=trainer, args=[brain])
train_thread.start()

# 게임을 시작합니다.
for episode in range(MAX_EPISODE):
    terminal = False
    total_reward = 0.
    # 다음에 취할 액션을 DQN 을 이용해 결정할 시기를 결정합니다.
    # 초반에는 액션을 랜덤값을 이용합니다. 아직 학습이 되지 않았기 때문입니다.
    # refer: https://github.com/hunkim/ReinforcementZeroToAll/
    epsilon = 1. / ((episode / 100) + 1)

    # 게임 상태를 초기화합니다.
    state = env.reset()
    state = preprocess(state, SCREEN_WIDTH, SCREEN_HEIGHT)
    brain.init_state(state)

    while not terminal:
        # 학습을 한 번도 하지 않았을 때와 입실론이 랜덤값보다 작은 경우에는 랜덤한 액션을 선택합니다.
        # 위의 수식에 의하면 랜덤값을 사용하는 빈도가 점점 줄어들다가
        # 1000번 정도의 에피소드가 지나면 랜덤값을 거의 사용하지 않게됩니다.
        if episode < 10 or np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = brain.get_action()

        state, reward, terminal, info = env.step(action)
        total_reward += reward

        if terminal:
            reward = -1

        # 현재 상태를 Brain에 기억시킵니다.
        # 기억한 상태를 이용해 학습하고, 다음 상태에서 취할 행동을 결정합니다.
        state = preprocess(state, SCREEN_WIDTH, SCREEN_HEIGHT)
        brain.remember(state, action, reward, terminal)

        # 화면을 그립니다.
        env.render()

    print('게임횟수: {} 점수: {}'.format(episode + 1, total_reward))

    # 에피소드 10회가 지나면 매 회 3번의 샘플링 학습을 시킵니다.
    # 학습을 쓰레드로 동시에 시키지 않고 에피소드가 끝난 후 시키려면 이 코드를 사용하세요.
    # if episode > 9:
    #     for count in range(3):
    #         brain.train()
    #
    #     # 타겟 네트웍을 업데이트 해 줍니다.
    #     brain.update_target_network()
