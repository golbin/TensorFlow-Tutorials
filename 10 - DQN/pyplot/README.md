# Deep Q-network

- 구글의 딥마인드에서 개발한 Deep Q-network (DQN)을 이용하여 Q-learning 을 구현해봅니다.
- 빠른 학습을 위해 게임은 간단한 장애물 피하기 게임을 직접 구현했습니다
  - 화면 출력은 matplotlib 으로 구현하였습니다.
  - OpenAI의 Gym과 거의 동일한 인터페이스로 만들었습니다.

### 파일 설명

- agent.py
  - 게임을 진행하거나 학습시키는 에이전트입니다.
- game.py
  - 게임을 구현해 놓은 파일입니다. 게임의 상태를 화면의 픽셀로 가져오지 않고, 좌표값을 이용하여 계산량을 줄이도록 하였습니다.
- model.py
  - DQN을 구현해 놓은 파일입니다.
  - 논문에서는 CNN 모델을 사용하였지만, 구현을 간단히 하고 성능을 빠르게 하기 위해 기본적인 신경망 모델을 사용합니다.

### 핵심 코드

게임 구현을 위한 다양한 내용들이 들어있어 코드분량이 꽤 많지만, 핵심 내용은 딱 다음과 같습니다.

1. Q_value 를 이용해 얻어온 액션을 수행하고, 해당 액션에 의한 게임의 상태와 리워드를 획득한 뒤, 이것을 메모리에 순차적으로 쌓아둡니다. (model.py/step 함수 참고)
2. 일정 수준 이상의 메모리가 쌓이면, 메모리에 저장된 것들 중 샘플링을 하여 논문의 수식을 이용해 다음처럼 최적화를 수행합니다.
3. Q_value로 예측한 액션으로 상태를 만들어내고, target_Q_value와 비교하는 형식을 취합니다.
  - 학습시 예측을 위한 Q_value와 손실값을 위해 실측값을 계산할 Q_value는 네트웍을 따로 구성하여 계산합니다.
  - target_Q_value를 구하는 타겟 네트웍은 메인 네트웍과 항상 같이 학습시키는 것이 아니라, 메인 네트웍을 일정 횟수 이상 학습시킨 뒤, 정해진 횟수가 지나면 메인 네트웍의 신경망 변수 값들을 타겟 네트웍에 복사하는 방법을 사용합니다.
  - 학습시 타겟을 메인 네트웍에서 구하면, 비교할 대상이 예측한 것과 계속 비슷한 상태로 비교하기 때문에 학습이 안정적으로 이루어지지 않기 때문입니다.

```python
def _build_op(self):
    # DQN 의 손실 함수를 구성하는 부분입니다. 다음 수식을 참고하세요.
    # Perform a gradient descent step on (y_j-Q(ð_j,a_j;θ))^2
    one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
    Q_value = tf.reduce_sum(tf.multiply(self.Q_value, one_hot), axis=1)
    cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
    train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

    return cost, train_op

def get_action(self):
    Q_value = self.session.run(self.Q_value,
                               feed_dict={self.input_X: [self.state]})

    action = np.argmax(Q_value[0])

    return action

def train(self):
    # 게임 플레이를 저장한 메모리에서 배치 사이즈만큼을 샘플링하여 가져옵니다.
    state, next_state, action, reward, terminal = self._sample_memory()

    # 학습시 다음 상태를 만들어 낸 Q value를 입력값으로
    # 타겟 네트웍의 Q value를 실측값으로하여 학습합니다
    Q_value = self.session.run(self.target_Q_value,
                               feed_dict={self.input_X: next_state})

    # DQN 의 손실 함수에 사용할 핵심적인 값을 계산하는 부분입니다. 다음 수식을 참고하세요.
    # if episode is terminates at step j+1 then r_j
    # otherwise r_j + γ*max_a'Q(ð_(j+1),a';θ')
    # input_Y 에 들어갈 값들을 계산해서 넣습니다.
    Y = []
    for i in range(self.BATCH_SIZE):
        if terminal[i]:
            Y.append(reward[i])
        else:
            Y.append(reward[i] + self.GAMMA * np.max(Q_value[i]))

    self.session.run(self.train_op,
                     feed_dict={
                         self.input_X: state,
                         self.input_A: action,
                         self.input_Y: Y
                     })
```

### 결과물

- 상상력을 발휘해주세요. 검정색 배경은 도로, 사각형을 자동차들로 그리고 녹색 사각형을 자율 주행차라고 상상하고 즐겨주시면 감사하겠습니다. :-D
- 180만번 정도의 학습 후 최고의 성능을 내기 시작했으며, 2012 맥북프로 CPU 버전으로 최고 성능을 내는데까지 약 3시간 정도 걸렸습니다.

![게임](screenshot_game.gif)

![텐서보드](screenshot_tensorboard.png)

### 사용법

자가 학습시키기

```
python agent.py --train
```

얼마나 잘 하는지 확인해보기

```
python agent.py
```

텐서보드로 평균 보상값 확인해보기

```
tensorboard --logdir=./logs
```
