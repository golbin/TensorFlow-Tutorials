# -*- coding: utf-8 -*-
# 신경망의 레이어를 여러개로 구성하여 이제 진짜 딥러닝을 해 봅시다!

import tensorflow as tf
import numpy as np

# 파일에서 자료 읽기
data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')

# csv 자료의 0,1 번째 열(특성)을 x_data 로
# 나머지 열(분류)을 y_data 로 만들어줍니다.
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])


#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 3개의 가중치 행렬 변수로 2개의 히든 레이어를 구성합니다.
# 히든레이어에는 각각 10개, 20의 뉴런이 생기고 다음처럼 연결시킬 것 입니다.
# 2 -> 10 -> 20 -> 3
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))

# 입력값 X 와 변수 W1 을 이용해 첫번째 레이어를
# 첫번째 레이어와 W2 를 이용해 두번째 레이어를 구성합니다.
L1 = tf.nn.relu(tf.matmul(X, W1))
L2 = tf.nn.relu(tf.matmul(L1, W2))

# 마지막으로 아웃풋을 만들기 위해 W3 를 곱해줍니다.
# 비용 함수에 텐서플로우가 제공하는 softmax_cross_entropy_with_logits 함수를 사용하면,
# 출력값에 먼저 softmax 함수를 적용할 필요가 없습니다.
model = tf.matmul(L2, W3)

# 텐서플로우에서 기본적으로 제공되는 크로스 엔트로피 함수를 이용해
# 복잡한 수식을 사용하지 않고도 최적화를 위한 비용 함수를 다음처럼 간단하게 적용할 수 있습니다.
cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(model, Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in xrange(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print (step + 1), sess.run(cost, feed_dict={X: x_data, Y: y_data})


#########
# 결과 확인
# 0: 기타 1: 포유류, 2: 조류
######
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print '예측값:', sess.run(prediction, feed_dict={X: x_data})
print '실제값:', sess.run(target, feed_dict={Y: y_data})

check_prediction = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print '정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data})
