# -*- coding: utf-8 -*-
# 텐서보드를 이용하기 위해 각종 변수들을 설정하고 저장하는 방법을 익혀봅니다.

import tensorflow as tf
import numpy as np


data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])


#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# with tf.name_scope 으로 묶은 블럭은 텐서보드에서 한 레이어안에 표현해줍니다
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name="W1")
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name="W2")
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name="W3")
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(model, Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost)

    # tf.summary.scalar 를 이용해 수집하고 싶은 값들을 지정할 수 있습니다.
    tf.summary.scalar('cost', cost)


#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 텐서보드에서 표시해주기 위한 값들을 수집합니다.
merged = tf.summary.merge_all()
# 저장할 그래프와 로그를 저장할 디렉토리를 설정합니다.
writer = tf.summary.FileWriter('./logs', sess.graph)
# 이렇게 저장한 로그는, 학습 후 다음의 명령어를 이용해 웹서버를 실행시킨 뒤
# tensorboard --logdir=./logs
# 다음 주소와 웹브라우저를 이용해 텐서보드에서 확인할 수 있습니다.
# http://localhost:6006

for step in xrange(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    # 적절한 시점에 저장할 값들을 수집하고 저장합니다.
    summary = sess.run(merged, feed_dict={X: x_data, Y:y_data})
    writer.add_summary(summary, step)


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
