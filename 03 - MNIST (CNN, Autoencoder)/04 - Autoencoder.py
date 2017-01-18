# -*- coding: utf-8 -*-
# 대표적인 비감독(Unsupervised) 학습 방법인 Autoencoder 를 사용해봅니다.

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


#########
# 옵션 설정
######
learning_rate = 0.01
training_epoch = 20
batch_size = 100
# 신경망 레이어 구성 옵션
n_hidden_1 = 256  # 첫번째 레이어의 특성 갯수
n_hidden_2 = 128  # 두번째 레이어의 특성 갯수
n_input = 28*28   # 입력값 크기 - 이미지 픽셀수


#########
# 신경망 모델 구성
######
# Y 가 없습니다. 입력값을 Y로 사용하기 때문입니다.
X = tf.placeholder("float", [None, n_input])

# 인코더 레이어와 디코더 레이어의 가중치와 편향 변수를 설정합니다.
# 다음과 같이 이어지는 레이어를 구성하기 위한 값들 입니다.
# encode1 -> encode2 -> decode1 -> decode2
# encode1 에서는 입력값보다 작은 값의 특성치를 갖게 하여 정보를 압축하고,
# 최종적으로 decode2 의 출력을 입력값과 동일한 크기의 특성치를 갖도록 만듭니다.
weights = {
    'encode1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encode2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decode1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decode2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
    'encode1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encode2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decode1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decode2': tf.Variable(tf.random_normal([n_input]))
}

# sigmoid 함수를 이용해 신경망 레이어를 구성합니다.
# sigmoid(X * W + b)
# 인코더 레이어 구성
encode_layer_1 = tf.nn.sigmoid(
                    tf.add(tf.matmul(X, weights['encode1']), biases['encode1']))
encode_layer_2 = tf.nn.sigmoid(
                    tf.add(tf.matmul(encode_layer_1, weights['encode2']), biases['encode2']))

# 디코더 레이어 구성
decode_layer_1 = tf.nn.sigmoid(
                    tf.add(tf.matmul(encode_layer_2, weights['decode1']), biases['decode1']))
decode_layer_2 = tf.nn.sigmoid(
                    tf.add(tf.matmul(decode_layer_1, weights['decode2']), biases['decode2']))


# 예측값을 최종 레이어의 출력값으로 설정합니다.
prediction = decode_layer_2
# Y 값, 즉 예측을 평가하기 위한 실제 값을 입력값으로 설정합니다.
Y = X

cost = tf.reduce_mean(tf.pow(Y - prediction, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(training_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        total_cost += cost_val

    print 'Epoch:', '%04d' % (epoch + 1), \
        'Avg. cost =', '{:.6f}'.format(total_cost / total_batch)

print '최적화 완료!'


#########
# 결과 확인
# 입력값(위쪽)과 모델이 생성한 값(아래쪽)을 시각적으로 비교해봅니다.
######
sample_size = 10
sample_xs = random.sample(mnist.test.images, sample_size)

predicted_samples = sess.run(prediction,
                             feed_dict={X: mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, 10, figsize=(10, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(predicted_samples[i], (28, 28)))

plt.show()
