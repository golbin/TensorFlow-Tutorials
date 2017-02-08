# -*- coding: utf-8 -*-
# 신경망 구성을 손쉽게 해 주는 유틸리티 모음인 tensorflow.contrib.layers 를 사용해봅니다.
# 03 - CNN.py 를 재구성한 것이니, 소스를 한 번 비교해보세요.
# 이처럼 TensorFlow 에는 간단하게 사용할 수 있는 다양한 함수와 유틸리티들이 매우 많이 마련되어 있습니다.
# 다만, 처음에는 기본적인 개념에 익숙히지는 것이 좋으므로 이후에도 가급적 기본 함수들을 이용하도록 하겠습니다.

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


#########
# 옵션 설정
######
n_width = 28  # MNIST 이미지의 가로 크기
n_height = 28  # MNIST 이미지의 세로 크기
n_output = 10

#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32, [None, n_width, n_height, 1])
Y = tf.placeholder(tf.float32, [None, n_output])

# 기본적으로 inputs, outputs size, kernel_size 만 넣어주면
# 활성화 함수 적용은 물론, 컨볼루션 신경망을 만들기 위한 나머지 수치들은 알아서 계산해줍니다.
# 특히 Weights 를 계산하는데 xavier_initializer 를 쓰고 있는 등,
# 크게 신경쓰지 않아도 일반적으로 효율적인 신경망을 만들어줍니다.
L1 = tf.contrib.layers.conv2d(X, 32, [3, 3])
L2 = tf.contrib.layers.max_pool2d(L1, [2, 2])
# normalizer_fn 인자를 사용하면 과적합등을 막아주는 normalizer 기법을 간단히 적용할 수 있습니다.
L3 = tf.contrib.layers.conv2d(L2, 64, [3, 3],
                              normalizer_fn=tf.nn.dropout,
                              normalizer_params={'keep_prob': 0.8})
L4 = tf.contrib.layers.max_pool2d(L3, [2, 2])

L5 = tf.contrib.layers.flatten(L4)
L5 = tf.contrib.layers.fully_connected(L5, 256,
                                       normalizer_fn=tf.contrib.layers.batch_norm)
model = tf.contrib.layers.fully_connected(L5, n_output)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print 'Epoch:', '%04d' % (epoch + 1), \
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch)

print '최적화 완료!'


#########
# 결과 확인
######
check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print '정확도:', sess.run(accuracy,
                       feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                  Y: mnist.test.labels})
