# -*- coding: utf-8 -*-
# 과적합 방지 기법 중 하나인 Dropout 을 사용해봅니다.

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))

L1 = tf.nn.relu(tf.matmul(X, W1))
# 텐서플로우에 내장된 함수를 이용하여 dropout 을 적용합니다.
# 함수에 적용할 레이어와 확률만 넣어주면 됩니다. 겁나 매직!!
L1 = tf.nn.dropout(L1, 0.8)
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, 0.8)
model = tf.matmul(L2, W3)

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
        sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})

    print 'Epoch:', '%04d' % (epoch + 1), \
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch)

print '최적화 완료!'


#########
# 결과 확인
######
check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print '정확도:', sess.run(accuracy,
                       feed_dict={X: mnist.test.images,
                                  Y: mnist.test.labels})
