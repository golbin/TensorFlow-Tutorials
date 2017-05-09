# -*- coding: utf-8 -*-
# 다중 레이어의 RNN 과 더 효율적인 RNN 학습을 위해 텐서플로우에서 제공하는 Dynamic RNN 을 사용해봅니다.

import tensorflow as tf
import numpy as np


num_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
num_dic = {n: i for i, n in enumerate(num_arr)}
dic_len = len(num_dic)

# 크기가 다른 순차열을 같이 학습시켜봅니다.
# 123 -> X, 4 -> Y
# 12 -> X, 3 -> Y
seq_data = ['1234', '2345', '3456', '4567', '5678', '6789', '7890']
seq_data2 = ['123', '234', '345', '456', '567', '678', '789', '890']


def one_hot_seq(seq_data):
    x_batch = []
    y_batch = []
    for seq in seq_data:
        x_data = [num_dic[n] for n in seq[:-1]]
        y_data = num_dic[seq[-1]]
        x_batch.append(np.eye(dic_len)[x_data])
        # 이 예제에서 사용할 손실 함수인 sparse_softmax_cross_entropy_with_logits 는
        # one-hot 인코딩을 사용하지 않으므로 index 를 그냥 넘겨주면 됩니다.
        y_batch.append([y_data])

    return x_batch, y_batch


#########
# 옵션 설정
######
n_input = 10
n_classes = 10
n_hidden = 128
# RNN 셀을 다중 레이어로 사용해봅니다.
n_layers = 3


#########
# 신경망 모델 구성
######
# 다양한 길이의 시퀀스를 다루기 위해 time steps 크기를 None 으로 둡니다.
# [batch size, time steps, input size]
X = tf.placeholder(tf.float32, [None, None, n_input])
# 비용함수에 sparse_softmax_cross_entropy_with_logits 을 사용하므로
# 출력값과의 계산을 위한 원본값의 형태는 다음과 같다.
# [batch size, time steps]
Y = tf.placeholder(tf.int32, [None, 1])

W = tf.Variable(tf.random_normal([n_hidden, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

# tf.nn.dynamic_rnn 옵션에서 time_major 값을 True 로 설정하면
# 입력값을 적게 변형해도 되므로 다음과 같이 간단하게 사용할 수 있습니다.
# [batch_size, n_steps, n_input] -> Tensor[n_steps, batch_size, n_input]
X_t = tf.transpose(X, [1, 0, 2])

# RNN 셀을 생성합니다.
# 다중 레이어와 과적합 방지를 위한 Dropout 기법을 사용합니다.
def cell():
    rnn_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
    rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=0.5)
    return rnn_cell

# 다중 레이어 구성을 다음과 같이 아주 간단하게 만들 수 있습니다.
cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(n_layers)])

# tf.nn.dynamic_rnn 함수를 이용해 순환 신경망을 만듭니다.
outputs, states = tf.nn.dynamic_rnn(cell, X_t, dtype=tf.float32, time_major=True)

# logits 는 one-hot 인코딩을 사용합니다.
logits = tf.matmul(outputs[-1], W) + b
# sparse_softmax_cross_entropy_with_logits 함수의 labels 는
# one-hot 인코딩을 사용하지 않기 때문에, 1차원 배열로 넘겨줍니다. (time step 이 1이기 때문)
# (logits 의 랭크가 2 이므로 [batch_size, n_classes])
labels = tf.reshape(Y, [-1])


cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

train_op = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(cost)


#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_batch, y_batch = one_hot_seq(seq_data)
x_batch2, y_batch2 = one_hot_seq(seq_data2)

for epoch in range(30):
    _, loss4 = sess.run([train_op, cost], feed_dict={X: x_batch, Y: y_batch})
    _, loss3 = sess.run([train_op, cost], feed_dict={X: x_batch2, Y: y_batch2})

    print ('Epoch:', '%04d' % (epoch + 1), 'cost =', \
        'bucket[4] =', '{:.6f}'.format(loss4), \
        'bucket[3] =', '{:.6f}'.format(loss3))

print ('최적화 완료!')


#########
# 결과 확인
######
# 테스트 데이터를 받아 결과를 예측해보는 함수
def prediction(seq_data):
    prediction = tf.cast(tf.argmax(logits, 1), tf.int32)
    prediction_check = tf.equal(prediction, labels)
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

    x_batch_t, y_batch_t = one_hot_seq(seq_data)
    real, predict, accuracy_val = sess.run([labels, prediction, accuracy],
                                           feed_dict={X: x_batch_t, Y: y_batch_t})

    print ("\n=== 예측 결과 ===")
    print ('순차열:', seq_data)
    print ('실제값:', [num_arr[i] for i in real])
    print ('예측값:', [num_arr[i] for i in predict])
    print ('정확도:', accuracy_val)


# 학습 데이터에 있던 시퀀스로 테스트
seq_data_test = ['123', '345', '789']
prediction(seq_data_test)

seq_data_test = ['1234', '2345', '7890']
prediction(seq_data_test)

# 학습시키지 않았던 시퀀스를 테스트 해 봅니다.
seq_data_test = ['23', '78', '90']
prediction(seq_data_test)

seq_data_test = ['12345', '34567', '67890']
prediction(seq_data_test)
