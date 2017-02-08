# -*- coding: utf-8 -*-
# 자연어 처리나 음성 처리 분야에 많이 사용되는 RNN 의 기본적인 사용법을 익힙니다.
# 1 부터 0 까지 순서대로 숫자를 예측하여 세는 모델을 만들어봅니다.

import tensorflow as tf
import numpy as np


num_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
# one-hot 인코딩을 사용하기 위해 연관 배열을 만듭니다.
# {'1': 0, '2': 1, '3': 2, ..., '9': 9, '0', 10}
num_dic = {n: i for i, n in enumerate(num_arr)}
dic_len = len(num_dic)

# 다음 배열은 입력값과 출력값으로 다음처럼 사용할 것 입니다.
# 123 -> X, 4 -> Y
# 234 -> X, 5 -> Y
seq_data = ['1234', '2345', '3456', '4567', '5678', '6789', '7890']


# 위의 데이터에서 X,Y 값을 뽑아 one-hot 인코딩을 한 뒤 배치데이터로 만드는 함수
def one_hot_seq(seq_data):
    x_batch = []
    y_batch = []
    for seq in seq_data:
        # 여기서 생성하는 x_data 와 y_data 는
        # 실제 숫자가 아니라 숫자 리스트의 인덱스 번호 입니다.
        # [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5] ...
        x_data = [num_dic[n] for n in seq[:-1]]
        # 3, 4, 5, 6...10
        y_data = num_dic[seq[-1]]
        # one-hot 인코딩을 합니다.
        # if x_data is [0, 1, 2]:
        # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
        x_batch.append(np.eye(dic_len)[x_data])
        # if 3: [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
        y_batch.append(np.eye(dic_len)[y_data])

    return x_batch, y_batch


#########
# 옵션 설정
######
# 입력값 크기. 10개의 숫자에 대한 one-hot 인코딩이므로 10개가 됩니다.
# 예) 3 => [0 0 1 0 0 0 0 0 0 0 0]
n_input = 10
# 타입 스텝: [1 2 3] => 3
# RNN 을 구성하는 시퀀스의 갯수입니다.
n_steps = 3
# 출력값도 입력값과 마찬가지로 10개의 숫자로 분류합니다.
n_classes = 10
# 히든 레이어의 특성치 갯수
n_hidden = 128


#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32, [None, n_steps, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

W = tf.Variable(tf.random_normal([n_hidden, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

# RNN 학습을 위한 함수인 tf.nn.rnn 을 사용하기 위해 차원 구성을 변경합니다.
# [batch_size, n_steps, n_input]
#    -> Tensor[n_steps, batch_size, n_input]
X_t = tf.transpose(X, [1, 0, 2])
#    -> Tensor[n_steps*batch_size, n_input]
X_t = tf.reshape(X_t, [-1, n_input])
#    -> [n_steps, Tensor[batch_size, n_input]]
X_t = tf.split(0, n_steps, X_t)

# RNN 셀을 생성합니다.
# 다음 함수들을 사용하면 다른 구조의 셀로 간단하게 변경할 수 있습니다
# BasicRNNCell,BasicLSTMCell,GRUCell
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

# tf.nn.rnn 함수를 이용해 순환 신경망을 만듭니다.
# 역시 겁나 매직!!
outputs, states = tf.nn.rnn(cell, X_t, dtype=tf.float32)

# 손실 함수 작성을 위해 출력값을 Y 와 같은 형태의 차원으로 재구성합니다
logits = tf.matmul(outputs[-1], W) + b

cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, Y))

train_op = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(cost)


#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_batch, y_batch = one_hot_seq(seq_data)

for epoch in range(10):
    _, loss = sess.run([train_op, cost], feed_dict={X: x_batch, Y: y_batch})

    # 학습하는 동안 예측값의 변화를 출력해봅니다.
    print sess.run(tf.argmax(logits, 1), feed_dict={X: x_batch, Y: y_batch})
    print sess.run(tf.argmax(Y, 1), feed_dict={X: x_batch, Y: y_batch})

    print 'Epoch:', '%04d' % (epoch + 1), \
        'cost =', '{:.6f}'.format(loss)

print '최적화 완료!'


#########
# 결과 확인
######
prediction = tf.argmax(logits, 1)
prediction_check = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

seq_data = ['1234', '3456', '6789', '7890']
x_batch, y_batch = one_hot_seq(seq_data)

real, predict, accuracy_val = sess.run([tf.argmax(Y, 1), prediction, accuracy],
                                       feed_dict={X: x_batch, Y: y_batch})

print "\n=== 예측 결과 ==="
print '순차열:', seq_data
print '실제값:', [num_arr[i] for i in real]
print '예측값:', [num_arr[i] for i in predict]
print '정확도:', accuracy_val

