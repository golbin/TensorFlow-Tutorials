# -*- coding: utf-8 -*-
# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.

import tensorflow as tf
import numpy as np


# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       1234 -> [1, 2, 3, 4]
#       12   -> [1, 2, P, P]
num_arr = ['S', 'E', 'P', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
num_dic = {n: i for i, n in enumerate(num_arr)}
dic_len = len(num_dic)

# 동적인 값을 넣어주기 위해 입력값과 출력값을 나눠서 처리
# 12 -> X, 34 -> Y
# 123 -> X, 456 -> Y
seq_data = [['12', '34'], ['23', '45'], ['34', '56'], ['45', '67'], ['56', '78'], ['67', '89'], ['78', '90']]
seq_data2 = [['123', '456'], ['234', '567'], ['345', '678'], ['456', '789'], ['567', '890']]


def one_hot_seq(seq_data):
    x_batch = []
    y_batch = []
    target_batch = []
    for seq in seq_data:
        # 입력값과 출력값의 time step 을 같게 하기 위해 P 를 앞에 붙여준다. (안해도 됨)
        x_data = [num_dic[n] for n in ('P' + seq[0])]
        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        y_data = [num_dic[n] for n in ('S' + seq[1])]
        # 학습을 위해 비교할 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target_data = [num_dic[n] for n in (seq[1] + 'E')]

        x_batch.append(np.eye(dic_len)[x_data])
        y_batch.append(np.eye(dic_len)[y_data])
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits)
        target_batch.append(target_data)

    return x_batch, y_batch, target_batch


#########
# 옵션 설정
######
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
n_classes = n_input = dic_len
n_hidden = 128
n_layers = 3


#########
# 신경망 모델 구성
######
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
# [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])

W = tf.Variable(tf.ones([n_hidden, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))

# tf.nn.dynamic_rnn 옵션에서 time_major 값을 True 로 설정
# [batch_size, n_steps, n_input] -> Tensor[n_steps, batch_size, n_input]
enc_input = tf.transpose(enc_input, [1, 0, 2])
dec_input = tf.transpose(dec_input, [1, 0, 2])

def cell():
    rnn_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
    rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=0.5)
    return rnn_cell

# 인코더 셀을 구성한다.
with tf.variable_scope('encode'):
    # enc_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
    # enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    enc_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(n_layers)])

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

# 디코더 셀을 구성한다.
with tf.variable_scope('decode'):
    # dec_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
    # dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    dec_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(n_layers)])

    # Seq2Seq 모델은 인코더 셀의 최종 상태값을
    # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)


# sparse_softmax_cross_entropy_with_logits 함수를 사용하기 위해
# 각각의 텐서의 차원들을 다음과 같이 변형하여 계산한다.
#    -> [batch size, time steps, hidden layers]
time_steps = tf.shape(outputs)[1]
#    -> [batch size * time steps, hidden layers]
outputs_trans = tf.reshape(outputs, [-1, n_hidden])
#    -> [batch size * time steps, class numbers]
logits = tf.matmul(outputs_trans, W) + b
#    -> [batch size, time steps, class numbers]
logits = tf.reshape(logits, [-1, time_steps, n_classes])


cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_batch, y_batch, target_batch = one_hot_seq(seq_data)
x_batch2, y_batch2, target_batch2 = one_hot_seq(seq_data2)

for epoch in range(100):
    _, loss4 = sess.run([train_op, cost],
                        feed_dict={enc_input: x_batch, dec_input: y_batch, targets: target_batch})
    _, loss3 = sess.run([train_op, cost],
                        feed_dict={enc_input: x_batch2, dec_input: y_batch2, targets: target_batch2})

    print ('Epoch:', '%04d' % (epoch + 1), 'cost =', \
        'bucket[4] =', '{:.6f}'.format(loss4), \
        'bucket[3] =', '{:.6f}'.format(loss3))

print ('최적화 완료!')


#########
# 결과 확인
######
def prediction_test(seq_data):
    prediction = tf.argmax(logits, 2)
    prediction_check = tf.equal(prediction, targets)
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

    x_batch_t, y_batch_t, target_batch_t = one_hot_seq(seq_data)
    real, predict, accuracy_val = sess.run([targets, prediction, accuracy],
                                           feed_dict={enc_input: x_batch_t,
                                                      dec_input: y_batch_t,
                                                      targets: target_batch_t})

    print ("\n=== 예측 결과 ===")
    print ('순차열:', seq_data)
    print ('실제값:', [[num_arr[j] for j in dec] for dec in real])
    print ('예측값:', [[num_arr[i] for i in dec] for dec in predict])
    print ('정확도:', accuracy_val)


# 학습 데이터에 있던 시퀀스로 테스트
prediction_test(seq_data)
prediction_test(seq_data2)

seq_data_test = [['12', '34'], ['23', '45'], ['78', '90']]
prediction_test(seq_data_test)

seq_data_test = [['123', '456'], ['345', '678'], ['567', '890']]
prediction_test(seq_data_test)


#########
# 입력만으로 다음 시퀀스를 예측해보자
######
# 시퀀스 데이터를 받아 다음 결과를 예측하고 디코딩하는 함수
def decode(seq_data):
    prediction = tf.argmax(logits, 2)
    x_batch_t, y_batch_t, target_batch_t = one_hot_seq([seq_data])

    result = sess.run(prediction,
                      feed_dict={enc_input: x_batch_t,
                                 dec_input: y_batch_t,
                                 targets: target_batch_t})

    decode_seq = [[num_arr[i] for i in dec] for dec in result][0]

    return decode_seq


# 시퀀스 데이터를 받아 다음 한글자를 예측하고,
# 종료 심볼인 E 가 나올때까지 점진적으로 예측하여 최종 결과를 만드는 함수
def decode_loop(seq_data):
    decode_seq = ''
    current_seq = ''

    while current_seq != 'E':
        decode_seq = decode(seq_data)
        seq_data = [seq_data[0], ''.join(decode_seq)]
        current_seq = decode_seq[-1]

    return decode_seq


print ("\n=== 한글자씩 점진적으로 시퀀스를 예측 ===")

seq_data = ['123', '']
print ("123 ->", decode_loop(seq_data))

seq_data = ['67', '']
print ("67 ->", decode_loop(seq_data))

seq_data = ['3456', '']
print ("3456 ->", decode_loop(seq_data))

print ("\n=== 전체 시퀀스를 한 번에 예측 ===")

seq_data = ['123', 'PPP']
print ("123 ->", decode(seq_data))

seq_data = ['67', 'PP']
print ("67 ->", decode(seq_data))

seq_data = ['3456', 'PPPP']
print ("3456 ->", decode(seq_data))
