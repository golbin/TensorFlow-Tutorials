# 자연어 처리나 음성 처리 분야에 많이 사용되는 RNN 의 기본적인 사용법을 익힙니다.
# 1 부터 0 까지 순서대로 숫자를 예측하여 세는 모델을 만들어봅니다.
import tensorflow as tf
import numpy as np


# char_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

# one-hot 인코딩 사용 및 디코딩을 하기 위해 연관 배열을 만듭니다.
# {'a': 0, 'b': 1, 'c': 2, ..., 'j': 9, 'k', 10, ...}
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 다음 배열은 입력값과 출력값으로 다음처럼 사용할 것 입니다.
# wor -> X, d -> Y
# woo -> X, d -> Y
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']


def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        # 여기서 생성하는 x_data 와 y_data 는
        # 실제 숫자가 아니라 숫자 리스트의 인덱스 번호 입니다.
        # [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5] ...
        input = [num_dic[n] for n in seq[:-1]]
        # 3, 4, 5, 6...10
        target = num_dic[seq[-1]]
        # one-hot 인코딩을 합니다.
        # if x_data is [0, 1, 2]:
        # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
        input_batch.append(np.eye(dic_len)[input])
        # 지금까지 손실함수로 사용하던 softmax_cross_entropy_with_logits 함수는
        # label 값을 one-hot 인코딩으로 넘겨줘야 하지만,
        # 이 예제에서 사용할 손실 함수인 sparse_softmax_cross_entropy_with_logits 는
        # one-hot 인코딩을 사용하지 않으므로 index 를 그냥 넘겨주면 됩니다.
        target_batch.append([target])

    return input_batch, target_batch

#########
# 옵션 설정
######
learning_rate = 0.01
# 입력값 크기. 10개의 숫자에 대한 one-hot 인코딩이므로 10개가 됩니다.
# 예) 3 => [0 0 1 0 0 0 0 0 0 0 0]
# 출력값도 입력값과 마찬가지로 10개의 숫자로 분류합니다.
n_input = n_class = dic_len
# 타입 스텝: [1 2 3] => 3
# RNN 을 구성하는 시퀀스의 갯수입니다.
n_step = 3
n_class = dic_len
# 히든 레이어의 특성치 갯수
n_hidden = 128
total_epoch = 30

#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32, [None, n_step, n_input])
# 비용함수에 sparse_softmax_cross_entropy_with_logits 을 사용하므로
# 출력값과의 계산을 위한 원본값의 형태는 다음과 같습니다.
# [batch size, time steps]
# 기존처럼 one-hot 인코딩을 사용한다면 입력값의 형태는 [None, n_class] 여야합니다.
Y = tf.placeholder(tf.int32, [None, 1])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# RNN 셀을 생성합니다.
cell1 = tf.contrib.rnn.BasicLSTMCell(n_hidden)
# 과적합 방지를 위한 Dropout 기법을 사용합니다.
cell1 = tf.contrib.rnn.DropoutWrapper(cell1, output_keep_prob=0.5)
# 여러개의 셀을 조합해서 사용하기 위해 셀을 추가로 생성합니다.
cell2 = tf.contrib.rnn.BasicLSTMCell(n_hidden)

# 여러개의 셀을 조합한 RNN 셀을 생성합니다.
multi_cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2])

# tf.nn.dynamic_rnn 함수를 이용해 순환 신경망을 만듭니다.
# time_major=True
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

# 최종 결과는 one-hot 인코딩 형식으로 만듭니다
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b
# 단, sparse_softmax_cross_entropy_with_logits 함수의 labels 는
# one-hot 인코딩을 사용하지 않기 때문에, 1차원 배열로 넘겨줍니다. (time step 이 1이기 때문)
# [[1]]
# [[2]]
# [[3]]
#  ...  -> [1], [2], [3] ...
labels = tf.reshape(Y, [-1])

cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=labels))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={X: input_batch, Y: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')

#########
# 결과 확인
######
# 레이블값이 정수이므로 예측값도 정수로 변경해줍니다.
prediction = tf.cast(tf.argmax(model, 1), tf.int32)
# one-hot 인코딩이 아니므로 입력값을 그대로 비교합니다.
prediction_check = tf.equal(prediction, labels)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

# wor, coo, lov, kis 를 가지고 단어를 추측해보겠습니다.
seq_data = ['word', 'cool', 'love', 'kiss']
input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy],
                                 feed_dict={X: input_batch, Y: target_batch})

predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n=== 예측 결과 ===')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)
