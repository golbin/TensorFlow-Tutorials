# -*- coding: utf-8 -*-
# 텐서플로우의 기본적인 구성을 익힙니다.

import tensorflow as tf


# tf.constant: 말 그대로 상수입니다.
hello = tf.constant('Hello, TensorFlow!')

a = tf.constant(10)
b = tf.constant(32)
c = a + b

# tf.placeholder: 계산을 실행할 때 입력값을 받는 변수로 사용합니다.
# None 은 크기가 정해지지 않았음을 의미합니다.
X = tf.placeholder("float", [None, 3])

# tf.Variable: 그래프를 계산하면서 최적화 할 변수들입니다. 이 값이 바로 신경망을 좌우하는 값들입니다.
# tf.random_normal: 각 변수들의 초기값을 정규분포 랜덤 값으로 초기화합니다.
# name: 나중에 텐서보드등으로 값의 변화를 추적하거나 살펴보기 쉽게 하기 위해 이름을 붙여줍니다.
W = tf.Variable(tf.random_normal([3, 2]), name='Weights')
b = tf.Variable(tf.random_normal([2, 1]), name='Bias')

x_data = [[1, 2, 3], [4, 5, 6]]

# 입력값과 변수들을 계산할 수식을 작성합니다.
# tf.matmul 처럼 mat* 로 되어 있는 함수로 행렬 계산을 수행합니다.
expr = tf.matmul(X, W) + b

# 그래프를 실행할 세션을 구성합니다.
sess = tf.Session()
# sess.run: 설정한 텐서 그래프(변수나 수식 등등)를 실행합니다.
# 최초에 tf.global_variables_initializer 를 한 번 실행해야 합니다.
sess.run(tf.global_variables_initializer())

# 위에서 변수와 수식들을 정의했지만, 실행이 정의한 시점에서 실행되는 것은 아닙니다.
# 다음처럼 sess.run 함수를 사용하면 그 때 계산이 됩니다.
# 따라서 모델을 구성하는 것과, 실행하는 것을 분리하여 프로그램을 깔끔하게 작성할 수 있습니다.
print "=== contants ==="
print sess.run(hello)
print "a + b = c =", sess.run(c)
print "=== x_data ==="
print x_data
print "=== W ==="
print sess.run(W)
print "=== b ==="
print sess.run(b)
print "=== expr ==="
# expr 수식에는 X 라는 입력값이 필요합니다.
# 따라서 expr 실행시에는 이 변수에 대한 실제 입력값을 다음처럼 넣어줘야합니다.
print sess.run(expr, feed_dict={X: x_data})

# 세션을 닫습니다.
sess.close()
