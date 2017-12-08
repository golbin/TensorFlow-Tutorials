# 자세한 설명은 상위 폴더의 03 - Seq2Seq.py 등에서 찾으실 수 있습니다.
import tensorflow as tf


# Seq2Seq 기본 클래스
class Seq2Seq:

    logits = None
    outputs = None
    cost = None
    train_op = None

    def __init__(self, vocab_size, n_hidden=128, n_layers=3):
        self.learning_late = 0.001

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.enc_input = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        self.dec_input = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        self.targets = tf.placeholder(tf.int64, [None, None])

        self.weights = tf.Variable(tf.ones([self.n_hidden, self.vocab_size]), name="weights")
        self.bias = tf.Variable(tf.zeros([self.vocab_size]), name="bias")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self._build_model()

        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self):
        # self.enc_input = tf.transpose(self.enc_input, [1, 0, 2])
        # self.dec_input = tf.transpose(self.dec_input, [1, 0, 2])

        enc_cell, dec_cell = self._build_cells()

        with tf.variable_scope('encode'):
            outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.enc_input, dtype=tf.float32)

        with tf.variable_scope('decode'):
            outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, self.dec_input, dtype=tf.float32,
                                                    initial_state=enc_states)

        self.logits, self.cost, self.train_op = self._build_ops(outputs, self.targets)

        self.outputs = tf.argmax(self.logits, 2)

    def _cell(self, output_keep_prob):
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=output_keep_prob)
        return rnn_cell

    def _build_cells(self, output_keep_prob=0.5):
        enc_cell = tf.nn.rnn_cell.MultiRNNCell([self._cell(output_keep_prob)
                                                for _ in range(self.n_layers)])
        dec_cell = tf.nn.rnn_cell.MultiRNNCell([self._cell(output_keep_prob)
                                                for _ in range(self.n_layers)])

        return enc_cell, dec_cell

    def _build_ops(self, outputs, targets):
        time_steps = tf.shape(outputs)[1]
        outputs = tf.reshape(outputs, [-1, self.n_hidden])

        logits = tf.matmul(outputs, self.weights) + self.bias
        logits = tf.reshape(logits, [-1, time_steps, self.vocab_size])

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_late).minimize(cost, global_step=self.global_step)

        tf.summary.scalar('cost', cost)

        return logits, cost, train_op

    def train(self, session, enc_input, dec_input, targets):
        return session.run([self.train_op, self.cost],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.targets: targets})

    def test(self, session, enc_input, dec_input, targets):
        prediction_check = tf.equal(self.outputs, self.targets)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        return session.run([self.targets, self.outputs, accuracy],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.targets: targets})

    def predict(self, session, enc_input, dec_input):
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input})

    def write_logs(self, session, writer, enc_input, dec_input, targets):
        merged = tf.summary.merge_all()

        summary = session.run(merged, feed_dict={self.enc_input: enc_input,
                                                 self.dec_input: dec_input,
                                                 self.targets: targets})

        writer.add_summary(summary, self.global_step.eval())
