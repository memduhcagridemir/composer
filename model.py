import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model:
    def __init__(self, params, training=True):
        if not training:
            params.batch_size = 1
            params.seq_length = 1

        cells = []
        for _ in range(params.num_layers):
            cell = rnn.BasicLSTMCell(params.rnn_size)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.placeholder(tf.int32, [params.batch_size, params.seq_length])
        self.targets = tf.placeholder(tf.int32, [params.batch_size, params.seq_length])
        self.initial_state = cell.zero_state(params.batch_size, tf.float32)

        with tf.variable_scope('lstm_lm'):
            softmax_w = tf.get_variable("softmax_w", [params.rnn_size, params.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [params.vocab_size])

        embedding = tf.get_variable("embedding", [params.vocab_size, params.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        inputs = tf.split(inputs, params.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell,
                                                         loop_function=loop if not training else None, scope='lstm_lm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, params.rnn_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([params.batch_size * params.seq_length])])
        self.cost = (tf.reduce_sum(loss) / params.batch_size) / params.seq_length
        with tf.name_scope('cost'):
            self.cost = (tf.reduce_sum(loss) / params.batch_size) / params.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), params.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='A'):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(t, np.random.rand(1)*s))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
