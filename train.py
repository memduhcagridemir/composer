import os
import time
from six.moves import cPickle

import tensorflow as tf
import logging

from model import Model
from config import Config
from batcher import Batcher
from utils import setup_logger


def train(params):
    data_loader = Batcher(params)
    params.vocab_size = data_loader.vocab_size

    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    with open(os.path.join(params.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(params, f)
    with open(os.path.join(params.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(params)

    with tf.Session() as sess:
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(params.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

        for e in range(params.num_epochs):
            sess.run(tf.assign(model.lr, params.learning_rate * (0.97 ** e)))

            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):
                start = time.time()

                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)

                summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                writer.add_summary(summ, e * data_loader.num_batches + b)

                end = time.time()
                logging.info("Epoch #{e} / Batch #{b} -- Loss {train_loss:.3f} "
                             "Time {time_diff:.3f}".format(e=e, b=b, train_loss=train_loss, time_diff=end - start))

            if e % params.save_every == 0 or e == params.num_epochs - 1:
                checkpoint_path = os.path.join(params.save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=e)

if __name__ == '__main__':
    setup_logger()

    parameters = Config()
    train(parameters)
