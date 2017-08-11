import os
import re
from subprocess import call
from six.moves import cPickle

from model import Model
from config import Config

import tensorflow as tf


def compose(params, filename="outs/last-output.abc"):
    with open(os.path.join(params.save_dir, 'config.pkl'), 'rb') as f:
        params = cPickle.load(f)
    with open(os.path.join(params.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)

    model = Model(params, training=False)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(params.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        with open(os.path.join(filename), 'wb') as fw:
            sampled_abc = model.sample(sess, chars, vocab, params.n, params.prime).encode('utf-8')

            sampled_abc = re.sub(r"\].*\[", "][", sampled_abc)
            sampled_abc = re.sub(r"\\\s*\n[^\n]*$", "", sampled_abc)

            fw.write("X: 1\n")
            fw.write("T: Composer\n")
            fw.write("M: 4/4\n")
            fw.write("K: A\n")
            fw.write(sampled_abc)


def listen(params, filename="outs/last-output.abc"):
    with open(os.path.join(params.save_dir, 'config.pkl'), 'rb') as f:
        params = cPickle.load(f)

    output_mid = os.path.join(params.out_dir, "output.mid")
    output_mp3 = os.path.join(params.out_dir, "output.mp3")

    call("abc2midi {filename} -o  {output_mid}".format(filename=filename, output_mid=output_mid), shell=True)
    call("timidity -Or -o - {output_mid} | lame -r - {output_mp3}".format(output_mid=output_mid,
                                                                          output_mp3=output_mp3), shell=True)


def main():
    params = Config()

    compose(params, "outs/last-output.abc")
    listen(params, "outs/last-output.abc")

if __name__ == '__main__':
    main()
