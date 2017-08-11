import os
import codecs
import collections
import numpy as np
from six.moves import cPickle


class Batcher():
    def __init__(self, params):
        self.pointer = 0
        self.x_batches = None
        self.y_batches = None

        self.batch_size = params.batch_size

        input_file = os.path.join(params.data_dir, "input.txt")
        vocab_file = os.path.join(params.data_dir, "vocab.pkl")
        tensor_file = os.path.join(params.data_dir, "data.npy")

        with codecs.open(input_file, "r", encoding='utf-8') as f:
            data = f.read()

        counter = collections.Counter(data)

        self.chars, _ = zip(*counter.items())
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

        self.num_batches = int(self.tensor.size / (self.batch_size * params.seq_length))
        if self.num_batches <= 0:
            exit("num_batches is smaller than 0.")

        self.tensor = self.tensor[:self.num_batches * self.batch_size * params.seq_length]

        self.create_batches()

    def reset_batch_pointer(self):
        self.pointer = 0

    def create_batches(self):
        xdata = self.tensor
        ydata = np.zeros_like(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y
