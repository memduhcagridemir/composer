class Config(object):
    def __init__(self):
        self.data_dir = "data/bach"
        self.save_dir = "save"
        self.log_dir = "logs"
        self.out_dir = "outs"

        self.rnn_size = 128
        self.num_layers = 3
        self.batch_size = 128
        self.seq_length = 15
        self.num_epochs = 32
        self.save_every = 2
        self.vocab_size = None
        self.grad_clip = 5.
        self.learning_rate = 3e-3
        self.n = 1000
        self.prime = 'G/2A/2'
