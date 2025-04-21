import collections
import numpy
import random


class PER:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        delta = numpy.array([args[-1] for args in self.buffer])
        buffer_batch = [self.buffer[i] for i in
                        numpy.random.choice(len(self.buffer), batch_size, p=delta / delta.sum())]
        return [[buffer_batch[i][j] for i in range(batch_size)] for j in range(5)]


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))
