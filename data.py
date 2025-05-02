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
        delta = numpy.array([self.buffer[i][-1] for i in range(len(self.buffer))])
        batch_mask = numpy.random.choice(
            len(self.buffer), batch_size, p=delta / delta.sum()
        )
        return zip(*[self.buffer[batch_mask[i]][:-1] for i in range(len(batch_mask))])


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))
