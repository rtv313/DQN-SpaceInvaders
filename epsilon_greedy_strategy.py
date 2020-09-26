import math


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.actual = None

    # Our agent is going to be able to use the exploration rate to determine how it should select it’s actions,
    # either by exploring or exploiting the environment.
    def get_exploration_rate(self, current_step):
        self.actual = self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)
        return self.actual

    def get_actual_exploration_rate(self):
        return self.actual