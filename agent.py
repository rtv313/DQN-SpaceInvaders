import random
import numpy as np


class Agent():

    def __init__(self, strategy, num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            # Explore
            action = random.randrange(self.num_actions)
            return action
        else:
            # Exploit
            return np.argmax(policy_net.predict(state))
