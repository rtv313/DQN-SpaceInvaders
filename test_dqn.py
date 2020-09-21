import numpy as np
from deep_q_network import DeepQNetwork
from environment_manager import EnvManager
from epsilon_greedy_strategy import EpsilonGreedyStrategy
from agent import Agent
from replay_memory import ReplayMemory, Experience
import tensorflow as tf

num_episodes = 10
render = True

dqn  = tf.keras.models.load_model('models/ep-20.h5')
environment_manager = EnvManager('SpaceInvaders-v0')

for episode in range(num_episodes):
    max_episode_reward = 0
    environment_manager.reset()
    state = environment_manager.get_state()
    environment_manager.done = False

    # Steps loop
    steps = 0
    while not environment_manager.done:

        if render:
            environment_manager.render()

        action = np.argmax(dqn.predict(state))
        experience = environment_manager.take_action(action)
        state = experience[0]
        action = experience[1]
        next_state = experience[2]
        state = next_state