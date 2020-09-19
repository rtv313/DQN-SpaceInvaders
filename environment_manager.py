import gym
import numpy as np

class EnvManager():
    def __init__(self, environment):
        self.done = False
        self.env = gym.make(environment)
        state = self.env.reset()
        self.current_state = np.reshape(state, [1, self.env.observation_space.shape[0]])

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def __process_state(self, state):
        processed_image_state = np.reshape(state,[1,self.env.observation_space.shape[0]])
        return processed_image_state

    def take_action(self, action):
        next_state_before_proccess, reward, done, info = self.env.step(action)
        self.done = done
        next_state = self.__process_state(next_state_before_proccess)
        experience_tuple = (self.current_state, action, next_state, reward, done, info)
        self.current_state = next_state
        return experience_tuple

    def num_actions_available(self):
        return self.env.action_space.n

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        return self.current_state

    def get_input_shape(self):
        input_shape = (self.env.observation_space.shape[0])
        return input_shape