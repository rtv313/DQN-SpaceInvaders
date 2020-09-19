import gym
import cv2
import numpy as np

class EnvManager():
    def __init__(self, environment):
        self.done = False
        self.env = gym.make(environment)
        state = self.env.reset()
        state = cv2.resize(state,(40,40))
        reshape_dim_one = state.shape[0]
        reshape_dim_two = state.shape[1]
        reshape_dim_three = state.shape[2]
        self.final_reshape = reshape_dim_one * reshape_dim_two * reshape_dim_three
        self.input_shape = (self.final_reshape)
        state_reshaped = np.reshape(state, [1,self.final_reshape])
        self.current_state = state_reshaped

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def __process_state(self, state):
        state = cv2.resize(state,(40,40))
        processed_image_state = np.reshape(state, [1, self.final_reshape])
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
        return (self.input_shape)