import gym
import numpy as np

env = gym.make('SpaceInvaders-v0')
state = env.reset()
zeros_m_array = np.zeros(state.shape)
zeros_reshape = np.reshape(zeros_m_array,(1,100800))
print(state.shape)
print(zeros_m_array.shape)
print(zeros_reshape.shape)
print(zeros_reshape)
##state_reshaped = np.reshape(state, [1, env.observation_space.shape[0]])
##current_state = state_reshaped