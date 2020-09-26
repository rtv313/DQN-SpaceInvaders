import numpy as np
from environment_manager import EnvManager
from save_load_module import SaveLoadModule

num_episodes = 10
render = True

dqn = SaveLoadModule.load_nn_model(SaveLoadModule.get_most_trained_model())
environment_manager = EnvManager('SpaceInvaders-v0')

for episode in range(num_episodes):
    max_episode_reward = 0
    environment_manager.reset()
    state = environment_manager.get_state()
    environment_manager.done = False
    total_reward = 0
    # Steps loop
    steps = 0
    while not environment_manager.done:

        if render:
            environment_manager.render()

        action = np.argmax(dqn.predict(state))
        experience = environment_manager.take_action(action)
        state = experience[0]
        next_state = experience[2]
        state = next_state
        reward = experience[3]
        total_reward += reward

    print('Episode ' + str(episode) + ' Total Reward: ' + str(total_reward))
