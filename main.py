import numpy as np
from deep_q_network import DeepQNetwork
from environment_manager import EnvManager
from epsilon_greedy_strategy import EpsilonGreedyStrategy
from agent import Agent
from replay_memory import ReplayMemory, Experience

render = False
batch_size = 160
gamma = 0.999  # Is the discount factor used in the Bellman equation
eps_start = 1  # Starting value of epsilon
eps_end = 0.1  # Ending value of epsilon
eps_decay = 0.001  # Decay rate we’ll use to decay epsilon over time
target_update = 10  # How frequently, in terms of episodes, we’ll update the target network weights with the policy network weights.
memory_size = 2000  # Capacity of the replay memory
lr = 0.001  # Learning rate
num_episodes = 1000  # Number of episodes we want to play

environment_manager = EnvManager('SpaceInvaders-v0')
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, environment_manager.num_actions_available())
memory = ReplayMemory(memory_size)
policy_net = DeepQNetwork(input_shape=(environment_manager.get_input_shape(),),
                          action_space=environment_manager.num_actions_available(), batch_size=batch_size)
target_net = DeepQNetwork(input_shape=(environment_manager.get_input_shape(),),
                          action_space=environment_manager.num_actions_available(), batch_size=batch_size)

episode_max_rewards = []

# Episode loop
for episode in range(num_episodes):
    max_episode_reward = 0
    environment_manager.reset()
    state = environment_manager.get_state()
    environment_manager.done = False

    # Steps loop
    steps = 0
    while not environment_manager.done:

        if render == True:
            environment_manager.render()

        action = agent.select_action(state, policy_net)
        experience = environment_manager.take_action(action)
        state = experience[0]
        action = experience[1]
        next_state = experience[2]
        reward = experience[3]
        steps += 1
        memory.push(Experience(state, action, next_state, reward))
        max_episode_reward += reward
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences_batch = memory.sample(batch_size)
            states = np.zeros((batch_size, environment_manager.final_reshape))
            next_states = np.zeros((batch_size, environment_manager.final_reshape))
            actions, rewards = [], []

            # Prepare data batch
            for i in range(batch_size):
                states[i] = experiences_batch[i][0]
                actions.append(experiences_batch[i][1])
                next_states[i] = experiences_batch[i][2]
                rewards.append(experiences_batch[i][3])

            current_q_values = policy_net.predict(states)
            target_q_values = target_net.predict(next_states)

            # Create Q_targets
            for i in range(batch_size):
                # Q_max = max_a' Q_target(s', a')
                target_q_values[i][actions[i]] = rewards[i] + gamma * (np.amax(target_q_values[i]))

            # Train Policy Network
            policy_net.train(states, target_q_values)

        if environment_manager.done:
            print("Episode: " + str(episode) + " Max reward:" + str(max_episode_reward))
            episode_max_rewards.append(max_episode_reward)
            break
    # update target network and save network
    if episode % target_update == 0:
        target_net.copy_weights_from_nn(policy_net)
        policy_net.save()
