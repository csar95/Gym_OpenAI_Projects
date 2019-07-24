import numpy as np
import random
import gym


# Eplison-Greedy policy: Choose best action with probability 1-E

env = gym.make('FrozenLake-v0')  # Left: 0 | Down: 1 | Right: 2 | Up: 3

nTiles = env.observation_space.n
nActions = env.action_space.n

Q_table = np.full((nTiles, nActions), 0.5)
Q_table[:, 0] = 1

epsilon = 0.1

for episode in range(10):
    observation = env.reset()

    for i in range(100):
        env.render()
        if random.uniform(0, 1) < 1-epsilon:
            action = np.argmax(Q_table[observation])
        else:
            action = env.action_space.sample()

        print('Action: {}'.format(action))

        observation, reward, done, info = env.step(action)

        print('Observation: {} | Reward: {} | Done: {}'.format(observation, reward, done))

        if done:
            print("Episode {} finished after {} iterations".format(episode+1, i+1))
            break
