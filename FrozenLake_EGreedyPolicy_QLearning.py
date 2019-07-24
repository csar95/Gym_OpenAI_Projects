import numpy as np
import random
import gym


# Eplison-Greedy policy: Choose best action with probability 1-E

env = gym.make('FrozenLake-v0')  # Left: 0 | Down: 1 | Right: 2 | Up: 3

nTiles = env.observation_space.n
nActions = env.action_space.n

Q_table = np.zeros((nTiles, nActions))
previousRewards = np.zeros(50)

epsilon = 0.1
discounting = 0.99
learningRate = 0.15
foundGoal = False

for episode in range(80000):
    done = False
    newObservation = env.reset()
    totalReward = .0

    itr = 1
    while not done:
        # env.render()
        oldObservation = newObservation

        if random.uniform(0, 1) < 1-epsilon:
            action = np.argmax(Q_table[oldObservation])
        else:
            action = env.action_space.sample()

        newObservation, reward, done, info = env.step(action)

        Q_table[oldObservation][action] += learningRate * (reward + discounting * np.amax(Q_table[newObservation]) -
                                                           Q_table[oldObservation][action])
        totalReward += reward

        if done:
            print("Episode {} finished after {} iterations with total reward: {}.".format(episode+1, itr, totalReward))
            break

        itr += 1

    previousRewards = np.roll(previousRewards, 1)
    previousRewards[0] = totalReward
    if sum(previousRewards) / len(previousRewards) > 0.85:
        break

    if not foundGoal and totalReward == 1:
        foundGoal = True
    if foundGoal and episode > 5000:
        epsilon *= 0.9999

# print(Q_table)
# print(previousRewards)

done = False
observation = env.reset()
while not done:
    env.render()
    action = np.argmax(Q_table[observation])
    observation, reward, done, info = env.step(action)
env.render()
