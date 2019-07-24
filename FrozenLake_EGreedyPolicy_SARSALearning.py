import numpy as np
import random
import gym


class Trainer:
    @staticmethod
    def take_new_action(env, Q_table, epsilon, newObservation):
        if random.uniform(0, 1) < 1 - epsilon:
            return np.argmax(Q_table[newObservation])
        else:
            return env.action_space.sample()


# Eplison-Greedy policy: Choose best action with probability 1-E
# SARSA is an on-policy algorithm. It learns the Q-value based on the action performed by the current policy instead of
# the greedy policy.

env = gym.make('FrozenLake-v0')  # Left: 0 | Down: 1 | Right: 2 | Up: 3

nTiles = env.observation_space.n
nActions = env.action_space.n

Q_table = np.zeros((nTiles, nActions))
previousRewards = np.zeros(50)

epsilon = 0.1
discounting = 0.99
learningRate = 0.1
foundGoal = False

for episode in range(80000):
    totalReward = .0
    done = False
    newObservation = env.reset()
    newAction = Trainer.take_new_action(env, Q_table, epsilon, newObservation)

    itr = 1
    while not done:
        oldObservation = newObservation
        newObservation, reward, done, info = env.step(newAction)

        oldAction = newAction
        newAction = Trainer.take_new_action(env, Q_table, epsilon, newObservation)

        Q_table[oldObservation][oldAction] += learningRate * (reward + discounting * Q_table[newObservation][newAction]
                                                              - Q_table[oldObservation][oldAction])
        totalReward += reward

        if done:
            print("Episode {} finished after {} iterations with total reward: {}.".format(episode+1, itr, totalReward))
            break

        itr += 1

    if not foundGoal and totalReward == 1:
        foundGoal = True
    previousRewards = np.roll(previousRewards, 1)
    previousRewards[0] = totalReward
    if sum(previousRewards) / len(previousRewards) > 0.85:
        break
    if foundGoal and episode > 5000:
        epsilon *= 0.9999

print(Q_table)
print(previousRewards)
