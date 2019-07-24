import gym
import numpy as np
import random


# class Board:
#     def __init__(self):
#         self.board = np.zeros((5, 5))


env = gym.make('Taxi-v2')  # 0: South | 1: North | 2: East | 3: West | 4: Pickup | 5: Dropoff
# Blue: passenger | Magenta: destination | Yellow: empty taxi | Green: full taxi |
# Other letters (R, G, B and Y): locations for passengers and destinations

Q_table = np.zeros((env.observation_space.n, env.action_space.n))  # 500 x 6

previousRewards = np.zeros(50)

epsilon = 0.1
discounting = 0.99
learningRate = 0.25
achievedGoal = False

for episode in range(15000):
    totalReward = .0  # +20: Successful dropoff | -1: Every timestep | -10: Illegal pickup or dropoff actions
    done = False
    newObservation = env.reset()

    itr = 1
    while not done:
        oldObservation = newObservation
        # print(observation)
        # env.render()
        if random.uniform(0,1) < 1 - epsilon:
            action = np.argmax(Q_table[oldObservation])
        else:
            action = env.action_space.sample()

        newObservation, reward, done, info = env.step(action)
        totalReward += reward

        Q_table[oldObservation][action] += learningRate * (reward + discounting * np.amax(Q_table[newObservation]) -
                                                           Q_table[oldObservation][action])

        if done:
            print("Episode {} finished after {} iterations with total reward: {}.".format(episode+1, itr, totalReward))
            break

        itr += 1

    previousRewards = np.roll(previousRewards, 1)
    previousRewards[0] = totalReward
    # if sum(previousRewards) / len(previousRewards) > 0.85:
    #     break

    # if not foundGoal and totalReward == 1:
    #     foundGoal = True
    if episode > 7500:
        epsilon *= 0.9999

print(previousRewards)

done = False
observation = env.reset()
while not done:
    print(observation)
    env.render()
    action = np.argmax(Q_table[observation])
    observation, reward, done, info = env.step(action)
env.render()
