import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class NeuralNetwork:

    @staticmethod
    # will use this to initialize both the actor network its slowly-changing target network with same structure
    def build_actor_network(state_input_size):
        # Input layer
        network = input_data(shape=[None, state_input_size, 1], name='input')
        # Hidden layers
        network = fully_connected(network, n_units=8, activation='relu')
        # network = dropout(incoming=network, keep_prob=.8)
        network = fully_connected(network, n_units=8, activation='relu')
        # network = dropout(incoming=network, keep_prob=.8)
        network = fully_connected(network, n_units=8, activation='relu')
        # network = dropout(incoming=network, keep_prob=.8)
        # Output layer
        action = fully_connected(network, n_units=1, activation='linear')
        network = regression(incoming=network, optimizer='adam', loss='mean_square', learning_rate=1e-3, name='targets')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model




'''
Observation: [cos(theta) sin(theta) theta] - [(-1, 1), (-1, 1), (-8, 8)]
Action: Joint effort - (-2, 2)
Reward: Theta is normalized between -pi and pi. -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
In essence, the goal is to remain at zero angle (vertical), with the least rotational velocity, and the least effort.
'''
env = gym.make('Pendulum-v0')
training_data = NeuralNetwork.find_training_data(env, n_episodes=50000, score_goal=-255, n_steps=100)
model = NeuralNetwork.train_model(training_data)

# It is possible to save the model
# model.save('pendulum.model')
# model.load('pendulum.model')

for _ in range(10):
    observation = env.reset()
    score = 0
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, _, _ = env.step(action)
        score += reward

    print('Score: {}'.format(score))
