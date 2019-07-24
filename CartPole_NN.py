import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class NeuralNetwork:

    @staticmethod
    def neural_network_model(inputSize):
        # Input layer (The observation)
        network = input_data(shape=[None, inputSize, 1], name='input')

        # Hidden layers
        network = fully_connected(network, 128, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 516, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 128, activation='relu')
        network = dropout(network, 0.8)

        # Output layer (The action to be taken)
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                             name='targets')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    @staticmethod
    def train_model(trainingData, model=False):
        # '-1' means that it is an unknown dimension and we want numpy to figure it out.
        # Numpy will figure this by looking at the 'length of the array and remaining dimensions' and
        # making sure it satisfies the above mentioned criteria.
        observations_X = np.array([sample[0] for sample in trainingData]).reshape(-1, len(trainingData[0][0]), 1)
        actions_y = np.array([sample[1] for sample in trainingData])

        if not model:
            model = NeuralNetwork.neural_network_model(inputSize=len(observations_X[0]))

        model.fit(observations_X, actions_y, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openaistuff')

        return model

    @staticmethod
    def find_training_data(env, nEpisodes, scoreGoal):
        trainingData = np.empty(shape=(0, 2), dtype=[('input', object), ('output', object)])
        scores = []

        for _ in range(nEpisodes):
            score = 0
            done = False
            # List of observations and correspondent actions
            gameMemory = np.empty(shape=(0, 2), dtype=[('observation', object), ('action', int)])

            observation = env.reset()
            while not done:
                action = env.action_space.sample()
                gameMemory = np.append(gameMemory, [(np.array(observation), action)], axis=0)
                observation, reward, done, info = env.step(action)
                score += reward

            if score > scoreGoal:
                for move in gameMemory:
                    if move[1] == 1:  # Action taken was 1
                        trainingData = np.append(trainingData, [(move[0], np.array([0, 1]))], axis=0)
                    elif move[1] == 0:  # Action taken was 0
                        trainingData = np.append(trainingData, [(move[0], np.array([1, 0]))], axis=0)
                scores.append(score)

        return trainingData


"""
Num	Observation                 Min         Max
0   Cart Position             -4.8            4.8
1	Cart Velocity             -Inf            Inf
2	Pole Angle                 -24 deg        24 deg
3	Pole Velocity At Tip      -Inf            Inf
"""
# Reward: 1 for every step taken, including the termination step
env = gym.make('CartPole-v1')

learningRate = 1e-3

trainingData = NeuralNetwork.find_training_data(env, nEpisodes=15000, scoreGoal=70)

model = NeuralNetwork.train_model(trainingData)

# It is possible to save the model
# model.save('cartPole.model')
# model.load('cartPole.model')

for episode in range(10):
    done = False
    score = 0

    observation = env.reset()
    while not done:
        env.render()
        # model.predict() devuelve un array de la forma [[]]
        action = np.argmax(model.predict(np.array(observation).reshape(-1, len(observation), 1))[0])
        observation, reward, done, info = env.step(action)
        score += reward

    print('Episode {} got a score of {}'.format(episode, score))
