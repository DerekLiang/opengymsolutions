import gym

import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import bcolz
import itertools

class MyModel():
    def __init__(self):
        self.states, self.targets = [], []
        self.statesExploring, self.targetsExploring = [], []

    def buildDefaultModel(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(4,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='mse', optimizer='adadelta', metrics=['mse', 'accuracy'])
        model.summary()
        self.model = model

    def predict(self, state):
        return np.argmax(self.model.predict(np.array([state]))[0])

    def addExploring(self, state, target):
        self.statesExploring.append(state)
        self.targetsExploring.append(target)

    def train(self):
        self.states.extend(self.statesExploring)

        self.targets.extend(self.targetsExploring[:-3])
        self.targets.extend( [ 1 if x==0 else 0 for x in self.targetsExploring[-3:]] ) #switch the target of the last action

        self.statesExploring, self.targetsExploring = [], []
        self.states = self.states[:4000]
        self.targets = self.targets[:4000]

        # self.targets = self.targets[-1000:]
        targets = np_utils.to_categorical(self.targets, 2)
        states = np.array(self.states).reshape(-1, 4)

        self.model.fit(states, targets, shuffle=True, epochs=3, verbose=1)

    def currentexploringDepth(self):
        return len(self.statesExploring)

env = gym.make('CartPole-v0')

model = MyModel()
model.buildDefaultModel()

while True:
    observation = env.reset()
    while True:
        env.render()

        #exploring
        action = model.predict(observation)
        model.addExploring(observation, action)
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(model.currentexploringDepth()))
            #learn
            model.train()
            break


# print('generating learning data')
# generate_data()

# print('learning...')
# meme.learn()

# print('load model')
# meme.loadModel()

# print('show off...')
# while True:
#     observation = env.reset()
#     for t in range(1000**2):
#         env.render()


#         if len(obs) >= learnDepth:
#             (action, predict) = meme.getAction(observation[:-learnDepth])
#         else:
#             action = env.action_space.sample()

#         observation, reward, done, info = env.step(action)
#         obs.append((observation, action))

#         if len(obs) > learnDepth:
#             print(action, observation, (observation - predict)*100/observation)

#         if abs(observation[2])> 0.17:
#             print('dangerous')
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
