import gym

import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import bcolz
import itertools

learnDepth = 3
version = 2
class Meme():
    def getAction(self, ob):
        return np.argmax(np.array(self.model.predict(np.array([ob])))[0])

    def learn(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(4,)))
        model.add(Dense(512, ))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='mse', optimizer='sgd', metrics=['mse', 'accuracy'])
        model.summary()
        self.model = model

        states = bcolz.carray(rootdir='data/cartpole_state_v{0}'.format(version), mode='r')
        targets = bcolz.carray(rootdir='data/cartpole_target_v{0}'.format(version), mode='r')
        states = states.reshape((-1,4))
        print('total targets', len(targets))

        targets = np_utils.to_categorical(targets, 2)
        self.model.fit(states, targets, shuffle=True, validation_split=0.2, epochs=70, verbose=1)

        self.model.save('data/cartpole1.model')

    def loadModel(self):
        self.model = load_model('data/cartpole1.model_v{0}'.format(version))

env = gym.make('CartPole-v0')
meme = Meme()

def generate_data():
    states = bcolz.carray([], rootdir='data/cartpole_state_v{0}'.format(version), mode='w')
    targets = bcolz.carray([], rootdir='data/cartpole_target_v{0}'.format(version), mode='w', dtype='i')
    sample = 0
    while sample < 20000:
        observation = env.reset()
        for t in range(1000):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

            with states, targets:
                states.append(observation)
                targets.append(action)
            sample += 1


# print('generating learning data')
# generate_data()

print('learning...')
meme.learn()

print('load model')
meme.loadModel()

print('show off...')
while True:
    observation = env.reset()
    for t in range(1000**2):
        env.render()

        action = meme.getAction(observation)
        observation, reward, done, info = env.step(action)

        if abs(observation[2])> 0.17:
            print('dangerous')
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
