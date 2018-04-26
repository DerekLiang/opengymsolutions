import gym

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import bcolz

class Meme():
    def __init__(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(4,),  kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(512, activation='relu', input_shape=(4,),  kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                    metrics=['accuracy'])
        model.summary()
        self.model = model

    def getAction(self, obs):
        return np.argmax(np.array(self.model.predict(np.array([observation])))[0])

    def learn(self):
        states = bcolz.carray(rootdir='data/cartpole_state', mode='r')
        targets = bcolz.carray(rootdir='data/cartpole_target', mode='r')
        states = states.reshape((-1,4))
        targets = np_utils.to_categorical(targets, 2)

        self.model.fit(states, targets, shuffle=True, epochs=20, verbose=1)

env = gym.make('CartPole-v0')
meme = Meme()

def generate_data():
    states = bcolz.carray([], rootdir='data/cartpole_state', mode='w')
    targets = bcolz.carray([], rootdir='data/cartpole_target', mode='w', dtype='i')
    sample = 1
    while sample < 2000:
        observation = env.reset()
        observations = []
        for t in range(1000):
            env.render()
            action = env.action_space.sample()
            #print(observation, action)
            observations.append((observation, action))
            #meme.saveTraining(observation, action)

            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

        if len(observations) > 20:
            observations = observations[5:-5]
            for (o,a) in observations:
                states.append(o)
                targets.append(a)
            states.flush()
            targets.flush()
            sample += 1


#print('generating learning data')
#generate_data()

print('learning...')
meme.learn()

print('show off...')

while True:
    observation = env.reset()
    for t in range(1000**2):
        env.render()
        action = meme.getAction(observation)
        observation, reward, done, info = env.step(action)
        print('action:', action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
