import gym

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils

class Meme():
    def __init__(self):
        self.observations = []
        self.targts = []
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(4,)))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001),
                    metrics=['accuracy'])
        self.model = model

    def saveTraining(self, observation, action):
        self.observations.append(observation)
        self.targts.append(action)

    def getAction(self, obs):
        return np.argmax(np.array(self.model.predict(np.array([observation])))[0])

    def learn(self, done):
        if not done:
            self.observations = self.observations[:-3]
            self.targts = self.targts[:-3]
        self.model.fit(np.array(self.observations), np_utils.to_categorical(self.targts, 2), epochs=3, verbose=1)

env = gym.make('CartPole-v0')
meme = Meme()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = meme.getAction(observation)
        print(observation, action)
        meme.saveTraining(observation, action)
        #action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    meme.learn(done)