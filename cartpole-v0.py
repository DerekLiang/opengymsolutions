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
        model.add(Dense(512, activation='relu', input_shape=(4,),  kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001),
                    metrics=['accuracy'])
        self.model = model

    def saveTraining(self, observation, action):
        self.observations.append(observation)
        self.targts.append(action)

    def getAction(self, obs):
        return np.argmax(np.array(self.model.predict(np.array([observation])))[0])

    def learn(self):
        self.model.fit(np.array(self.observations), np_utils.to_categorical(self.targts, 2), shuffle=True, epochs=20, verbose=1)

env = gym.make('CartPole-v0')
meme = Meme()

for i_episode in range(200):
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

    if len(observations) > 6:
        [ meme.saveTraining(o,a) for (o,a) in observations]

print('learning...')
meme.learn()

print('show off...')
observation = env.reset()

while True:
    env.render()
    action = meme.getAction(observation)
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
