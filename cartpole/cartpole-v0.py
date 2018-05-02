import gym

import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import bcolz
import itertools

learnDepth = 3

class Meme():
    def getAction(self, obs, currentOb):
        resultMove = []
        resultScore = []
        predStates = []

        for moves in itertools.product([0,1], repeat=4):
            o = obs.copy()
            co = currentOb
            for idx, m in enumerate(moves):
                o.append((co, m))
                states = np.array([])
                for j in range(learnDepth):
                    (po,pa) = o[len(o)-1-j]
                    states = np.insert(states, len(states), po)
                    states = np.insert(states, len(states), pa-0.5)

                co = self.model.predict(np.array([states]))[0]
                if idx==0:
                    predStates.append(co)

            resultMove.append(moves[0])
            resultScore.append(self.score(co))

        bestMove = resultMove[ np.argmin(resultScore) ]
        predictState = predStates[ np.argmin(resultScore) ]
        return (bestMove, predictState)

    def score(self, pred):
        return pred[0]*pred[0] + 10*pred[2]*pred[2]

    def learn(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(15,),  kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(512, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(4))

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        self.model = model

        states = bcolz.carray(rootdir='data/cartpole_state', mode='r')
        targets = bcolz.carray(rootdir='data/cartpole_target', mode='r')
        states = states.reshape((-1,5*learnDepth))
        targets = targets.reshape((-1,4))

        print('total targets', len(targets))
        self.model.fit(states, targets, shuffle=True, epochs=250, verbose=1)

        self.model.save('data/cartpole.model')

    def loadModel(self):
        self.model = load_model('data/cartpole.model')

env = gym.make('CartPole-v0')
meme = Meme()

def generate_data():
    states = bcolz.carray([], rootdir='data/cartpole_state', mode='w')
    targets = bcolz.carray([], rootdir='data/cartpole_target', mode='w', dtype='i')
    sample = 0
    while sample < 2000:
        observation = env.reset()
        observations = []
        for t in range(1000):
            env.render()
            action = env.action_space.sample()
            observations.append((observation, action))
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

        for i in range(learnDepth, len(observations)):
            for j in range(learnDepth):
                (po,pa) = observations[i-1-j]
                states.append(po)
                states.append(pa-0.5)

            (co,_) = observations[i]
            targets.append(co)

        states.flush()
        targets.flush()
        sample += 1


print('generating learning data')
generate_data()

print('learning...')
meme.learn()

print('load model')
meme.loadModel()

print('show off...')
while True:
    observation = env.reset()
    obs = []
    for t in range(1000**2):
        env.render()

        if len(obs) >= learnDepth:
            (action, predict) = meme.getAction(obs[len(obs)-learnDepth:], observation)
        else:
            action = env.action_space.sample()

        obs.append((observation, action))
        observation, reward, done, info = env.step(action)

        if len(obs) > learnDepth:
            print(action, observation, predict, (observation - predict)*100/observation)

        if abs(observation[2])> 0.17:
            print('dangerous')
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
