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

        for moves  in itertools.product([0,1], repeat=4):
            o = obs[:-learnDepth]
            co = currentOb
            for m in moves:
                o.append((co, m))
                states = []
                for j in range(learnDepth):
                    (po,pa) = observations[i-1-j]
                    state = np.insert(po, len(po), pa-0.5)
                    states.append(state)

                co = self.model.predict(np.array([states]))[0]

            resultMove.append(moves[0])
            resultScore.append(self.score(o))

        bestMove = resultMove[ np.argmin(np.array(resultScore)) ]
        predictState = self.model.predict(np.array([ np.insert(ob, len(ob), bestMove-0.5) ]))[0]
        return (bestMove, predictState)

    def score(self, ob):
        return ob[0]*ob[0] + 50*ob[2]*ob[2]

    def learn(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(5,),  kernel_initializer='random_uniform', bias_initializer='zeros'))
        #model.add(Dense(512, activation='relu', input_shape=(4,),  kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(4))

        model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])
        model.summary()
        self.model = model

        states = bcolz.carray(rootdir='data/cartpole_state', mode='r')
        targets = bcolz.carray(rootdir='data/cartpole_target', mode='r')
        states = states.reshape((-1,5*learnDepth))
        targets = targets.reshape((-1,4))

        self.model.fit(states, targets, shuffle=True, epochs=180, verbose=1)

        self.model.save('data/cartpole.model')

    def loadModel(self):
        self.model = load_model('data/cartpole.model')

env = gym.make('CartPole-v0')
meme = Meme()

def generate_data():
    states = bcolz.carray([], rootdir='data/cartpole_state', mode='w')
    targets = bcolz.carray([], rootdir='data/cartpole_target', mode='w', dtype='i')
    sample = 1
    while sample < 100:
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
                state = np.insert(po, len(po), pa-0.5)
                states.append(state)

            (co,_) = observations[i]
            targets.append(co)

        states.flush()
        targets.flush()
        sample += 1


#print('generating learning data')
#generate_data()

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
            (action, predict) = meme.getAction(observation[:-learnDepth])
        else:
            action = env.action_space.sample()

        obs.append((observation, action))
        observation, reward, done, info = env.step(action)

        if len(obs) > learnDepth:
            print(action, observation, (observation - predict)*100/observation)

        if abs(observation[2])> 0.17:
            print('dangerous')
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
