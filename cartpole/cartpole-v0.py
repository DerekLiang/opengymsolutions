import gym

import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import bcolz
import itertools

class Meme():
    def getAction(self, ob):
        resultMove = []
        resultScore = []
        for moves  in itertools.product([0,1], repeat=4):
            o = ob
            for m in moves:
                s = np.insert(o, len(o), m-0.5)
                o = self.model.predict(np.array([s]))[0]
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

        model.compile(loss='mse', optimizer='adadelta',
                    metrics=['accuracy'])
        model.summary()
        self.model = model

        states = bcolz.carray(rootdir='data/cartpole_state', mode='r')
        targets = bcolz.carray(rootdir='data/cartpole_target', mode='r')
        states = states.reshape((-1,5))
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
            #print(observation, action)
            observations.append((observation, action))
            #meme.saveTraining(observation, action)

            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

        for i in range(1, len(observations)):
            (po,pa) = observations[i-1]
            (co,_) = observations[i]
            state = np.insert(po, len(po), pa-0.5)
            states.append(state)
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
    for t in range(1000**2):
        env.render()
        (action, predict) = meme.getAction(observation)
        observation, reward, done, info = env.step(action)
        print(action, observation, (observation - predict)*100/observation)
        if abs(observation[2])> 0.17:
            print('dangerous')
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
