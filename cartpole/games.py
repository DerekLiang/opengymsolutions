import gym

import numpy as np
from scipy.sparse import *
from scipy import *

env = gym.make('CartPole-v0')

def __digitize(v, lo, hi, granula):
    assert(v<hi and v>lo)
    return int((v-lo) / ((hi-lo)/granula))

def __toState(o):
    a = __digitize(o[0], env.observation_space.low[0], env.observation_space.high[0], 10)
    b = __digitize(np.tanh(o[1]), -1, 1, 10)
    c = __digitize(o[2], env.observation_space.low[2], env.observation_space.high[2], 10)
    d = __digitize(np.tanh(o[3]), -1, 1, 10)

    return (d+(c+(b+a*10)*10)*10,)

N0 = 100
values = np.zeros((10**4), dtype=np.int16)
visitedCount = np.zeros((10**4), dtype=np.int16)
actionCount = np.zeros((10**4, 2, 10**4), dtype=np.int16)

while True:
    observation = env.reset()
    counter = 0
    while True:
        env.render()

        s = __toState(observation)
        e = N0/(N0 + visitedCount[s])
        if np.random.rand()<=e:
            action = np.random.randint(2)
        else:
            q = [ (values*actionCount[s + (x,)]).sum() for x in range(2) ]
            action = np.random.randint(2) if q[0]==q[1] else np.argmax(q)

        observation, reward, done, info, counter = env.step(action) + (counter+1,)

        visitedCount[s] += 1
        actionCount[ s + (action,) + s] += 1
        values[s] += (reward -values[s])/visitedCount[s]

        if done:
            s = __toState(observation)
            visitedCount[s] += 1
            values[s] += (reward -values[s])/visitedCount[s]

            print("Episode finished after {0} timesteps".format(counter))
            break
