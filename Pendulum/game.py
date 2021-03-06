import gym

import numpy as np
from scipy.sparse import *
from scipy import *

class Action():
    def __init__(self):
        self.min, self.max, self.step = -2.0, 2.0, 0.5

    def get_all_actions(self):
        s = self.min
        while s <= self.max + self.step - 0e-10:
            yield round(s, 1)
            s += self.step

    def get_all_action_map(self):
        r = {}
        for action in self.get_all_actions():
            r[action] = {}
        return r

class State():
    def __init__(self, observation, *args, **kwargs):
        self.__serialize_str = self.__internal_serialize_str(observation)
        self.visited_count = 0
        self.value = -16.2736044
        self.actions = Action().get_all_action_map()
        self.priority = 0

    def __internal_serialize_str(self, observation):
        return "s {0:5.2f} {1:5.2f} {2:5.1f}".format(*observation)

    def visit(self, action, observation, reward):
        self.visited_count += 1
        self.value += (reward - self.value) / self.visited_count
        h = self.__internal_serialize_str(observation)
        observations = self.actions[action]
        observations[h] = observations[h] + 1 if h in observations else 1

    def get_visited_observation_count_map_by_action(self, action):
        return self.actions[action]

    def get_visited_action_observation_count(self, action, observation):
        h = self.__internal_serialize_str(observation)
        return self.actions[action][h] if  h in self.actions[action] else 0

    def __str__(self):
        return self.__serialize_str

class Game():
    def __init__(self):
        self.states = {}
        self.actions = [ x for x in Action().get_all_actions()]

    def find_state(self, observation):
        state = State(observation)
        h = str(state)
        if h not in self.states:
            self.states[h] = state
        return self.states[h]

    def get_action_value(self, state, action):
        observation_count_map = state.get_visited_observation_count_map_by_action(action)
        value, count = 0, 0
        for h, visited_count in observation_count_map.items():
            if h in self.states:
                value += self.states[h].value * visited_count
                count +=  visited_count
        return value / count if count != 0 else -16.2736044

    def get_next_action(self, observation):
        state = self.find_state(observation)
        return self.get_next_action_by_state(state)

    def get_next_best_action(self, observation):
        state = self.find_state(observation)
        return self.get_next_action_by_state(state, N0=1e-10)

    def get_next_action_by_state(self, state, N0=1000):
        e = N0/(N0 + state.visited_count)
        if np.random.rand()>e or N0 == 1e-10:
            q = [ self.get_action_value(state, x) for x in self.actions ]
            return self.actions[np.argmax(q)]
        return self.actions[np.random.randint(len(self.actions))]

    def update_value(self, observation, action, newObservation, reward):
        state = self.find_state(observation)
        state.visit(action, newObservation, reward)

    def backup(self, loop = 2):
        lr = 0.1
        for i in range(loop):
            for k in sorted(self.states, key=lambda x: self.states[x].priority):
                state  = self.states[k]
                action = self.get_next_action_by_state(state, N0=1e-10)
                value  = self.get_action_value(state, action)
                difference = value - state.value
                state.value += difference * lr
                state.priority = -abs(difference)


env = gym.make('Pendulum-v0')
game = Game()
newObservation = env.reset()
counter = 0

while True:
    # env.render()
    action = game.get_next_action(newObservation)
    newObservation, reward, done, info, counter, preObservation = env.step([action]) + (counter + 1, newObservation)
    game.update_value(preObservation, action, newObservation, reward)

    if counter % 200 + 1 == 200:
        newObservation = env.reset()

    if counter % 10000 + 1 == 10000:
        print('learning... {0}'.format(counter))
        game.backup()

        practice_reward = 0
        while practice_reward > -250 :
            newObservation = env.reset()
            practice_reward = 0
            for i in range(200):
                env.render()
                action = game.get_next_best_action(newObservation)
                newObservation, reward, done, info = env.step([action])
                practice_reward += reward
            print('practice reward: {0:4.2f}'.format(practice_reward))




