import gym

import numpy as np
from scipy.sparse import *
from scipy import *

class State():
    def __init__(self, observation, *args, **kwargs):
        self.__serialize_str = self.__internal_serialize_str(observation)
        self.visited_count = 0
        self.value = 0
        self.actions = {0: {}, 1: {}}
        self.priority = 0

    def __internal_serialize_str(self, observation):
        return "s {0:5.1f} {1:5.1f} {2:5.1f} {3:5.1f}".format(*observation)

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
    def __init__(self, *args, **kwargs):
        self.states = {}

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
        return value / count if count != 0 else 0

    def get_next_action(self, observation):
        state = self.find_state(observation)
        return self.get_next_action_by_state(state)

    def get_next_best_action(self, observation):
        state = self.find_state(observation)
        return self.get_next_action_by_state(state, N0=1e-10)

    def get_next_action_by_state(self, state, N0=10):
        e = N0/(N0 + state.visited_count)
        if np.random.rand()>e or N0 == 1e-10:
            q = [ self.get_action_value(state, x) for x in range(2) ]
            if q[0] != q[1]:
                return np.argmax(q)
        return np.random.randint(2)

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

env = gym.make('CartPole-v0')
game = Game()

# s0 = (0,0,0,0)
# s1 = (1,1,1,1)
# s2 = (2,2,2,2)

explore_episode_length = []
exploit_episode_length = []
while True:
    newObservation, counter, try_best_action = env.reset(), 0, len(explore_episode_length) % 10 + 1 == 10

    if try_best_action:

        while True:
            while True:
                action = game.get_next_best_action(newObservation)
                newObservation, reward, done, info, counter = env.step(action) + (counter+1, )
                if done:
                    exploit_episode_length.append(counter)
                    print("Try best action -- Episode finished after {0} timesteps. goal {1:5.1f} of 195.".format(counter, np.average(exploit_episode_length[-100:])))
                    break

            newObservation, counter, preCounter = env.reset(), 0, counter
            if preCounter<195:
                break

    while True:
        env.render()
        action = game.get_next_action(newObservation)
        newObservation, reward, done, info, counter, preObservation = env.step(action) + (counter+1, newObservation)
        game.update_value(preObservation, action, newObservation, 1 if not done or counter == 200 else 0)
        if done:
            game.backup()
            explore_episode_length.append(counter)
            print("Episode finished after {0} timesteps, average length of last 5/10 episode {1:5.1f}/{2:5.1f},".format(counter, np.average(explore_episode_length[-5:]), np.average(explore_episode_length[-10:])))
            break
