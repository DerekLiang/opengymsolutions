import gym
env = gym.make('Pendulum-v0')

observation = env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
