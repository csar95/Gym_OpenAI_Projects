import gym

env = gym.make('FrozenLake-v0')  # Left: 0 | Down: 1 | Right: 2 | Up: 3
env.reset()

for _ in range(10):
    env.render()
    env.step(env.action_space.sample())  # Take a random action
