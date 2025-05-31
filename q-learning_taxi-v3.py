import gym
import random

env = gym.make("Taxi-v3", render_mode="ansi")
env.reset()

row = random.randint(0, 4)
column = random.randint(0, 4)
index = random.randint(0, 3)
destination = random.randint(0, 3)
state = env.unwrapped.encode(row, column, index, destination)

print("State:", state)
env.unwrapped.s = state  # Setting the environment state directly

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

print(env.unwrapped.P[state])  # Access transition probabilities

output = env.render()
print(output)
