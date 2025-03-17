import gym
import numpy as np

env =gym.make('Acrobot-v1')
env.reset()
while(1):
    print(np.power(0.99,2))