# TODO：选择一个gym环境, 构建它的状态转移模型、回报模型, 写出代码对应的MDP元素（S,A,P,R,gamma）
import time
import random
import numpy as np
import gymnasium as gym

env = gym.make("Taxi-v3",render_mode="human")
env.reset()
while(1):
    env.render()
    env.step(env.action_space.sample())