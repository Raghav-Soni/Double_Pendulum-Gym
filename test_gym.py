import os
import gym
import dp_gym
import time

env = gym.make("dp_gym-v0", render = True, robot = "pendubot")
obs = env.reset()
time.sleep(5)
actions = [-1, 0]
obs, reward, done, _ = env.step(actions)
while(True):
    obs, reward, done, _ = env.step(actions)
    # actions[0] += 0.01
    # actions[1] -= 0.01
    if done==True:
        obs = env.reset()
        done = False
    # print(obs)