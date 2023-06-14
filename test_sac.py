import os
import gym
import dp_gym
import time

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

env = gym.make("dp_gym-v0", render = True, robot = "pendubot")

model = SAC.load("../logs/stabilise/14JunPS1/model_10.zip") 
obs = env.reset()
total_reward = 0
i = 0
while True:
    action, _states = model.predict(obs, deterministic = True)
    # print(action)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards
    if(dones == True):
        print(total_reward)
        break