import os
import gym
import dp_gym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = gym.make("dp_gym-v0", render = True, robot = "pendubot")

model = PPO.load("../logs/18Jul4/model_169.zip")
obs = env.reset()
total_reward = 0
while True:
    action, _states = model.predict(obs, deterministic = True)
    # print(action)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards
    if(dones == True):
        print(total_reward)
        break