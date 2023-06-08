import gym
from gym import error,spaces,utils
from gym.utils import seeding
import time
import random
import numpy as np
from dp_gym.envs.src.dp_sim import dp_simulation

class dp_gym(gym.Env):
    
    def __init__(self, design = "design_C.0", model = "model_3.0", robot = "acrobot", render = False, dt = 0.005):

        self.design = design
        self.model = model
        self.robot = robot
        self.render = render

        self._action_dim = 2

        self.dt = dt
        self.t = 0

        if(robot == "acrobot"):
            action_high = np.array([0.0, 1.0])
            action_low = -action_high
        if(robot == "pendubot"):
            action_high = np.array([1.0, 0.0])
            action_low = -action_high            
        
        self.action_space = spaces.Box(action_low, action_high)

        self._obs_dim = 12
        observation_high = np.array([1] * self._obs_dim)
        observation_low = -observation_high

        self.observation_space = spaces.Box(observation_low, observation_high)

        self.obs_buffer = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

        self.dp = dp_simulation(self.design, self.model, self.robot, self.render, self.dt)

        self.max_vel = 20  #rad/sec
        self.max_tq = 6    #Newtom-meter



    def step(self, action):
        #assuming the actions are normalized, raw torque values  

        #Scaling the action values to torque limits
        action[0] = self.max_tq*action[0]
        action[1] = self.max_tq*action[1]

        self.dp.step(action)  
        self.t += self.dt

        observation = self.get_obs()
        reward, done = self._caclulate_reward()

        return observation, reward, done, {}

    def reset(self):
        self.t = 0
        self.dp.reset_state()
        observation = self.get_obs()     

        return observation

    def render(self, mode='human'):
        pass

    def _caclulate_reward(self):
        state = self.dp.get_state()

        max_vel_flag = False
        if(abs(state[1][2]) > self.max_vel or abs(state[1][3]) > self.max_vel):
            max_vel_flag = True


        reward = 0 #Need to calculate reward based on the state
        if(max_vel_flag == True):
            reward -= 0    #0 should be replaced by a high negative value

        if(self.t > 60 or max_vel_flag == True ):   #Each episode will be of 1 minute, if swinged up in time, good enough, otherwise end
            done = True
        else:
            done = False
        
        return reward , done

    def get_obs(self):
        state = self.dp.get_state()  #State in of the form [ang1, ang2, vel1, vel2]   

        #Need to learn about state and normalise it before the final code

        self.obs_buffer[0] = self.obs_buffer[1]
        self.obs_buffer[1] = self.obs_buffer[2]

        a1_cos = np.cos(state[1][0])
        a1 = np.arccos(a1_cos)

        a2_cos = np.cos(state[1][1])
        a2 = np.arccos(a2_cos)

        self.obs_buffer[2][0] = a1/np.pi
        self.obs_buffer[2][1] = a2/np.pi
        self.obs_buffer[2][2] = state[1][2]/self.max_vel
        self.obs_buffer[2][3] = state[1][3]/self.max_vel


        return np.concatenate((self.obs_buffer[0], self.obs_buffer[1], self.obs_buffer[2]))



