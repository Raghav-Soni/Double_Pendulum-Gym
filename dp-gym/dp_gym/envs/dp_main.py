import gym
from gym import error,spaces,utils
from gym.utils import seeding
import time
import random
import numpy as np
from dp_gym.envs.src.dp_sim import dp_simulation

class dp_gym(gym.Env):
    
    def __init__(self, design = "design_C.0", model = "model_3.0", robot = "pendubot", render = False, dt = 0.005, mode = 0):

        self.design = design
        self.model = model
        self.robot = robot
        self.render = render

        self.mode = mode        # mode = 0 for swing up and 1 for stabilising at the top

        self._action_dim = 2

        self.dt = dt
        self.t = 0

        action_high = np.array([1.0])
        action_low = -action_high         
        
        self.action_space = spaces.Box(action_low, action_high)

        self._obs_dim = 12
        observation_high = np.array([1] * self._obs_dim)
        observation_low = -observation_high

        self.observation_space = spaces.Box(observation_low, observation_high)

        self.roa = [165*np.pi/180, 165*np.pi/180]  #Region of attraction for which stabilising controller is trained

        self.obs_buffer = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

        self.dp = dp_simulation(self.design, self.model, self.robot, self.render, self.dt, self.mode, self.roa)

        self.max_vel = 30  #rad/sec
        self.max_tq = 6    #Newtom-meter

        



    def step(self, action):
        #assuming the actions are normalized, raw torque values  

        #Scaling the action values to torque limits

        tq = np.array([0.0, 0.0])
        if(self.robot == "pendubot"):
            tq[0] = self.max_tq*action[0]
        if(self.robot == "acrobot"):
            tq[1] = self.max_tq*action[0]

        self.dp.step(tq)  
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
        state = self.dp.get_state()[1]

        max_vel_flag = False
        out_roa_flag = False
        if(abs(state[2]) > self.max_vel or abs(state[3]) > self.max_vel):
            max_vel_flag = True
        

        a1_cos = np.cos(state[0])
        a1_abs = np.arccos(a1_cos)

        a2_cos = np.cos(state[1])
        a2_abs = np.arccos(a2_cos)

        if(self.mode == 1):
            if(a1_abs < self.roa[0] or a2_abs < self.roa[1]):
                out_roa_flag = True


        if(self.mode == 0):   #Swing up
            reward = 0.001*(np.pi-a1_abs)*state[2] + 0.005*(np.pi-a2_abs)*state[3] + 3*(a1_abs) + 6*(a2_abs) #Need to calculate reward based on the state
        else:    #Stabilise
            reward = -0.001*state[2] - 0.005*state[3] + 3*(a1_abs - np.pi) + 6*(a2_abs - np.pi)

        if(max_vel_flag == True):
            reward -= 300   #0 should be replaced by a high negative value
        
        if(out_roa_flag == True):
            reward -= 500
        


        if(self.t > 15 or max_vel_flag == True or out_roa_flag == True):   #Each episode will be of 1 minute, if swinged up in time, good enough, otherwise end
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
        a1_abs = np.arccos(a1_cos)

        a2_cos = np.cos(state[1][1])
        a2_abs = np.arccos(a2_cos)

        self.obs_buffer[2][0] = a1_abs/np.pi
        self.obs_buffer[2][1] = a2_abs/np.pi
        self.obs_buffer[2][2] = state[1][2]/self.max_vel
        self.obs_buffer[2][3] = state[1][3]/self.max_vel


        return np.concatenate((self.obs_buffer[0], self.obs_buffer[1], self.obs_buffer[2]))



