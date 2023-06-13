import os
from datetime import datetime
import numpy as np
import yaml
import random

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory

from dp_gym.envs.src.data.dp_anim import dp_plot

class dp_simulation:

    def __init__(self, design, model, robot, render, dt, mode, roa):

        # # model parameters
        if robot == "acrobot":
            torque_limit = [0.0, 6.0]
        if robot == "pendubot":
            torque_limit = [6.0, 0.0]

        self.render = render
        self.mode = mode
        self.roa = roa


        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_par_path = dir_path + "/data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
        mpar = model_parameters(filepath=model_par_path)

        l1 = mpar.l[0]
        l2 = mpar.l[1]


        mpar_con = model_parameters(filepath=model_par_path)

        # simulation parameter
        self.dt = dt
        t_final = 5.0  # 4.985
        integrator = "runge_kutta"
        start = [0., 0., 0., 0.]
        goal = [np.pi, 0., 0., 0.]

        # noise
        process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
        meas_noise_sigmas = [0.0, 0.0, 0.05, 0.05]
        delay_mode = "vel"
        delay = 0.01
        u_noise_sigmas = [0.01, 0.01]
        u_responsiveness = 1.0
        perturbation_times = []
        perturbation_taus = []

        # filter args
        meas_noise_vfilter = "none"
        meas_noise_cut = 0.1
        filter_kwargs = {"lowpass_alpha": [1., 1., 0.3, 0.3],
                        "kalman_xlin": goal,
                        "kalman_ulin": [0., 0.],
                        "kalman_process_noise_sigmas": process_noise_sigmas,
                        "kalman_meas_noise_sigmas": meas_noise_sigmas,
                        "ukalman_integrator": integrator,
                        "ukalman_process_noise_sigmas": process_noise_sigmas,
                        "ukalman_meas_noise_sigmas": meas_noise_sigmas}


        # create save directory
        # timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
        # save_dir = os.path.join("data", robot, "ilqr", "mpc", timestamp)
        # os.makedirs(save_dir)

        plant = SymbolicDoublePendulum(model_pars=mpar)

        self.sim = Simulator(plant=plant)
        self.sim.set_process_noise(process_noise_sigmas=process_noise_sigmas)
        self.sim.set_measurement_parameters(meas_noise_sigmas=meas_noise_sigmas,
                                    delay=delay,
                                    delay_mode=delay_mode)
        self.sim.set_motor_parameters(u_noise_sigmas=u_noise_sigmas,
                                u_responsiveness=u_responsiveness)

        if(self.render == True):
            self.anim = dp_plot(l1, l2, self.dt)

    def step(self, action):
        self.sim.step(action, self.dt)
        state = self.sim.get_state()
        if(self.render == True):
            self.anim.animate_step(state[1])

    def get_state(self):
        return self.sim.get_state()

    def reset_state(self):
        time = 0
        if(self.mode == 0): #Swing up
            state = np.array([0.0, 0.0, 0.0, 0.0])
        else:   #Stabilising on the top
            state = np.array([random.uniform(self.roa[0], 2*np.pi - self.roa[0]), random.uniform(self.roa[1], 2*np.pi - self.roa[1]), 0.0, 0.0])
        self.sim.set_state(time, state)
