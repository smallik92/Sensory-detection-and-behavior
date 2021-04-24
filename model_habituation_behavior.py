import numpy as np
from numpy import matlib as mb
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import pylab
import math
mpl.style.use('bmh')
from scipy.stats import multivariate_normal

import model_habituation
from model_habituation import get_tuning_matrix, get_full_system_params, sensory_module

# GLOBAL PARAMETERS

#Information along temporal axis
DT = 0.1
TRIAL_DUR = 60
T_INIT = 5
T_ON = 20
T_OFF = TRIAL_DUR - (T_INIT+T_ON)
NUM_TRIALS = 25
T_END = TRIAL_DUR*NUM_TRIALS

#Dimensionality of different stages of signal processing
N_DIM = 5
M_DIM = 2
TOTAL_DIM = 3*M_DIM+N_DIM
NUM_BEHAV = 3

#Penalty on different performance measures
ERROR_PENALTY = 10
ENERGY_PENALTY = 2
SMOOTHNESS_PENALTY = 0.1

def create_stimulus():
    t_index = np.arange(0,T_END,DT)

    z_target = np.zeros((M_DIM, int(TRIAL_DUR/DT)))
    z_target[0, int(T_INIT/DT):int((T_INIT+T_ON)/DT)] = 1
    z_stim = mb.repmat(z_target, 1 , NUM_TRIALS)
    return z_stim

class behavioral_module(object):

    def __init__(self, transition_mat, valence_mat, uncertainty, weights, stim_1_loc, stim_2_loc):
        self.transition_mat = transition_mat
        self.valence_mat = valence_mat
        self.nominal_uncertainty = uncertainty
        self.weight_nu = weights[0]
        self.weight_gamma = weights[1]

        self.stim_1_loc = stim_1_loc
        self.stim_2_loc = stim_2_loc
        self.t_index = np.arange(0,T_END,DT)
        
    def sensory_input_to_behav(self, nu_response, gamma_response):
        sensory_input_agg = self.weight_nu*nu_response+self.weight_gamma*gamma_response
        noise = np.squeeze(np.random.multivariate_normal([0,0], 0.1*np.identity(M_DIM), size = (1, int(T_END/DT))))
        return sensory_input_agg + noise.T

    def forward_sim_behav(self, nu_response, gamma_response, behav_init):

        sensory_input = self.sensory_input_to_behav(nu_response, gamma_response)

        prob_sensory_1 = np.zeros((NUM_BEHAV, int(T_END/DT)))
        prob_sensory_2 = np.zeros((NUM_BEHAV, int(T_END/DT)))

        behav_response = np.zeros((NUM_BEHAV, int(T_END/DT)))
        behav_response[:,[0]] = behav_init

        mvn_1_1 = multivariate_normal(self.stim_1_loc, 0.75*self.nominal_uncertainty*np.identity(M_DIM))
        mvn_2_1 = multivariate_normal(self.stim_1_loc, self.nominal_uncertainty*np.identity(M_DIM))
        mvn_3_1 = multivariate_normal(self.stim_1_loc, 1.5*self.nominal_uncertainty*np.identity(M_DIM))

        mvn_1_2 = multivariate_normal(self.stim_2_loc, 0.75*self.nominal_uncertainty*np.identity(M_DIM))
        mvn_2_2 = multivariate_normal(self.stim_2_loc, self.nominal_uncertainty*np.identity(M_DIM))
        mvn_3_2 = multivariate_normal(self.stim_2_loc, 1.5*self.nominal_uncertainty*np.identity(M_DIM))

        valence_sum = self.valence_mat.sum(axis= 1)

        for t in range(1, int(T_END/DT)):
            sensory_input_t = sensory_input[:,[t]].T

            prob_sensory_1[0, [t]] = mvn_1_1.pdf(sensory_input_t)
            prob_sensory_1[1, [t]] = mvn_2_1.pdf(sensory_input_t)
            prob_sensory_1[2, [t]] = mvn_3_1.pdf(sensory_input_t)

            prob_sensory_2[0, [t]] = mvn_1_2.pdf(sensory_input_t)
            prob_sensory_2[1, [t]] = mvn_2_2.pdf(sensory_input_t)
            prob_sensory_2[2, [t]] = mvn_3_2.pdf(sensory_input_t)

            behav_response[0, [t]] = ((self.valence_mat[0,0]*prob_sensory_1[0,[t]] + self.valence_mat[0,1]*prob_sensory_2[0,[t]])/valence_sum[0])*\
                                    np.matmul(self.transition_mat[0,:], behav_response[:, [t-1]])

            behav_response[1, [t]] = ((self.valence_mat[1,0]*prob_sensory_1[1,[t]] + self.valence_mat[1,1]*prob_sensory_2[1,[t]])/valence_sum[1])*\
                                    np.matmul(self.transition_mat[1,:], behav_response[:, [t-1]])

            behav_response[2, [t]] = ((self.valence_mat[2,0]*prob_sensory_1[2,[t]] + self.valence_mat[2,1]*prob_sensory_2[2,[t]])/valence_sum[2])*\
                                    np.matmul(self.transition_mat[2,:], behav_response[:, [t-1]])

            total_behav = behav_response[0,[t]]+behav_response[1,[t]]+behav_response[2,[t]]

            temp1 = behav_response[0,[t]]/total_behav
            temp2 = behav_response[1,[t]]/total_behav
            temp3 = behav_response[2,[t]]/total_behav

            behav_response[0, [t]] = temp1
            behav_response[1, [t]] = temp2
            behav_response[2, [t]] = temp3

        return behav_response  

class action_module(object):

    def __init__(self, t_index, subsample_rate, surge, hover, cast, start_pos, target_pos, spread):
        self.t_index = t_index
        self.ss_rate = subsample_rate   
        self.surge = surge
        self.hover = hover
        self.cast = cast
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.spread = spread

    def subsample_behavior(self, behav_response):
        behav_response_ss = behav_response[:,1:int(T_END/DT)-1:self.ss_rate]
        return behav_response_ss

    def concentration_field(self, axis_min, axis_max, delta = 0.1):
        x_scale = y_scale = np.arange(axis_min, axis_max, delta)
        X, Y = np.meshgrid(x_scale,y_scale)
        Z = np.exp(-(1/self.spread**2)*((X-self.target_pos[0])**2 + (Y - self.target_pos[1])**2))
        return Z        

    def concentration_grad(self, pos):
        grad_conc_x = -(1/self.spread**2)*(pos[0] - self.target_pos[0])
        grad_conc_y = -(1/self.spread**2)*(pos[1] - self.target_pos[1])
        return grad_conc_x, grad_conc_y

    def forward_sim_action(self, behav_response):

        behav_response_ss = self.subsample_behavior(behav_response)
        pos = np.zeros((2, len(behav_response_ss[0])))
        pos[:,[0]] = self.start_pos
        actions = [0, 1, 2]

        for i in range(1, len(behav_response_ss[0])):
            # action_idx = np.argmax(behav_response[:,[i]])
            action_idx = np.random.choice(actions, p = np.reshape(behav_response[:,[i-1]], (NUM_BEHAV,)))
            grad_conc_x, grad_conc_y = self.concentration_grad(pos[:,[i-1]])
            grad = np.array([grad_conc_x, grad_conc_y])
            # print(grad)
            if action_idx == 0:
                pos[:,[i]] = pos[:,[i-1]]+DT*self.ss_rate*self.surge*grad+0.005*np.random.randn()
            elif action_idx == 1:
                pos[:,[i]] = pos[:,[i-1]]+DT*self.ss_rate*self.hover*grad+0.005*np.random.randn()
            elif action_idx == 2:
                pos[:,[i]] = pos[:,[i-1]]+DT*self.ss_rate*self.cast*grad+0.005*np.random.randn()

        return pos

def run_compiled_prog():

    # Parameters of the sensory model
    drift_nu = -0.2852
    drift_nu_gamma = 0.1925
    drift_gamma = -0.0466
    tau_nu = 1.0168
    tau_gamma = 14.5714
    beta = 0.3059

    # Create sensory system and get its output
    z_stim = create_stimulus()

    gamma_init = np.zeros((M_DIM, 1))
    nu_init = np.zeros((M_DIM, 1))
    x_init = np.zeros((N_DIM,1))

    sensory_model = sensory_module(drift_nu, drift_nu_gamma, drift_gamma, tau_nu, tau_gamma, beta)
    x_response, nu_response, gamma_response = sensory_model.forward_sim(z_stim, x_init, nu_init, gamma_init)

    transition_mat = np.array([[0.75, 0.15, 0.1], [0.1, 0.7, 0.2], [0.15, 0.15, 0.7]])
    valence_mat = np.array([[0.3, 0.6], [0.35, 0.2], [0.35, 0.2]])
    uncertainty = 0.2
    weights = np.array([0.75, 0.25])
    stim_1_loc = np.array([0, 0])
    stim_2_loc = np.array([1, 0])

    print('Simulating behavior..................')
    behav_model = behavioral_module(transition_mat, valence_mat, uncertainty, weights, stim_1_loc, stim_2_loc)

    behav_init = (1/3)*np.ones((NUM_BEHAV, 1))
    behav_response = behav_model.forward_sim_behav(nu_response, gamma_response, behav_init)
    print('Done..................')

    t_index = np.arange(0,T_END,DT)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10), sharex=True)
    ax1.fill_between(t_index/60, behav_response[0,:].T, facecolor = 'gray', alpha = 0.75)
    ax1.plot(t_index/60, behav_response[0,:].T, ':', linewidth = 0.35, color = 'black')
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('Forward behavior')
    ax1.set_xlim([0, T_END/60])

    ax2.fill_between(t_index/60, behav_response[1,:].T, facecolor = 'red', alpha = 0.75)
    ax2.plot(t_index/60, behav_response[1,:].T, ':', linewidth = 0.35, color = 'black')
    ax2.set_ylabel('Pause behavior') 
    ax2.set_xlim([0, T_END/60]) 
    ax2.set_ylim([0, 1])

    ax3.fill_between(t_index/60, behav_response[2,:].T, color = 'salmon', alpha = 0.75)
    ax3.plot(t_index/60, behav_response[2,:].T, ':', linewidth = 0.35, color = 'black')
    ax3.set_xlabel('Time in mins')
    ax3.set_ylabel('Reverse behavior')
    ax3.set_xlim([0, T_END/60]) 
    ax3.set_ylim([0, 1])
    plt.show() 

    # print(behav_response.shape)

    # plt.figure()
    # plt.imshow(behav_response[:,:int(TRIAL_DUR/DT)], extent = [0, TRIAL_DUR, 3, 1])
    # plt.colorbar()
    # plt.gca().invert_yaxis()
    # plt.show()

    subsample_rate = 1
    surge = 0.1
    hover = 0.01*np.random.randn()
    cast = -0.1
    start_pos = np.array([[0], [0.2]])
    target_pos = np.array([[1], [1]])
    spread = 1

    print('Simulating action....................')
    action_model = action_module(t_index, subsample_rate, surge, hover, cast, start_pos, target_pos, spread)
    axis_min, axis_max = 0, 2
    odor_conc = action_model.concentration_field(axis_min, axis_max, 0.01)
    action_response = action_model.forward_sim_action(behav_response)
    print('Done................................')

    plt.figure(figsize=(10,10))
    plt.plot(start_pos[0], start_pos[1], marker = '*', markersize = 10)
    plt.plot(action_response[0, :], action_response[1, :], color = 'black', linewidth = 0.5)
    # plt.plot(action_response[0, :], action_response[1, :], marker = '+', markersize = 0.5, color = 'black')
    plt.plot(action_response[0,-1], action_response[1,-1], marker = '*', markersize = 10, color = 'red')
    plt.imshow(odor_conc, extent = [axis_min, axis_max, axis_max, axis_min])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Movement in a 2D plane')
    plt.show()

run_compiled_prog()


