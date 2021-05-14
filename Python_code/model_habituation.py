import numpy as np
from numpy import matlib as mb
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('bmh')
from scipy.stats import norm
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
import scipy.linalg as splinalg
import math
"""
Neural model that displays habituation 
"""
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

#Penalty on different performance measures
ERROR_PENALTY = 10
ENERGY_PENALTY = 2
SMOOTHNESS_PENALTY = 0.1

def get_tuning_matrix(to_state_dim, from_state_dim, spread = 1.5, intensity = 1):

    """
    Creates the tuning matrix between stage 1 and stage 2
    """
    
    mean_1 = from_state_dim/3
    mean_2 = from_state_dim - mean_1 + 1
    
    stim_indices = np.linspace(0, from_state_dim, from_state_dim)
    dim_1_weights = norm.pdf(stim_indices, loc = mean_1, scale = spread)
    dim_2_weights = norm.pdf(stim_indices, loc = mean_2, scale = spread)

    tuning_matrix = np.vstack((dim_1_weights, dim_2_weights))
    return intensity*tuning_matrix

def get_full_system_params(drift_nu, drift_nu_gamma, drift_gamma, tau_nu, tau_gamma, beta, b_matrix):

    """
    Combines all system parameters into system matrices
    """

    A_modified = np.block([[(drift_gamma/tau_gamma)*np.identity(M_DIM), (beta/tau_gamma)*np.identity(M_DIM), np.zeros((M_DIM, N_DIM))],
                            [(drift_nu_gamma/tau_nu)*np.identity(M_DIM), (drift_nu/tau_nu)*np.identity(M_DIM), (1/tau_nu)*b_matrix],
                            [np.zeros((N_DIM, M_DIM)), np.zeros((N_DIM, M_DIM)), np.zeros((N_DIM, N_DIM))]])

    aux_eig_A = -1*(10**-2) #very small negative values on the diagonal for stability
    aux_A_matrix = aux_eig_A*(np.identity(M_DIM))

    A_final = block_diag(A_modified,aux_A_matrix)
    B_final = np.block([[np.zeros((2*M_DIM,N_DIM))], [np.identity(N_DIM)],[np.zeros((M_DIM,N_DIM))]])

    R_control = ENERGY_PENALTY*np.identity(N_DIM)
    Q = ERROR_PENALTY*np.identity(M_DIM)

    Q_modified = np.block([[Q, np.zeros((M_DIM, N_DIM)), -Q],
                        [np.zeros((N_DIM, M_DIM)), R_control,np.zeros((N_DIM, M_DIM))],
                        [-Q, np.zeros((M_DIM, N_DIM)), Q]])
    Q_final = block_diag(np.zeros((M_DIM, M_DIM)), Q_modified)
    R_final = SMOOTHNESS_PENALTY*np.identity(N_DIM)

    return A_final, B_final, Q_final, R_final

class sensory_module(object):

    def __init__(self, drift_nu, drift_nu_gamma, drift_gamma, tau_nu, tau_gamma, beta):

        self.drift_nu = drift_nu
        self.drift_nu_gamma = drift_nu_gamma
        self.drift_gamma = drift_gamma

        self.tau_nu = tau_nu
        self.tau_gamma = tau_gamma

        self.beta = beta
        self.b_matrix = get_tuning_matrix(M_DIM, N_DIM)

        self.t_index = np.arange(0,T_END,DT)

    def get_conn_matrix(self, initial = 'i'):

        A_final, B_final, Q_final, R_final = get_full_system_params(self.drift_nu, self.drift_nu_gamma, self.drift_gamma,
                                                                self.tau_nu, self.tau_gamma, self.beta, self.b_matrix)

        assert(A_final.shape == Q_final.shape)

        K, W = self.solve_lqr_infinite_horizon()
        
        if initial == 'z':
            K0 = np.zeros((TOTAL_DIM, TOTAL_DIM)).flatten()
        elif initial == 'i':
            K0 = np.identity(TOTAL_DIM).flatten()
        elif initial == 'c':
            K0 = np.identity(TOTAL_DIM).flatten()+1.0*np.ravel(K)

        def riccati(t, K):
            
            K = np.reshape(K, (TOTAL_DIM, TOTAL_DIM))
            dKdt = np.matmul(A_final.T, K)+np.matmul(K, A_final) - \
                    np.matmul(np.matmul(K, B_final), np.matmul(np.linalg.inv(R_final), np.matmul(B_final.T, K))) + Q_final
            dKdt = dKdt.flatten()
            return dKdt

        # print(initial)
        sol = solve_ivp(riccati, [0, T_END], K0, t_eval=self.t_index)
        return sol 
    
    def get_conn_matrix_backwards(self):

        A_final, B_final, Q_final, R_final = get_full_system_params(self.drift_nu, self.drift_nu_gamma, self.drift_gamma,
                                                                self.tau_nu, self.tau_gamma, self.beta, self.b_matrix)

        assert(A_final.shape == Q_final.shape)
        total_dim = A_final.shape[0]

        Kf = np.zeros((TOTAL_DIM, TOTAL_DIM))
        
        def riccati(t, K):
            
            K = np.reshape(K, (total_dim, total_dim))
            dKdt = -1*(np.matmul(A_final.T, K)+np.matmul(K, A_final) - \
                    np.matmul(np.matmul(K, B_final), np.matmul(np.linalg.inv(R_final), np.matmul(B_final.T, K))) + Q_final)
            dKdt = dKdt.flatten()
            return dKdt

        sol = solve_ivp(riccati, [T_END, 0], Kf.flatten(), t_eval=self.t_index[::-1])
        return sol 
    
    def solve_lqr_infinite_horizon(self):
        A_final, B_final, Q_final, R_final = get_full_system_params(self.drift_nu, self.drift_nu_gamma, self.drift_gamma,
                                                                self.tau_nu, self.tau_gamma, self.beta, self.b_matrix)

        #Solve the Continuous time Algebraic Riccati Equation
        K = np.matrix(splinalg.solve_continuous_are(A_final, B_final, Q_final, R_final))

        #compute the LQR gain
        W = np.matrix(splinalg.inv(R_final)*(B_final.T*K))
        return K, W 

    def forward_sim_infinite_horizon(self):
        pass            

    def forward_sim(self, z_stim, x_init, nu_init, gamma_init, K):
        A_final, B_final, Q_final, R_final = get_full_system_params(self.drift_nu, self.drift_nu_gamma, self.drift_gamma,
                                                                self.tau_nu, self.tau_gamma, self.beta, self.b_matrix)

        x_response = np.zeros((N_DIM, int(T_END/DT)))
        nu_response = np.zeros((M_DIM, int(T_END/DT)))
        gamma_response = np.zeros((M_DIM, int(T_END/DT)))

        x_response[:,[0]] = x_init
        nu_response[:,[0]] = nu_init
        gamma_response[:,[0]] = gamma_init

        for t in range(int(T_END/DT)-1):

            Kt = np.reshape(K[:,[t]], (TOTAL_DIM, TOTAL_DIM))
            W = - np.matmul(np.linalg.inv(R_final), np.matmul(B_final.T, Kt))

            W_gamma = W[:,:M_DIM]
            W_nu = W[:,M_DIM:2*M_DIM]
            W_x = W[:, 2*M_DIM:2*M_DIM+N_DIM]
            W_z = W[:, 2*M_DIM+N_DIM:]

            gamma_response[:,[t+1]] = gamma_response[:,[t]]+DT*((self.drift_gamma/self.tau_gamma)*gamma_response[:,[t]]+
                                                                (self.beta/self.tau_gamma)*nu_response[:,[t]]+
                                                                0.01*np.random.randn())

            nu_response[:,[t+1]] = nu_response[:,[t]]+DT*((self.drift_nu/self.tau_nu)*nu_response[:,[t]]+
                                                            (self.drift_nu_gamma/self.tau_nu)*gamma_response[:,[t]]+
                                                            (1/self.tau_nu)*np.matmul(self.b_matrix,x_response[:,[t]])+
                                                            0.05*np.random.randn())

            x_response[:,[t+1]] = x_response[:,[t]]+DT*(np.matmul(W_gamma, gamma_response[:,[t]])+
                                                        np.matmul(W_nu, nu_response[:,[t]])+
                                                        np.matmul(W_x, x_response[:,[t]])+
                                                        np.matmul(W_z, z_stim[:,[t]])+ 0.05*np.random.randn())
            
        return x_response, nu_response, gamma_response


def compare_riccati_solutions(t_index, model):
    #Solve the problem using ARE
    K, W = model.solve_lqr_infinite_horizon()

    #Solve the problem using forward in time Riccati equation
    sol_f = model.get_conn_matrix(initial='z')
    K_f = sol_f.y

    #Solve the problem using backward in time Riccati equation
    sol_b = model.get_conn_matrix_backwards()
    K_b = sol_b.y

    # Look at the difference between forward in time and infinite horizon Riccati matrix
    error_f = np.zeros((1, len(t_index)))
    for t in range(len(t_index)):
        K_f_t = np.reshape(K_f[:,[t]], (TOTAL_DIM, TOTAL_DIM))
        # error_f[:,[t]] = np.linalg.norm(K_f_t[0:2*M_DIM+N_DIM, 0:2*M_DIM+N_DIM] - K[0:2*M_DIM+N_DIM, 0:2*M_DIM+N_DIM], ord= 2)
        error_f[:,[t]] = np.linalg.norm(K_f_t  - K , ord= 2)

    fig, ax = plt.subplots(3,1, figsize = (10, 10), sharex=True)
    ax[0].plot(t_index/60, error_f.T)
    ax[0].set_yscale('log')
    ax[0].set_xlim([0, T_END/60])
    ax[0].set_ylabel('log$||K_f(t) - K||$')

    # Look at the difference between backward in time and infinite horizon Riccati matrix
    K_b = np.fliplr(K_b)
    error_b = np.zeros((1, len(t_index)))
    for t in range(len(t_index)):
        K_b_t = np.reshape(K_b[:,[t]], (TOTAL_DIM, TOTAL_DIM))
        error_b[:,[t]] = np.linalg.norm(K_b_t - K, ord = 2)

    ax[1].plot(t_index/60, error_b.T, color = 'blue')
    ax[1].set_yscale('log')
    ax[1].set_xlim([0, T_END/60])
    ax[1].set_ylabel('log$||K_b(t) - K||$')
    # ax[1].title('BPRE')

    error_c = np.zeros((1, len(t_index)))
    for t in range(len(t_index)):
        K_b_t = np.reshape(K_b[:,[t]], (TOTAL_DIM, TOTAL_DIM))
        K_f_t = np.reshape(K_f[:,[t]], (TOTAL_DIM, TOTAL_DIM))
        error_c[:,[t]] = np.linalg.norm(K_b_t - K_f_t , ord = 2)

    ax[2].plot(t_index/60, error_c.T, color = 'red')
    ax[2].set_yscale('log')
    ax[2].set_xlim([0, T_END/60])
    ax[2].set_ylabel('log$||K_b(t)-K_f(t)||$')
    ax[2].set_xlabel('Time in mins') 
    # ax[2].title('Difference between FPRE and BPRE')
    plt.show()


def compare_init_FPRE(t_index, model):

    #Solve the problem using ARE
    K, W = model.solve_lqr_infinite_horizon()

    # Solve the problem using backward in time Riccati equation
    sol_b = model.get_conn_matrix_backwards()
    K_b = sol_b.y
    K_b = np.fliplr(K_b)

    #Solve the problem using forward in time Riccati equation under different initializations
    sol_f_z = model.get_conn_matrix(initial = 'z')
    K_f_z = sol_f_z.y
    # fig = plt.figure()
    # ax1 = plt.subplot(3,1,1)
    # ax1.plot(K_f_z)

    sol_f_i = model.get_conn_matrix(initial = 'i')
    K_f_i = sol_f_i.y
    # ax2 = plt.subplot(3,1,2)
    # ax2.plot(K_f_i)

    sol_f_c = model.get_conn_matrix(initial = 'c')
    K_f_c = sol_f_c.y
    # ax3 = plt.subplot(3,1,3)
    # ax3.plot(K_f_c)
    # plt.show()

    #Look at quality of solutions for different initializations
    error_z = np.zeros((1, len(t_index)))
    error_i = np.zeros((1, len(t_index)))
    error_c = np.zeros((1, len(t_index)))

    error_z_b = np.zeros((1, len(t_index)))
    error_i_b = np.zeros((1, len(t_index)))
    error_c_b = np.zeros((1, len(t_index)))

    for t in range(len(t_index)):

        K_b_t = np.reshape(K_b[:,[t]], (TOTAL_DIM, TOTAL_DIM))

        K_f_z_t = np.reshape(K_f_z[:,[t]], (TOTAL_DIM, TOTAL_DIM))
        K_f_i_t = np.reshape(K_f_i[:,[t]], (TOTAL_DIM, TOTAL_DIM))
        K_f_c_t = np.reshape(K_f_c[:,[t]], (TOTAL_DIM, TOTAL_DIM))

        error_z[:, [t]] = np.linalg.norm(K_f_z_t - K, ord = 2)
        error_i[:, [t]] = np.linalg.norm(K_f_i_t - K, ord = 2)
        error_c[:, [t]] = np.linalg.norm(K_f_c_t - K, ord = 2)

        error_z_b[:, [t]] = np.linalg.norm(K_f_z_t - K_b_t, ord = 2)
        error_i_b[:, [t]] = np.linalg.norm(K_f_i_t - K_b_t, ord = 2)
        error_c_b[:, [t]] = np.linalg.norm(K_f_c_t - K_b_t, ord = 2)   

    fig, (ax1, ax2) = plt.subplots(2,1)
    # ax1 = fig.add_subplot(2,1,1)
    ax1.plot(t_index/60, error_z.T, label = 'K(0) = 0')
    ax1.plot(t_index/60, error_i.T, label = 'K(0) = I', linestyle = 'dotted')
    ax1.plot(t_index/60, error_c.T, label = 'K(0) = I+$\lambda$K', linestyle = 'dashed')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xlim([0, T_END/60])
    ax1.set_ylabel('log$||K_f(t) - K||$')

    # ax2 = fig.add_subplot(2,1,2)
    ax2.plot(t_index/60, error_z_b.T, label = 'K(0) = 0')
    ax2.plot(t_index/60, error_i_b.T, label = 'K(0) = I', linestyle = 'dotted')
    ax2.plot(t_index/60, error_c_b.T, label = 'K(0) = I+$\lambda$K', linestyle = 'dashed')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.set_xlim([0, T_END/60])
    ax2.set_ylabel('log$||K_b(t) - K_f(t)||$')
    
    plt.show()

#main code
def main():

    drift_nu = -0.2852
    drift_nu_gamma = 0.1923
    drift_gamma = -0.0309

    tau_nu = 1.0168
    tau_gamma = 8.7054

    beta = 0.3059

    t_index = np.arange(0,T_END,DT)

    z_target = np.zeros((M_DIM, int(TRIAL_DUR/DT)))
    z_target[0, int(T_INIT/DT):int((T_INIT+T_ON)/DT)] = 1

    z_stim = np.zeros((M_DIM, int(T_END/DT)))
    z_stim = mb.repmat(z_target, 1,25)

    gamma_init = np.zeros((M_DIM, 1))
    nu_init = np.zeros((M_DIM, 1))
    x_init = np.zeros((N_DIM,1))

    model = sensory_module(drift_nu, drift_nu_gamma, drift_gamma, tau_nu, tau_gamma, beta)

    compare_riccati_solutions(t_index, model)

    compare_init_FPRE(t_index, model)

    # sol= model.get_conn_matrix(initial = 'c')
    # K = sol.y    
    # x_response, nu_response, gamma_response = model.forward_sim(z_stim, x_init, nu_init, gamma_init, K)


    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10), sharex=True)
    # ax1.plot(t_index/60, x_response.T)
    # ax1.set_ylabel('Stage 1 activity')
    # ax1.set_xlim([0, T_END/60])

    # ax2.plot(t_index/60, nu_response.T)
    # ax2.set_ylabel('Stage 2 activity') 
    # ax2.set_xlim([0, T_END/60]) 

    # ax3.plot(t_index/60, gamma_response.T)
    # ax3.set_xlabel('Time in mins')
    # ax3.set_ylabel('Stage 3 activity')
    # ax3.set_xlim([0, T_END/60]) 

    plt.show() 

    

if __name__ == "__main__":
    main()
