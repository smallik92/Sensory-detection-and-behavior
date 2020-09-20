import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.linalg as splinalg



"""
Normative model of olfactory detection: Tracking a nominal latent space representation
"""
# Global parameters
DT = 0.01 #Time interval
T_END = 10 #Total time for experiment
T_INIT = 1 #Initial no stimulus period
T_ON = 4 #Stimulus presence period
T_OFF = 5 #Stimulus withdrawal period

# Note: T_END = T_INIT + T_ON + T_OFF

M = 2 #Dimension of latent space
N = 41 #Dimension of neural state space
C = 11 #Number of untuned neurons in the neural state space

ERROR_PENALTY = 10 #penalty for error in latent state
ENERGY_PENALTY = 2 #penalty for energy expenditure
SMOOTHNESS_PENALTY = 0.1 #penalty for large fluctuations

def weighting_Matrix(control_Num, overlap_Num, spread, intensity):
    stim_A_num = np.ceil(0.5*(control_Num + overlap_Num))
    stim_B_num = stim_A_num

    stim_A_mean = np.ceil(0.5*stim_A_num)
    stim_A_std = spread

    stim_B_mean = np.ceil((control_Num - stim_A_mean+1))
    stim_B_std = spread

    stim_indices = np.linspace(1,control_Num,num=control_Num)
    stim_A_weights = norm.pdf(stim_indices,loc=stim_A_mean,scale=stim_A_std)
    stim_B_weights = norm.pdf(stim_indices, loc=stim_B_mean, scale=stim_B_std)

    b_Matrix = np.vstack((stim_A_weights,stim_B_weights))
    b_Matrix = intensity*b_Matrix

    return b_Matrix

def get_optim_params(drift, b_Matrix, target_penalty, energy_penalty, smoothness_penalty):
    A = drift*(np.identity(M))
    A_modified = np.block([[A,b_Matrix],[np.zeros((N,M)),np.zeros((N,N))]])
    aux_eig_A = -1*(10**-10) #very small negative values on the diagonal for stability
    aux_A_matrix = aux_eig_A*(np.identity(M))

    A_final = splinalg.block_diag(A_modified,aux_A_matrix)
    B_final = np.block([[np.zeros((M,N))], [np.identity(N)],[np.zeros((M,N))]])

    R_control = energy_penalty*np.identity(N)
    Q = target_penalty*np.identity(M)
    Q_final = np.block([[Q, np.zeros((M, N)), -Q],[np.zeros((N, M)), R_control,np.zeros((N, M))], [-Q, np.zeros((M, N)), Q]])
    R_final = smoothness_penalty*np.identity(N)

    return A_final, B_final, Q_final, R_final

class olfactory_model(object):

    def __init__(self, drift, b_Matrix, target_penalty, energy_penalty, smoothness_penalty):
        self.drift = drift
        self.b_Matrix = b_Matrix
        self.target_penalty = target_penalty
        self.energy_penalty = energy_penalty
        self.smoothness_penalty = smoothness_penalty

    def solve_lqr_problem(self):
        A_final, B_final, Q_final, R_final = get_optim_params(self.drift, self.b_Matrix, self.target_penalty, self.energy_penalty, self.smoothness_penalty)

        #Solve the Continuous time Algebraic Riccati Equation
        S = np.matrix(splinalg.solve_continuous_are(A_final, B_final, Q_final, R_final))

        #compute the LQR gain
        K = np.matrix(splinalg.inv(R_final)*(B_final.T*S))
        eigVals, eigVecs = splinalg.eig(A_final-B_final*K)

        return K, S, eigVals

    def forward_sim(self, z_stim, nu_init, x_init):
        K, S, eigVals = self.solve_lqr_problem()

        #extract the interaction matrices
        W_nu = -K[:,:M]
        W_x = -K[:,M:N+M]
        W_z = -K[:,N+M:]
        W_vb = np.matmul(W_nu,self.b_Matrix)

        #create simulation data matrices
        x_Response = np.zeros((N,int(T_END/DT)))
        nu_Response = np.zeros((M,int(T_END/DT)))

        #Initialize the matrices for t=0
        x_Response[:,[0]] = x_init
        nu_Response[:,[0]] = nu_init

        #Euler forward method for integration
        for t in range(int(T_END/DT)-1):
            nu_Response[:,[t+1]] = nu_Response[:,[t]]+DT*(self.drift*nu_Response[:,[t]]+np.matmul(self.b_Matrix,x_Response[:,[t]]))
            x_Response[:,[t+1]] = x_Response[:,[t]]+DT*(np.matmul(W_nu,nu_Response[:,[t]])+np.matmul(W_x,x_Response[:,[t]])+np.matmul(W_z,z_stim[:,[t]]))

        return x_Response, nu_Response

def postprocessing(x_Response):

    stim_A_num = int(np.ceil((N+C)/2))
    pure_stim_A = stim_A_num - C

    x_blue_tuned = x_Response[0:pure_stim_A,:]
    x_untuned = x_Response[pure_stim_A:stim_A_num, :]
    x_red_tuned = x_Response[stim_A_num:,:]

    x_blue_mean = x_blue_tuned.mean(axis = 0)
    x_blue_std = x_blue_tuned.std(axis = 0)

    x_untuned_mean = x_untuned.mean(axis = 0)
    x_untuned_std = x_untuned.std(axis = 0)

    x_red_mean = x_red_tuned.mean(axis = 0)
    x_red_std = x_red_tuned.std(axis = 0)

    return x_blue_mean, x_blue_std, x_untuned_mean, x_untuned_std, x_red_mean, x_red_std

def main():

    drift = -0.25
    spread = 10
    intensity = 10
    #combinatorial encoding
    b_Matrix = weighting_Matrix(N, C, spread, intensity)

    #create an instance of the olfactory model
    create_olfactory_model = olfactory_model(drift, b_Matrix, ERROR_PENALTY, ENERGY_PENALTY, SMOOTHNESS_PENALTY)

    z_stim = np.zeros((M,int(T_END/DT)))
    z_stim[0, int(T_INIT/DT):int((T_INIT+T_ON)/DT)] = 1

    nu_init = np.zeros((M, 1))
    x_init = np.zeros((N,1))

    #forward simulate the olfactory_model
    x_Response, nu_Response = create_olfactory_model.forward_sim(z_stim, nu_init, x_init)

    #Post processing for plotting data
    x_blue_mean, x_blue_std, x_untuned_mean, x_untuned_std, x_red_mean, x_red_std = postprocessing(x_Response)
    t_index = np.arange(0,T_END,DT)

    assert(t_index.shape == x_blue_mean.shape)
    assert(t_index.shape == x_untuned_mean.shape)
    assert(t_index.shape == x_red_mean.shape)

    # Figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10))

    ax1.plot(t_index, x_blue_mean, color='blue')
    ax1.plot(t_index, x_untuned_mean, color='magenta')
    ax1.plot(t_index, x_red_mean, color='red')
    ax1.fill_between(t_index, x_blue_mean+x_blue_std, x_blue_mean-x_blue_std, facecolor='blue', alpha=0.5)
    ax1.fill_between(t_index, x_untuned_mean+x_untuned_std, x_untuned_mean-x_untuned_std, facecolor='magenta', alpha=0.5)
    ax1.fill_between(t_index, x_red_mean+x_red_std, x_red_mean-x_red_std, facecolor='red', alpha=0.5)
    ax1.set_xlim([0, T_END])

    ax2.plot(nu_Response[0,:], nu_Response[1,:], color= 'blue')
    ax2.plot(z_stim[0,int(T_INIT/DT)], z_stim[1, int(T_INIT/DT)], marker='o', color='black')
    ax2.text(0.8, -0.15, 'Nominal Representation', fontsize=12)
    ax2.set_xlim([-0.2, 1.2])
    ax2.set_ylim([-0.2, 1.2])
    ax2.grid()

    plt.show()

main()
